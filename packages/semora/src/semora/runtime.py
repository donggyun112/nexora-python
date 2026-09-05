"""Run a Pydantic AI agent under a run lease with the execution boundary attached.

The runtime owns what happens between attempts: the lease, parking a suspended round durably,
routing a person's answer back to it, queuing input for a live worker, and finishing an
interrupted round from what the dead worker committed.
"""

import asyncio
from collections.abc import AsyncIterator, Collection, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessagesTypeAdapter,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequestPart,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_core import to_jsonable_python
from semora_store import (
    Contended,
    ExecutionContext,
    ExecutionStore,
    ExecutionTransition,
    Transcript,
)

from .contracts import AgentSuspended, PendingInput, StopReason
from .controls import Controls, Ctx, controls_of
from .dispatch import Command, default_router
from .effects import PENDING_ROUND, Effects, Resumed, step_key
from .transcript import Branch, messages_at, messages_of

__all__ = ["ACTIVE_SUSPENSION", "AgentRuntime", "Outcome", "unanswered_tool_calls"]

ACTIVE_SUSPENSION = "agent:active-suspension"
"""Control key holding the run-level continuation of a parked round."""

_PART: TypeAdapter[ModelRequestPart] = TypeAdapter(ModelRequestPart)


def suspend_key(call_id: str) -> str:
    """Per-call index into the one run-level continuation."""
    return f"suspend:{call_id}"


@dataclass(frozen=True, slots=True)
class Outcome:
    """The terminal state of one attempt: what the model said and why the run stopped.

    A parked attempt is an outcome too, with `stop_reason == "suspended"` and the undecided
    `(pending_id, tool_call_id)` pairs in `pending`, in model order.
    """

    output: Any
    stop_reason: StopReason
    result: AgentRunResult[Any] | None = None
    pending: tuple[tuple[str, str], ...] = ()

    @property
    def suspended(self) -> bool:
        """Whether the attempt parked for a person's answer."""
        return self.stop_reason == "suspended"

    @property
    def pending_id(self) -> str | None:
        """The first undecided request's external id, or `None` when nothing is parked."""
        return self.pending[0][0] if self.pending else None

    def all_messages(self) -> list[ModelMessage]:
        """The model-visible history at the end of the attempt."""
        return self.result.all_messages() if self.result is not None else []


class AgentRuntime:
    """Attach a ledger and a transcript to agent runs. Without them a run is Pydantic AI's."""

    def __init__(
        self,
        execution_store: ExecutionStore | None = None,
        *,
        transcript: Transcript | None = None,
        lease_ttl: float = 60.0,
        retry_running: bool = False,
    ) -> None:
        """Configure durability for every run this runtime drives.

        Args:
            execution_store: The ledger. `None` records nothing: a crash may run a tool again,
                and a suspension has nowhere to park.
            transcript: The committed conversation. Required by `dispatch` and `committed_history`.
            lease_ttl: Seconds a run lease lives between renewals.
            retry_running: Whether a step that started and never reported may run again.
        """
        self.store = execution_store
        self.transcript = transcript
        self.lease_ttl = lease_ttl
        self.retry_running = retry_running

    async def run(
        self,
        branch_id: str | ExecutionContext,
        agent: Agent[Any, Any],
        prompt: str | None = None,
        *,
        controls: Controls | None = None,
        rules_version: str = "",
        prompt_id: str | None = None,
        conversation_id: str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        deps: Any = None,
        capabilities: Sequence[AbstractCapability[Any]] = (),
        _resumed: dict[str, Resumed] | None = None,
        **options: Any,
    ) -> Outcome:
        """Drive one attempt at a run under its lease.

        `options` reach Pydantic AI's `run` untouched (`model`, `usage_limits`, `model_settings`,
        `toolsets`, `event_stream_handler`, ...), so a harness that delegates to this agent with
        its own arguments still lands inside the boundary.

        Without `message_history` the attempt continues the committed conversation. A prompt
        aimed at a parked run is queued behind the park and the park is announced again.
        `conversation_id` names the conversation the branch belongs to: its transcript, and the
        scope of its ledger. One conversation holds many branches. It defaults to the branch id,
        and an `ExecutionContext` that already carries one keeps it.

        Raises:
            AgentSuspended: A gate parked the round. Answer it with `resume`.
            Contended: Another worker holds the run's lease, or an answered continuation still
                needs finalization. A prompt aimed at that continuation is queued before raising.
            Indeterminate: A step started and never reported, and repeating it was not allowed.
        """
        execution = _execution_context(branch_id, conversation_id)
        if self.store is not None and prompt is not None:
            active = await self._active(execution)
            if active is not None:
                await self.submit(
                    execution, PendingInput("user_prompt", UserPromptPart(prompt), prompt_id)
                )
                undecided = _undecided(active)
                if not undecided:
                    raise Contended(execution.branch_id)
                raise AgentSuspended(undecided[0][0], undecided[0][1], pending=undecided)
        async with self._lease(execution) as token:
            return await self._attempt(
                execution,
                token,
                agent,
                prompt,
                controls=controls,
                rules_version=rules_version,
                prompt_id=prompt_id,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                deps=deps,
                capabilities=capabilities,
                _resumed=_resumed,
                **options,
            )

    async def _attempt(
        self,
        execution: ExecutionContext,
        token: int,
        agent: Agent[Any, Any],
        prompt: str | None = None,
        *,
        controls: Controls | None = None,
        rules_version: str = "",
        prompt_id: str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        deps: Any = None,
        capabilities: Sequence[AbstractCapability[Any]] = (),
        _resumed: dict[str, Resumed] | None = None,
        _regate: Collection[str] = (),
        **options: Any,
    ) -> Outcome:
        """Drive an attempt while the caller owns the branch lease."""
        conversation = execution.conversation_id or execution.branch_id
        branch = await self._open(execution, conversation)
        if branch is not None:
            if message_history is None:
                message_history = branch.messages or None
            else:
                await branch.replace(list(message_history))
        store = self._store_for(execution)
        if prompt is not None and prompt_id is not None and store is not None:
            item = PendingInput("user_prompt", UserPromptPart(prompt), prompt_id)
            if not await store.enqueue_input(execution.branch_id, prompt_id, _encode(item)):
                prompt = None  # delivered once: the earlier attempt already carried it
            else:
                await store.admit_inputs(execution.branch_id, [prompt_id], token)
        # An explicit control plane wins; otherwise the agent's own methods are the policy.
        controls = controls if controls is not None else controls_of(agent)
        effects = Effects(
            store,
            execution.branch_id,
            token,
            controls=controls,
            retry_running=self.retry_running,
            rules_version=rules_version,
            subject=execution.subject or "",
            resumed=_resumed,
            inputs=_InputSession(store, execution.branch_id, token) if store else None,
            record=branch.append if branch is not None else None,
            regate=_regate,
        )
        # Semora's branch id is the durable coordinate every attempt shares; Pydantic AI stamps
        # each attempt with its own `run_id`, so what we hand it is the conversation.
        # Called on the base class on purpose: our `Agent` overrides `run` to come here.
        result: AgentRunResult[Any] = await Agent.run(
            agent,
            prompt,
            conversation_id=conversation,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            deps=deps,
            output_type=_with_deferred(options.pop("output_type", None) or agent.output_type),
            capabilities=[effects, *(capabilities or ())],
            **options,
        )
        if branch is not None:
            await branch.replace(result.all_messages())
        if isinstance(result.output, DeferredToolRequests):
            await self._park(execution, conversation, effects, result, token)
        stop_reason = effects.stop_reason or "completed"
        if branch is not None:
            await branch.writer.closed(
                stop_reason, tool_calls=len(effects.calls_made), usage=_usage(result)
            )
        return Outcome(result.output, stop_reason, result)

    async def resume(
        self,
        branch_id: str | ExecutionContext,
        pending_id: str,
        answer: dict[str, Any],
        agent: Agent[Any, Any],
        *,
        controls: Controls | None = None,
        rules_version: str = "",
        conversation_id: str | None = None,
        deps: Any = None,
    ) -> Outcome:
        """Route an answer by its suspension's external `pending_id`.

        A batch park executes nothing until every parked call is decided: an answer for one of
        several parked calls is recorded durably and `AgentSuspended` is raised again with the
        still-undecided requests. The final answer revalidates and finishes every parked call in
        model order, and only then does the run continue. The answer is an *input* to `on_resume`,
        never the decision: current policy re-decides.

        A branch that ran inside a conversation parked inside it; name the same conversation here,
        by `conversation_id` or on the `ExecutionContext`, or the park is not found.
        """
        execution = _execution_context(branch_id, conversation_id)
        store = self._require_store(execution)
        async with self._lease(execution) as token:
            active = await self._active(execution)
            parked = _decode_parked(active) if active is not None else []
            by_pending = {str(request["pending_id"]): call for call, request in parked}
            if active is None or pending_id not in by_pending:
                raise LookupError(f"no active suspension for pending id {pending_id!r}")
            answers = {**active["answers"], by_pending[pending_id].tool_call_id: answer}
            await store.write_control(
                execution.branch_id, ACTIVE_SUSPENSION, {**active, "answers": answers}, token
            )
            undecided = _undecided({**active, "answers": answers})
            if undecided:
                raise AgentSuspended(undecided[0][0], undecided[0][1], pending=undecided)
            return await self._finalize(
                execution, token, agent, active, answers, parked, controls, rules_version, deps
            )

    async def recover(
        self,
        branch_id: str | ExecutionContext,
        agent: Agent[Any, Any],
        history: Sequence[ModelMessage],
        *,
        controls: Controls | None = None,
        rules_version: str = "",
        conversation_id: str | None = None,
        deps: Any = None,
    ) -> Outcome:
        """Finish an interrupted run from the transcript the dead worker committed.

        The history ends with the model's tool calls. Pydantic AI resumes from that round; the
        boundary answers committed calls from the record and runs the rest. The model is not asked
        again for calls it already made. A round parked before the crash stays parked under its
        original pending ids — re-gating would orphan every answer in flight.
        """
        execution = _execution_context(branch_id, conversation_id)
        store = self._store_for(execution)
        async with self._lease(execution) as token:
            if store is not None:
                active = await self._active(execution)
                if active is not None:
                    undecided = _undecided(active)
                    if undecided:
                        raise AgentSuspended(undecided[0][0], undecided[0][1], pending=undecided)
                    return await self._finalize(
                        execution,
                        token,
                        agent,
                        active,
                        active["answers"],
                        _decode_parked(active),
                        controls,
                        rules_version,
                        deps,
                    )
                recorded = await store.read(execution.branch_id, PENDING_ROUND)
                if recorded.status == "done":
                    ids = {call["id"] for call in recorded.value["calls"]}
                    if any(call.tool_call_id not in ids for call in unanswered_tool_calls(history)):
                        raise ValueError("agent history does not match the recorded pending round")
            return await self._attempt(
                execution,
                token,
                agent,
                message_history=history,
                controls=controls,
                rules_version=rules_version,
                deps=deps,
            )

    async def fork(
        self,
        source: str | ExecutionContext,
        at: str | None,
        target: str | ExecutionContext,
        agent: Agent[Any, Any],
        prompt: str | None = None,
        *,
        history: Sequence[ModelMessage] | None = None,
        regate: bool = False,
        controls: Controls | None = None,
        rules_version: str = "",
        source_conversation_id: str | None = None,
        conversation_id: str | None = None,
        deps: Any = None,
        **options: Any,
    ) -> Outcome:
        """Start `target` from one point of `source`'s transcript.

        `at` is a transcript entry uuid; `None` forks from the tip of the active branch. A host
        that keeps its own coordinates passes the messages as `history` instead, and no
        transcript is read. That history becomes the new run's, and `prompt` follows it.

        Calls in that history the source run finished are copied into the new run's ledger, so
        they replay instead of running again; the new run's `post_tool_use` still sees each of
        them once. By default no gate is asked about them. `regate=True` asks the new run's
        `pre_tool_use` first, so a changed policy can deny or park a call whose effect the
        source already made, and only a `Continue` replays it. A call the source started and
        never reported is copied as started: the new run inherits the doubt, and `retry_running`
        decides. Anything the source never began runs fresh. The source run is read, never
        written.
        """
        src = _execution_context(source, source_conversation_id)
        dst = _execution_context(target, conversation_id)
        if history is None:
            if self.transcript is None:
                raise TypeError("fork without history= requires AgentRuntime(transcript=...)")
            entries = await self.transcript.for_execution(src).read(
                src.conversation_id or src.branch_id
            )
            history = messages_at(entries, at) if at is not None else messages_of(entries)
        history = list(history)
        # Each end of the fork is read and written through its own conversation's view, so a
        # record copies only where both contexts can see it.
        from_store, into_store = self._store_for(src), self._store_for(dst)
        async with self._lease(dst) as token:
            copied: list[str] = []
            if from_store is not None and into_store is not None:
                for call in unanswered_tool_calls(history):
                    key = step_key(call.tool_call_id)
                    record = await from_store.read(src.branch_id, key)
                    if record.status == "absent" or not await into_store.start(
                        dst.branch_id, key, token
                    ):
                        continue
                    if record.status == "done":
                        await into_store.finish_effect(dst.branch_id, key, record.value, token)
                        copied.append(call.tool_call_id)
            return await self._attempt(
                dst,
                token,
                agent,
                prompt,
                controls=controls,
                rules_version=rules_version,
                message_history=history,
                deps=deps,
                _regate=copied if regate else (),
                **options,
            )

    async def submit(
        self,
        branch_id: str | ExecutionContext,
        item: PendingInput,
        *,
        conversation_id: str | None = None,
    ) -> PendingInput:
        """Durably queue one input for the branch's next model boundary.

        Unleased on purpose: enqueueing is append-only and keyed by the input's own id, so it
        needs no exclusion, and a live branch holds the lease itself.
        """
        execution = _execution_context(branch_id, conversation_id)
        normalized = PendingInput(item.kind, item.part, item.origin_id or str(uuid4()))
        assert normalized.origin_id is not None
        await self._require_store(execution).enqueue_input(
            execution.branch_id, normalized.origin_id, _encode(normalized)
        )
        return normalized

    async def dispatch(
        self,
        branch_id: str | ExecutionContext,
        agent: Agent[Any, Any],
        command: Command,
        *,
        controls: Controls | None = None,
        **options: Any,
    ) -> Any:
        """Route a host command to the transition the run's durable state allows.

        One state-aware entry point over ``run``/``resume``/``recover``/``submit`` so every
        adapter shares a single transition table. It needs both durable collaborators: state
        comes from the execution store, history from the transcript.
        """
        if self.store is None or self.transcript is None:
            raise TypeError("dispatch requires AgentRuntime(execution_store=..., transcript=...)")
        # The state read and the transition it picks must look at the same conversation's ledger.
        execution = _execution_context(branch_id, options.get("conversation_id"))
        return await default_router().dispatch(
            self, execution, agent, command, controls=controls, **options
        )

    async def pending(
        self, branch_id: str | ExecutionContext, conversation_id: str | None = None
    ) -> list[tuple[str, str]]:
        """The undecided `(pending_id, tool_call_id)` pairs of a parked branch, in model order."""
        if self.store is None:
            return []
        active = await self._active(_execution_context(branch_id, conversation_id))
        return _undecided(active) if active is not None else []

    async def state(
        self, branch_id: str | ExecutionContext, conversation_id: str | None = None
    ) -> str:
        """Name the run's durable state from one observation.

        A parked run reports its continuation state (``waiting``/``resuming``). Otherwise the
        transcript's run record names it: ``fresh`` (never ran), ``completed`` (ended), or
        ``interrupted`` (an open round — a crash, or a run another worker is still driving;
        only a lease attempt can tell those apart). Without a transcript an unparked run is
        just ``idle``.
        """
        execution = _execution_context(branch_id, conversation_id)
        if self.store is not None:
            active = await self._active(execution)
            if active is not None:
                return str(active["state"])
        if self.transcript is None:
            return "idle"
        record = await self.transcript.for_execution(execution).read_branch(execution.branch_id)
        if record is None:
            return "fresh"
        return "completed" if record.get("ended_at") is not None else "interrupted"

    async def committed_history(
        self, branch_id: str | ExecutionContext, conversation_id: str | None = None
    ) -> list[ModelMessage]:
        """The committed model history of the run's conversation."""
        if self.transcript is None:
            raise TypeError("committed_history requires AgentRuntime(transcript=...)")
        execution = _execution_context(branch_id, conversation_id)
        entries = await self.transcript.for_execution(execution).read(
            execution.conversation_id or execution.branch_id
        )
        return messages_of(entries)

    # ── suspension ────────────────────────────────────────────────────────────

    async def _park(
        self,
        execution: ExecutionContext,
        conversation: str,
        effects: Effects,
        result: AgentRunResult[Any],
        token: int,
    ) -> None:
        """Persist every parked call of the round in one transition, then announce it."""
        requests: DeferredToolRequests = result.output
        if requests.calls:
            raise RuntimeError("only a pre_tool_use permission gate may suspend a tool request")
        parked = [(call, requests.metadata[call.tool_call_id]) for call in requests.approvals]
        ids = [str(request.get("pending_id") or "") for _, request in parked]
        if len(set(ids)) != len(ids):
            # Before persistence: a park with two calls under one id could never be answered apart.
            raise ValueError(f"a parked round reuses a pending_id: {ids}")
        messages = result.all_messages()
        completed = _round_results(messages)
        if effects.controls is not None:
            ctx = Ctx(turn=effects.turn, messages=messages, subject=effects.subject)
            for call, request in parked:
                await effects.controls.on_suspend(ctx, call, request, messages, completed)
        pending = [(str(request["pending_id"]), call.tool_call_id) for call, request in parked]
        store = self._store_for(execution)
        if store is not None:
            first = parked[0][0].tool_call_id
            continuation = {
                "origin": "pre_tool_use",
                "calls": [
                    {
                        "call": {
                            "tool_name": call.tool_name,
                            "args": call.args_as_dict(),
                            "tool_call_id": call.tool_call_id,
                        },
                        "request": request,
                    }
                    for call, request in parked
                ],
                "messages": to_jsonable_python(messages),
                "completed": completed,
                "rules_version": effects.rules_version,
                "turn": effects.turn,
                "subject": effects.subject,
            }
            writes: dict[str, Any] = {
                suspend_key(call.tool_call_id): {"active_call_id": first} for call, _ in parked
            }
            writes[ACTIVE_SUSPENSION] = {
                "state": "waiting",
                "conversation_id": conversation,
                "call_id": first,
                "continuation": continuation,
                "call_ids": [call.tool_call_id for call, _ in parked],
                "answers": {},
            }
            await store.commit_transition(
                execution.branch_id, ExecutionTransition(controls=writes), token
            )
        raise AgentSuspended(pending[0][0], pending[0][1], pending=pending)

    async def _finalize(
        self,
        execution: ExecutionContext,
        token: int,
        agent: Agent[Any, Any],
        active: dict[str, Any],
        answers: dict[str, dict[str, Any]],
        parked: list[tuple[ToolCallPart, dict[str, Any]]],
        controls: Controls | None,
        rules_version: str,
        deps: Any,
    ) -> Outcome:
        """Idempotently finish a fully answered continuation after resume or recovery."""
        store = self._require_store(execution)
        continuation = active["continuation"]
        history = ModelMessagesTypeAdapter.validate_python(continuation["messages"])
        approvals: dict[str, bool | ToolApproved | ToolDenied] = {}
        resumed: dict[str, Resumed] = {}
        for call, request in parked:
            answer = answers[call.tool_call_id]
            if answer.get("type") == "error":
                approvals[call.tool_call_id] = ToolDenied(str(answer.get("message") or "denied"))
            else:
                # An answer may carry `args`: the person approved the call with these arguments
                # instead. Pydantic AI validates and runs them; `on_resume` sees them as the call.
                override = answer.get("args")
                approvals[call.tool_call_id] = ToolApproved(
                    override_args=dict(override) if isinstance(override, dict) else None
                )
            resumed[call.tool_call_id] = Resumed(answer, request, continuation["rules_version"])
        await store.write_control(
            execution.branch_id,
            ACTIVE_SUSPENSION,
            {**active, "state": "resuming", "answers": answers},
            token,
        )
        # The park remembers its conversation, so a caller that named only the branch rejoins it.
        # The transcript's fallback, the branch id itself, is not a conversation anyone named and
        # must not start scoping the ledger halfway through a branch.
        remembered = active.get("conversation_id")
        rejoined = _execution_context(
            execution, None if remembered == execution.branch_id else remembered
        )
        outcome = await self._attempt(
            rejoined,
            token,
            agent,
            message_history=history,
            deferred_tool_results=DeferredToolResults(approvals=approvals),
            controls=controls,
            rules_version=rules_version,
            deps=deps,
            _resumed=resumed,
        )
        await store.write_control(
            execution.branch_id,
            ACTIVE_SUSPENSION,
            {"state": "completed", "call_id": active["call_id"], "call_ids": active["call_ids"]},
            token,
        )
        return outcome

    async def _active(self, execution: ExecutionContext) -> dict[str, Any] | None:
        """The live continuation, or None when nothing is parked."""
        record = await self._require_store(execution).read(execution.branch_id, ACTIVE_SUSPENSION)
        if record.status != "done" or record.value.get("state") not in {"waiting", "resuming"}:
            return None
        return dict(record.value)

    # ── lease and transcript ──────────────────────────────────────────────────

    @asynccontextmanager
    async def _lease(self, execution: ExecutionContext) -> AsyncIterator[int]:
        store = self._store_for(execution)
        if store is None:
            yield 0
            return
        branch_id, owner = execution.branch_id, uuid4().hex
        token = await store.acquire(branch_id, owner, self.lease_ttl)
        if not token:
            raise Contended(branch_id)
        renewal = asyncio.create_task(self._renew(store, branch_id, owner))
        try:
            yield token
        finally:
            renewal.cancel()
            await store.release(branch_id, owner)

    async def _renew(self, store: ExecutionStore, branch_id: str, owner: str) -> None:
        while True:
            await asyncio.sleep(self.lease_ttl / 3)
            # A lost lease is not handled here: the next write carries a stale token and is Fenced.
            await store.acquire(branch_id, owner, self.lease_ttl)

    async def _open(self, execution: ExecutionContext, conversation: str) -> Branch | None:
        if self.transcript is None:
            return None
        return await Branch.open(
            self.transcript, conversation, execution.branch_id, context=execution
        )

    def _store_for(self, execution: ExecutionContext) -> ExecutionStore | None:
        """The ledger as this execution sees it: its conversation's view when it names one."""
        return self.store.for_execution(execution) if self.store is not None else None

    def _require_store(self, execution: ExecutionContext) -> ExecutionStore:
        store = self._store_for(execution)
        if store is None:
            raise TypeError("this needs a ledger; pass execution_store=")
        return store


class _InputSession:
    """Admit the durable input queue through one leased attempt."""

    def __init__(self, store: ExecutionStore, branch_id: str, token: int) -> None:
        self._store = store
        self._branch_id = branch_id
        self._token = token
        self._seen: set[str] = set()

    async def claim(self, represented: set[str]) -> list[PendingInput]:
        claimed: list[PendingInput] = []
        for record in await self._store.list_inputs(self._branch_id):
            # ponytail: an admitted input is trusted as delivered even if history lost it;
            # semora reclaims those by message id. Reinstate if transcripts prove lossy.
            if (
                record.status in {"discarded", "admitted"}
                or record.input_id in represented
                or record.input_id in self._seen
            ):
                continue
            await self._store.claim_input(self._branch_id, record.input_id, self._token)
            self._seen.add(record.input_id)
            claimed.append(_decode(record.value))
        return claimed

    async def admit(self, items: list[PendingInput]) -> None:
        ids = [item.origin_id for item in items if item.origin_id is not None]
        if ids:
            await self._store.admit_inputs(self._branch_id, ids, self._token)

    async def discard(self, items: list[PendingInput]) -> None:
        ids = [item.origin_id for item in items if item.origin_id is not None]
        if ids:
            await self._store.discard_inputs(self._branch_id, ids, self._token)


def unanswered_tool_calls(history: Sequence[ModelMessage]) -> list[ToolCallPart]:
    """Calls in the latest model response that have no following tool return.

    Also the answer to "where does a run resume from here": a history that still owes a tool
    answer resumes at the gate for that round, and one that does not hands the conversation
    back to the model.
    """
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, ModelResponse):
            continue
        answered = {
            part.tool_call_id
            for later in history[index + 1 :]
            for part in later.parts
            if isinstance(part, ToolReturnPart)
        }
        return [
            part
            for part in message.parts
            if isinstance(part, ToolCallPart) and part.tool_call_id not in answered
        ]
    return []


def _round_results(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Results of the calls that completed in the latest round, in model order."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], ModelResponse):
            return [
                {"id": part.tool_call_id, "result": to_jsonable_python(part.content)}
                for later in messages[index + 1 :]
                for part in later.parts
                if isinstance(part, ToolReturnPart)
            ]
    return []


def _decode_parked(active: dict[str, Any]) -> list[tuple[ToolCallPart, dict[str, Any]]]:
    return [
        (
            ToolCallPart(
                tool_name=entry["call"]["tool_name"],
                args=entry["call"]["args"],
                tool_call_id=entry["call"]["tool_call_id"],
            ),
            dict(entry["request"]),
        )
        for entry in active["continuation"]["calls"]
    ]


def _undecided(active: dict[str, Any]) -> list[tuple[str, str]]:
    answers = active["answers"]
    return [
        (str(request["pending_id"]), call.tool_call_id)
        for call, request in _decode_parked(active)
        if call.tool_call_id not in answers
    ]


def _with_deferred(spec: Any) -> list[Any]:
    """The requested output types plus the one a parked round ends with."""
    items = list(spec) if isinstance(spec, list | tuple) else [spec]
    if DeferredToolRequests not in items:
        items.append(DeferredToolRequests)
    return items


def _usage(result: AgentRunResult[Any]) -> dict[str, dict[str, Any]]:
    """Token counts keyed by the model that answered, in the transcript's vocabulary."""
    usage = result.usage
    counts = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
    if not counts["total_tokens"]:
        return {}
    model = next(
        (m.model_name for m in reversed(result.all_messages()) if isinstance(m, ModelResponse)),
        None,
    )
    return {model or "": counts}


def _encode(item: PendingInput) -> dict[str, Any]:
    """Serialize one queue item without teaching the execution ledger about messages."""
    return {"kind": item.kind, "part": to_jsonable_python(item.part), "origin_id": item.origin_id}


def _decode(payload: dict[str, Any]) -> PendingInput:
    return PendingInput(
        str(payload["kind"]), _PART.validate_python(payload["part"]), payload.get("origin_id")
    )


def _execution_context(
    value: str | ExecutionContext, conversation_id: str | None = None
) -> ExecutionContext:
    """The context as given, with a conversation named separately filled in if it had none."""
    if not isinstance(value, ExecutionContext):
        return ExecutionContext(branch_id=value, conversation_id=conversation_id)
    if value.conversation_id is None and conversation_id is not None:
        return replace(value, conversation_id=conversation_id)
    return value
