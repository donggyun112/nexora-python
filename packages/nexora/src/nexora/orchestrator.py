"""Durable execution, policy suspension, and recovery around an agent loop.

The while engine owns the small deterministic loop: ``model -> tool round -> model``. The
orchestrator owns the boundaries where middleware and recovery matter:

* `execute_round` saves the model-issued call order, applies the shared controls, and records every
  allowed call by its call id;
* `recover_pending` restores `done` results, retries `running` calls with the same idempotency key,
  executes `absent` calls in order, and reconstructs the `ToolMessage`s without replaying the
  model turn;
* `signal`/`suspend` end an attempt while a policy or human answer is outstanding;
* the durable input queue admits prompts, steer messages, background results, and resume answers
  through one ordered boundary before they enter model context;
* `run` provides the same durable-step primitive for effects outside the agent loop.

The agent transcript, input queue, and execution ledger are separate on purpose. The transcript
persists model-visible messages; the queue persists inputs waiting for admission; the ledger
persists the small pending-call record and each effect's state. Copying the growing transcript at
every tool boundary would erase the while loop's cost advantage and make persistence quadratic.

The ledger itself lives in `nexora.store` and is re-exported here, where every caller already
looks for it. It is the only layer in this file's neighbourhood that knows nothing about agents,
which is why it is the one that got its own module: a `StepLog` implementation should not have to
import a transcript type to store opaque values.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, NamedTuple, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from nexora_store import (
    Contended,
    Fenced,
    Indeterminate,
    InputRecord,
    MemorySteps,
    Step,
    StepLog,
)

from .contracts.events import EventType, RuntimeEvents
from .contracts.types import (
    Aborted,
    BaseMessage,
    Emit,
    OnSuspend,
    PendingInput,
    ToolCall,
    Tools,
)
from .controls import Continue, Controls, Ctx, Deny, ResumeInput, Suspend
from .history import (
    decode_pending_input,
    encode_continuation,
    encode_pending_input,
    suspend_history_snapshot,
)
from .tools import (
    Concurrent,
    Resolved,
    RoundSuspended,
    Stepped,
    absorb_round,
    execute_calls,
    record_resolved,
    require_call_ids,
    tool_payload,
    tool_result,
)

__all__ = [
    "AgentAborted",
    "AgentFailed",
    "AgentSuspended",
    "Contended",
    "Fenced",
    "Indeterminate",
    "InputRecord",
    "MemorySteps",
    "Orchestrator",
    "RecoveredTools",
    "Step",
    "StepLog",
    "Suspended",
    "run_agent",
]


class AgentFailed(Exception):
    """A run ended in `error`. Raised so a step cannot record a failure as an outcome."""

    def __init__(self, message: str, partial: str = "") -> None:
        super().__init__(message)
        self.partial = partial
        """Text the dying turn had streamed. Not an answer — nobody knows whether that turn was
        about to call a tool — but it is what a person already read, and the only copy of it."""


async def run_agent(
    events: AsyncIterator[dict[str, Any]],
    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Drive an agent to its end and return the outcome, so it can be one step.

        plan = await o.run("draft", lambda: run_agent(react_loop(model, tools, goal)))

    The loop yields a stream because a caller may want to watch text arrive; a workflow wants
    the answer. This is the adapter between the two, and it is the whole of what a durable
    orchestrator needs from an agent — no hooks, no middleware, one value.

    Only a *finished* run is an outcome. Everything else raises, because a step records whatever
    it returns and memoising a non-answer freezes it:

    * `error` — a recorded failure replays as failure and the retry never happens.
    * `suspended` — a recorded suspension replays as suspended and the agent never continues,
      however many approvals arrive.
    * `stop_reason == "aborted"` — an interruption is not an answer. Recorded, every replay
      returns "we were interrupted", which is the one thing a resume exists to get past.

    `stop_reason == "policy"` does return: a supervisor deciding to stop is a decision, not an
    interruption, and repeating the run would just reach the same decision.
    """
    async for event in events:
        if on_event is not None:
            await on_event(event)
        if event["type"] == "error":
            raise AgentFailed(event["message"], event.get("partial", ""))
        if event["type"] == "suspended":
            raise AgentSuspended(event["pending_id"], event["tool_call_id"])
        if event["type"] == "done":
            if event.get("stop_reason") == "aborted":
                raise AgentAborted(event.get("content", ""))
            return event
    raise AgentFailed("the agent produced no terminal event")


class AgentAborted(Exception):
    """The run was interrupted. Raised so a step does not record an interruption as its result.

    Distinct from `AgentFailed` because the cause is outside the agent — a deploy, a shutdown, an
    operator — and a supervisor usually wants to retry it rather than report it.
    """

    def __init__(self, partial: str = "") -> None:
        super().__init__("the run was aborted")
        self.partial = partial
        """Whatever text had streamed before the interruption. Not an answer; sometimes a clue."""


class Suspended(Exception):
    """A signal has no answer yet, so this attempt stops here.

    Not a failure. The caller records it as "waiting", and resuming means calling the workflow
    again once `resolve` has written the answer.
    """

    def __init__(self, signal: str) -> None:
        super().__init__(f"waiting for signal: {signal}")
        self.signal = signal


class AgentSuspended(Suspended):
    """A tool inside the agent asked for an answer, so the run stopped mid-round.

    A `Suspended`, because it is the same thing from the workflow's side: this attempt ends,
    nothing is recorded, and it resumes when the answer exists.

    The orchestrator persists the continuation before this reaches the agent driver. Once an
    approval arrives, a new attempt runs the answer-aware `on_resume` control under current policy,
    then executes the call only when its durable step is still absent. No generator, worker, or
    lease stays alive while the approval is outstanding.
    """

    def __init__(self, pending_id: str, tool_call_id: str) -> None:
        super().__init__(pending_id)
        self.pending_id = pending_id
        self.tool_call_id = tool_call_id


class RecoveredTools(NamedTuple):
    """A persisted assistant tool round completed without replaying its model call."""

    history: list[BaseMessage]
    """The supplied history plus reconstructed `ToolMessage`s, ready for `react_loop`."""
    resolved: list[Resolved]
    completed: list[dict[str, Any]]
    suspended: tuple[ToolCall, dict[str, Any]] | None


class Orchestrator:
    """One attempt at a workflow. Same `run_id` to resume; hold the lease while attempting.

        async with Orchestrator("patient-7", log, owner=worker_id) as o:
            ...

    Entering acquires the run's lease and leaving releases it. Without it two workers replaying
    the same run both see every step as `absent` and both do the effects — the failure a per-step
    record cannot catch, because neither worker is wrong about what it read.

    The lease is renewed at every step boundary and every write carries its fencing token. The
    renewal is for availability: a run whose steps are each shorter than the TTL never lets the
    lease lapse. The token is for correctness, and it is the one that matters — a worker can stall
    inside one long step past its TTL, and no renewal schedule prevents that. When it wakes up its
    writes are `Fenced`.

    ponytail: renewal happens on `run`/`signal`, not on a background heartbeat. A single step that
    outlives the TTL still lets another worker in; the token makes that harmless rather than
    impossible. A heartbeat task is the upgrade if long single steps become normal, and it changes
    nothing about the fencing.
    """

    def __init__(
        self,
        run_id: str,
        log: StepLog | None = None,
        *,
        owner: str = "local",
        ttl: float = 60.0,
        emit: Emit | None = None,
        on_suspend: OnSuspend | None = None,
        on_agent_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        rules_version: str = "",
    ) -> None:
        self.run_id = run_id
        self.owner = owner
        self._ttl = ttl
        self._log = log if log is not None else MemorySteps()
        self._seen: set[str] = set()
        self._token = 0
        """0 until a lease is taken, which is how a single-process caller runs without one."""
        self._emit = emit
        self._on_suspend = on_suspend
        self._on_agent_event = on_agent_event
        self._rules_version = rules_version
        self._seen_inputs: set[str] = set()
        self.events = RuntimeEvents(emit)
        """Session/integration events, backed by the same publisher as agent execution."""

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """The publisher passed to an engine so every event uses this orchestrator's stream."""
        if self._emit is not None:
            await self._emit(event_type, payload)

    async def enqueue_input(self, item: PendingInput) -> PendingInput:
        """Append one external input and return its normalized, retry-stable envelope."""
        normalized = self._normalize_input(item)
        input_id = normalized.origin_id
        assert input_id is not None
        inserted = await self._log.enqueue_input(
            self.run_id, input_id, encode_pending_input(normalized)
        )
        if inserted:
            await self._announce_input(normalized)
        return normalized

    async def commit_transition_inputs(
        self,
        items: list[PendingInput],
        steps: dict[str, Any],
    ) -> list[PendingInput]:
        """Atomically order protocol-closing inputs before the user input replacing them."""
        normalized = [self._normalize_input(item) for item in items]
        encoded = [
            (item.origin_id, encode_pending_input(item))
            for item in normalized
            if item.origin_id is not None
        ]
        inserted = await self._log.commit_transition(
            self.run_id,
            steps,
            [(input_id, value) for input_id, value in encoded if input_id is not None],
            self._token,
        )
        for item in normalized:
            if item.origin_id in inserted:
                await self._announce_input(item)
        return normalized

    async def active_continuation(self) -> dict[str, Any] | None:
        """The one run-level continuation state, or None when the run is not parked."""
        record = await self._log.read(self.run_id, _active_suspension_key())
        return record.value if record.status == "done" and isinstance(record.value, dict) else None

    async def cancel_and_switch(
        self,
        call: ToolCall,
        continuation: dict[str, Any],
        cancellation_result: dict[str, Any],
        cancellations: list[PendingInput],
        replacement: PendingInput,
    ) -> list[PendingInput]:
        """Close an unanswered model request and order the replacing user input after it."""
        call_id = call["id"] or ""
        steps: dict[str, Any] = {
            _active_suspension_key(): {
                "state": "switching",
                "call_id": call_id,
                "continuation": continuation,
            }
        }
        # Every suspension is now a pre-effect permission wait. Finishing it as cancelled prevents
        # a stale approval from executing it later with the same idempotency key.
        effect = await self._log.read(self.run_id, call_id)
        if effect.status == "absent":
            steps[call_id] = cancellation_result
        elif not (
            effect.status == "done"
            and isinstance(effect.value, dict)
            and effect.value.get("code") == "cancelled"
        ):
            raise RuntimeError(f"cannot cancel effect {call_id!r}: ledger says {effect.status!r}")
        return await self.commit_transition_inputs([*cancellations, replacement], steps)

    async def continue_with_input(
        self,
        call_id: str,
        continuation: dict[str, Any],
        item: PendingInput,
    ) -> PendingInput:
        """Atomically bind a suspension answer to the continuation it will resume."""
        [normalized] = await self.commit_transition_inputs(
            [item],
            {
                _active_suspension_key(): {
                    "state": "resuming",
                    "call_id": call_id,
                    "continuation": continuation,
                }
            },
        )
        return normalized

    async def complete_continuation(self, call_id: str) -> None:
        """Clear only the switching/resuming continuation this successful attempt consumed."""
        active = await self.active_continuation()
        if (
            active is not None
            and active.get("call_id") == call_id
            and active.get("state") in {"switching", "resuming"}
        ):
            await self._log.finish(
                self.run_id, _active_suspension_key(), None, self._token
            )

    def _normalize_input(self, item: PendingInput) -> PendingInput:
        input_id = item.origin_id or str(uuid4())
        message = item.message.model_copy(update={"id": input_id})
        return PendingInput(item.kind, message, input_id)

    async def _announce_input(self, item: PendingInput) -> None:
        """Announce that an input reached the inbox. Never its text.

        This fires at submission, which is *before* the run's `on_inputs` screens can mask
        anything, so any content here is the pre-mask original — and it would sit in the audit log
        forever, which is exactly what `Controls.on_inputs` promises cannot happen. The admitted
        text is published by `CONTEXT_INJECTED` after screening; a consumer that wants to show a
        prompt reads it there, and gets the version the model actually saw.
        """
        if self._emit is not None and item.kind in {"user_prompt", "user_steer"}:
            await self._emit(
                EventType.USER_PROMPT_SUBMIT,
                {"input_id": item.origin_id, "source": item.kind},
            )

    async def claim_inputs(self, represented: set[str] | None = None) -> list[PendingInput]:
        """Claim missing inputs in dependency order, preserving arrival order among peers.

        `represented` is the set of input ids already present in the caller's transcript. Ids and
        not messages: the queue has no business reading a conversation, and the one caller that
        holds one is the agent-facing facade.
        """
        represented = represented or set()
        claimed: list[PendingInput] = []
        decoded = [
            (record, decode_pending_input(record.value))
            for record in await self._log.list_inputs(self.run_id)
        ]
        # Protocol-closing answers must precede user input that arrived while a tool call was
        # parked, even though that user input reached the inbox first. Preserve order within both
        # groups; this is dependency ordering, not arbitrary priority.
        decoded.sort(
            key=lambda pair: (
                pair[1].kind not in {"resume_result", "cancelled_tool_result"},
                pair[0].sequence,
            )
        )
        for record, item in decoded:
            if record.input_id in represented or record.input_id in self._seen_inputs:
                continue
            await self._log.claim_input(self.run_id, record.input_id, self._token)
            self._seen_inputs.add(record.input_id)
            claimed.append(item)
        return claimed

    async def admit_inputs(self, inputs: list[PendingInput]) -> None:
        """Commit queue admission after the loop has appended the messages."""
        input_ids = [item.origin_id for item in inputs if item.origin_id is not None]
        if input_ids:
            await self._log.admit_inputs(self.run_id, input_ids, self._token)

    async def __aenter__(self) -> "Orchestrator":
        self._token = await self._log.acquire(self.run_id, self.owner, self._ttl)
        if not self._token:
            raise Contended(self.run_id)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._log.release(self.run_id, self.owner)
        self._token = 0

    async def _renew(self) -> None:
        """Extend the lease at a step boundary, if one is held at all."""
        if self._token:
            self._token = await self._log.acquire(self.run_id, self.owner, self._ttl) or self._token

    async def run(self, step: str, fn: Callable[[], Awaitable[Any] | Any]) -> Any:
        """Do this once, ever. On replay the recorded value comes back and `fn` is not called.

        `step` is the identity of the effect, so it is also its idempotency key — for a tool call
        that means the `call_id`, which is what ADR-002 already made the key.

        The intent is written **before** the effect and the result after, so a crash in between is
        visible as `running` rather than as "never happened". That case raises `Indeterminate`
        instead of re-running, because only the caller knows whether the effect is safe to repeat.
        """
        self._claim(step)
        await self._renew()
        record = await self._log.read(self.run_id, step)
        if record.status == "done":
            return record.value
        if record.status == "running":
            raise Indeterminate(self.run_id, step)

        await self._log.start(self.run_id, step, self._token)
        try:
            result = fn()
            if isinstance(result, Awaitable):
                result = await result
        except BaseException:
            # A raise and a crash are not the same fact. A raise means the step *reported* that it
            # did not complete, so the intent is cleared and the next attempt retries. A crash
            # leaves the intent standing, and that is what `Indeterminate` is for.
            #
            # The obligation this puts on a step function: raise only when a retry is safe. A
            # function that performs an external effect and *then* fails must swallow, or record
            # its own handle, or say so in its return value — it must not raise, because raising
            # here means "nothing happened".
            await self._clear(step)
            raise
        await self._log.finish(self.run_id, step, result, self._token)
        return result

    async def execute_round(
        self,
        tools: Tools,
        calls: list[ToolCall],
        aborted: Aborted,
        emit: Emit | None = None,
        turn: int = 0,
        controls: Controls | None = None,
        ctx: Ctx | None = None,
    ) -> list[Resolved]:
        """Own one live while-loop tool round.

        The engine still owns `model -> tools -> model`; this method owns everything between the
        model requesting calls and the engine receiving their results. Every allowed call is a
        durable step keyed by its call id. Calls remain sequential unless every definition in the
        batch explicitly declares concurrency safety.

        Pass the raw tool registry here. The durable and concurrency wrappers belong to this
        boundary so callers cannot accidentally compose them in the unsafe order.
        """
        publisher = emit if emit is not None else self._emit
        # Before `record_pending`, so an unkeyable round leaves no durable trace of itself. The
        # engine checks this too; this is the durable boundary, and it is public.
        require_call_ids(calls)
        await self.record_pending(calls, turn)
        durable = Concurrent(Stepped(tools, self), aborted)
        try:
            result = await execute_calls(durable, calls, aborted, publisher, turn, controls, ctx)
        except RoundSuspended as stopped:
            result = stopped.resolved
        await self._terminate_if_suspended(tools, result, publisher, turn, controls, ctx)
        return result

    async def resume_effect(
        self,
        tools: Tools,
        call: ToolCall,
        answer: dict[str, Any],
        request: dict[str, Any],
        suspended_rules_version: str,
        *,
        aborted: Aborted = lambda: False,
        emit: Emit | None = None,
        turn: int = 0,
        controls: Controls | None = None,
        ctx: Ctx | None = None,
    ) -> dict[str, Any]:
        """Revalidate and finish the effect that a permission gate parked before execution.

        This is deliberately not `execute_round`: the original approval-request gate has already
        run, and replaying it would ask the same question forever. `on_resume` receives the human
        answer and current policy window; only `Continue` crosses the durable effect boundary.
        """
        if aborted():
            raise AgentAborted()

        publisher = emit if emit is not None else self._emit
        context = ctx if ctx is not None else Ctx(turn=turn)
        if publisher is not None:
            await publisher(EventType.PRE_TOOL_USE, tool_payload(turn, call))

        resume = ResumeInput(
            answer=answer,
            request=request,
            suspended_rules_version=suspended_rules_version,
            current_rules_version=self._rules_version,
        )
        decision = (
            Deny(answer)
            if answer.get("type") == "error"
            else await controls.on_resume(context, call, resume)
            if controls is not None
            else Continue()
        )

        match decision:
            case Deny(result):
                if publisher is not None:
                    await publisher(
                        EventType.PERMISSION_DENIED,
                        tool_payload(turn, call, reason=result, source="on_resume"),
                    )
                resolved = Resolved(call, result, refused=True)
            case Suspend(new_request):
                if publisher is not None:
                    await publisher(
                        EventType.PERMISSION_REQUEST,
                        tool_payload(turn, call, request=new_request, source="on_resume"),
                    )
                resolved = Resolved(call, new_request, refused=True)
            case _:
                result = await Stepped(tools, self).execute(
                    call["name"], call["id"] or "", call["args"]
                )
                resolved = Resolved(call, result, refused=False)

        if not resolved.refused:
            await record_resolved(controls, publisher, context, call, resolved.result)
        items = [resolved]
        await self._terminate_if_suspended(tools, items, publisher, turn, controls, context)
        if self._on_agent_event is not None:
            event = tool_result(call, resolved.result)
            event["executed"] = not resolved.refused
            await self._on_agent_event(event)
        return resolved.result

    async def _terminate_if_suspended(
        self,
        tools: Tools,
        resolved: list[Resolved],
        emit: Emit | None,
        turn: int,
        controls: Controls | None,
        ctx: Ctx | None,
    ) -> None:
        round_ = absorb_round(tools, resolved)
        if round_.suspended is None:
            return
        call, request = round_.suspended
        context = ctx if ctx is not None else Ctx(turn=turn)
        messages = [*context.messages, *round_.answers]
        snapshot = suspend_history_snapshot(
            messages,
            call["id"] or "",
            [str(completed["id"]) for completed in round_.completed],
        )
        suspended_result = next(item for item in resolved if item.call["id"] == call["id"])
        if not suspended_result.refused:
            raise RuntimeError("only a pre_tool_use permission gate may suspend a tool request")
        if controls is not None:
            await controls.on_suspend(context, call, request, snapshot, round_.completed)
        await self.persist_suspension(
            call,
            request,
            snapshot,
            round_.completed,
            turn=turn,
        )
        if self._on_agent_event is not None:
            for item in resolved:
                event = tool_result(item.call, item.result)
                event["executed"] = not item.refused
                await self._on_agent_event(event)
            await self._on_agent_event(
                {
                    "type": "suspended",
                    "pending_id": request["pending_id"],
                    "tool_call_id": call["id"],
                }
            )
        raise AgentSuspended(str(request["pending_id"]), call["id"] or "")

    async def record_pending(self, calls: list[ToolCall], turn: int) -> None:
        """Commit call order before the first policy check or external effect."""
        # Do not copy the transcript here: the agent's history store already owns it, and copying
        # a growing history every round would turn while-loop persistence quadratic.
        await self._log.finish(
            self.run_id,
            _pending_round_key(),
            {"calls": list(calls), "turn": turn},
            self._token,
        )

    async def pending_calls(self) -> list[ToolCall]:
        """The latest model-issued call list, in order, or an empty list.

        Outcomes are intentionally not duplicated here; inspect each call id in `StepLog` to tell
        `done`, `running`, and `absent` apart.
        """
        record = await self._log.read(self.run_id, _pending_round_key())
        if record.status != "done" or not isinstance(record.value, dict):
            return []
        calls = record.value.get("calls")
        if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
            raise TypeError("pending tool round did not record a call list")
        return [cast(ToolCall, call) for call in calls]

    async def recover_pending(
        self,
        history: list[BaseMessage],
        tools: Tools,
        *,
        aborted: Aborted = lambda: False,
        emit: Emit | None = None,
        turn: int | None = None,
        controls: Controls | None = None,
        retry_running: bool = True,
    ) -> RecoveredTools:
        """Complete the latest unanswered tool round without replaying its model call.

        Recovery first inspects every call. `done` results are restored from the ledger; `absent`
        calls are gated and executed. A `running` call is retried with the same tool-call id by
        default — ADR-002 makes that id the receiver's idempotency key. Set `retry_running=False`
        to surface `Indeterminate` instead when a tool cannot honour that contract. No new effect
        begins until every ambiguous step has either been cleared for keyed retry or rejected.
        Calls are merged in the model's original order; only batches whose tools all opted into
        concurrency may execute together.

        The agent transcript owns `history`; `execute_round` separately saves the pending call list
        and the per-call ledger owns their outcomes. Keeping those stores separate avoids copying
        a growing conversation on every round. The returned history is ready to pass back to
        `react_loop(history=...)`.

        Replaying a cached result calls `controls.after_tool_call` again. That closes the crash
        window between committing the tool result and committing its journal entry, so journal
        writers must deduplicate by call id.
        """
        stored_turn = 0
        round_record = await self._log.read(self.run_id, _pending_round_key())
        if round_record.status == "done" and isinstance(round_record.value, dict):
            stored_turn = int(round_record.value.get("turn", 0))

        recovery_turn = stored_turn if turn is None else turn
        pending = _unanswered_tool_calls(history)
        if not pending:
            return RecoveredTools(list(history), [], [], None)
        if round_record.status == "done" and isinstance(round_record.value, dict):
            recorded_calls = round_record.value.get("calls", [])
            recorded_ids = {call.get("id") for call in recorded_calls if isinstance(call, dict)}
            if any(call["id"] not in recorded_ids for call in pending):
                raise ValueError("agent history does not match the orchestrator's pending round")

        records: list[tuple[ToolCall, Step]] = []
        for call in pending:
            call_id = call["id"] or ""
            record = await self._log.read(self.run_id, call_id)
            if record.status == "running" and not retry_running:
                raise Indeterminate(self.run_id, call_id)
            records.append((call, record))

        for call, record in records:
            if record.status == "running":
                await self.force_retry(call["id"] or "")
        records = [
            (call, Step("absent") if record.status == "running" else record)
            for call, record in records
        ]

        context = Ctx(turn=recovery_turn, messages=list(history))
        publisher = emit if emit is not None else self._emit
        resolved: list[Resolved] = []
        for call, record in records:
            if aborted():
                break
            if record.status == "done":
                if not isinstance(record.value, dict):
                    raise TypeError(f"tool step {call['id']!r} did not record a result dict")
                await record_resolved(controls, publisher, context, call, record.value)
                item = Resolved(call, record.value, refused=False)
            else:
                # The original pending-round record keeps the complete model order. Calling the
                # public `execute_round` for this one recovery item would overwrite it with a
                # one-call subset and make a second crash lose the remaining calls.
                durable = Concurrent(Stepped(tools, self), aborted)
                try:
                    fresh = await execute_calls(
                        durable,
                        [call],
                        aborted,
                        publisher,
                        recovery_turn,
                        controls,
                        context,
                    )
                except RoundSuspended as stopped:
                    fresh = stopped.resolved
                if not fresh:
                    break
                item = fresh[0]
            resolved.append(item)
            if item.result.get("type") == "suspend":
                break

        await self._terminate_if_suspended(
            tools,
            resolved,
            publisher,
            recovery_turn,
            controls,
            context,
        )
        absorbed = absorb_round(tools, resolved)
        return RecoveredTools(
            [*history, *absorbed.answers],
            resolved,
            absorbed.completed,
            absorbed.suspended,
        )

    async def _clear(self, step: str) -> None:
        forget = getattr(self._log, "forget", None)
        if forget is not None:
            await forget(self.run_id, step)

    async def force_retry(self, step: str) -> None:
        """Clear an `Indeterminate` step's intent so the next attempt runs it again.

        Ordinary workflow `run` steps require this explicit claim. Agent recovery calls it
        automatically only for tool calls, whose call id is the required receiver idempotency key;
        `recover_pending(retry_running=False)` disables that policy.
        """
        if getattr(self._log, "forget", None) is None:
            raise NotImplementedError(f"{type(self._log).__name__} cannot clear a step")
        await self._clear(step)
        self._seen.discard(step)

    async def signal(self, name: str) -> Any:
        """The answer to `name`, or end this attempt until someone provides it.

        Waiting costs nothing while stopped, which is the whole reason an approval may take days
        — the same reason the tool gate suspends instead of blocking on a human.
        """
        self._claim(name)
        record = await self._log.read(self.run_id, _signal_key(name))
        if record.status != "done":
            raise Suspended(name)
        return record.value

    async def suspend(self, key: str, payload: dict[str, Any]) -> None:
        """Park a continuation. `payload` is opaque — encode it with a codec first.

        The state belongs here because resuming a run is this layer's job. What is *in* it does
        not: `nexora.history.encode_continuation` turns messages into something this can store
        without learning what a message is.

        Keyed by the call id, already the idempotency key (ADR-002), so a second suspension of the
        same call overwrites and there is nothing to reconcile.
        """
        await self._log.finish(self.run_id, _suspend_key(key), payload, self._token)

    async def persist_suspension(
        self,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
        *,
        turn: int,
    ) -> None:
        """Commit the waiting record owned by this execution layer.

        The execution boundary calls this before terminating a suspended round. No agent
        generator or worker remains alive while approval is outstanding; only this durable
        record does.
        """
        call_id = call["id"] or ""
        continuation = encode_continuation(
            call,
            request,
            snapshot,
            completed,
            self._rules_version,
            turn=turn,
        )
        await self._log.commit_transition(
            self.run_id,
            {
                _suspend_key(call_id): continuation,
                _active_suspension_key(): {
                    "state": "waiting",
                    "call_id": call_id,
                    "continuation": continuation,
                },
            },
            [],
            self._token,
        )
        if self._on_suspend is not None:
            await self._on_suspend(call, request, snapshot, completed)

    async def suspension(self, key: str) -> dict[str, Any] | None:
        """The parked payload, still opaque, or None if there is none."""
        record = await self._log.read(self.run_id, _suspend_key(key))
        return record.value if record.status == "done" else None

    async def resolve(self, name: str, answer: Any) -> None:
        """Write a signal's answer. Called from outside the workflow, then replay it."""
        # No token: an answer arrives from outside the run — a webhook, a dialog, an operator — and
        # that caller holds no lease. Fencing a signal would refuse the very writes it is for.
        await self._log.finish(self.run_id, _signal_key(name), answer)

    def _claim(self, key: str) -> None:
        """Two steps sharing a name would replay as one another's result.

        Checked rather than trusted, because the failure is silent: the second `run("send")`
        would return the first one's value and never execute.
        """
        if key in self._seen:
            raise ValueError(f"duplicate step name in one attempt: {key!r}")
        self._seen.add(key)


def _signal_key(name: str) -> str:
    """Signals and steps share one keyspace, so they are namespaced apart."""
    return f"signal:{name}"


def _suspend_key(call_id: str) -> str:
    return f"suspend:{call_id}"


def _active_suspension_key() -> str:
    return "agent:active-suspension"


def _pending_round_key() -> str:
    return "agent:pending-round"


def _unanswered_tool_calls(history: list[BaseMessage]) -> list[ToolCall]:
    """Calls in the latest assistant tool round that have no following `ToolMessage`."""
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, AIMessage):
            continue
        if not message.tool_calls:
            return []
        answered = {
            later.tool_call_id for later in history[index + 1 :] if isinstance(later, ToolMessage)
        }
        return [call for call in message.tool_calls if (call["id"] or "") not in answered]
    return []
