"""Durable effect execution, suspension, and recovery for agent runs.

The orchestrator owns leases, step idempotency, input admission, permission continuations, and
tool-round recovery. The planner remains responsible for the model/tool control loop.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, cast
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from nexora_store import (
    ClearableSteps,
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
    ControlSignal,
    Emit,
    ModelErrorKind,
    ModelFailure,
    ModelFailureAction,
    ModelStepError,
    ModelStreamFactory,
    OnSuspend,
    PendingInput,
    ToolCall,
    Tools,
)
from .controls import Continue, Controls, Ctx, Deny, ResumeInput, Suspend
from .history import (
    compact_continuation,
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
    "ModelFailurePolicy",
    "Orchestrator",
    "RecoveredTools",
    "Step",
    "StepLog",
    "Suspended",
    "run_agent",
]


@dataclass(frozen=True, slots=True)
class ModelFailurePolicy:
    """Bound automatic retries and compaction for model request failures.

    Attributes:
        max_retries: Retry limit for rate-limit and server failures.
        max_compactions: Compaction limit for context-overflow failures.
        backoff: Optional callback invoked before retrying a model request.
    """

    max_retries: int = 2
    max_compactions: int = 1
    backoff: Callable[[ModelFailure], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        """Reject negative bounds instead of turning them into surprising comparisons."""
        if self.max_retries < 0 or self.max_compactions < 0:
            raise ValueError("model failure recovery bounds must be non-negative")

    async def __call__(self, failure: ModelFailure) -> ModelFailureAction:
        """Return the recovery action for one classified provider failure."""
        if failure.partial:
            return "fail"
        if (
            failure.error_kind == "context_overflow"
            and failure.attempt <= self.max_compactions
        ):
            return "compact"
        if (
            failure.error_kind in {"rate_limit", "server"}
            and failure.attempt <= self.max_retries
        ):
            if self.backoff is not None:
                await self.backoff(failure)
            return "retry"
        return "fail"


class AgentFailed(Exception):
    """Report an agent run that ended with an error event."""

    def __init__(
        self,
        message: str,
        partial: str = "",
        *,
        error_type: str = "UnknownError",
        error_kind: ModelErrorKind = "unknown",
    ) -> None:
        """Initialize the error with its message and streamed partial text."""
        super().__init__(message)
        self.partial = partial
        self.error_type = error_type
        self.error_kind = error_kind


async def run_agent(
    events: AsyncIterator[dict[str, Any]],
    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Consume an agent event stream and return its completed outcome.

    Args:
        events: Agent planner event stream.
        on_event: Optional observer invoked for every event.

    Returns:
        Terminal ``done`` event for a completed or policy-stopped run.

    Raises:
        AgentFailed: If the stream fails or ends without a terminal event.
        AgentSuspended: If the run suspends.
        AgentAborted: If the run is aborted.
    """
    async for event in events:
        if on_event is not None:
            await on_event(event)
        if event["type"] == "error":
            raise AgentFailed(
                event["message"],
                event.get("partial", ""),
                error_type=event.get("error_type", "UnknownError"),
                error_kind=event.get("error_kind", "unknown"),
            )
        if event["type"] == "suspended":
            raise AgentSuspended(event["pending_id"], event["tool_call_id"])
        if event["type"] == "done":
            if event.get("stop_reason") == "aborted":
                raise AgentAborted(event.get("content", ""))
            return event
    raise AgentFailed("the agent produced no terminal event")


class AgentAborted(Exception):
    """Report a run interrupted before producing an outcome."""

    def __init__(self, partial: str = "") -> None:
        """Initialize the interruption with any streamed partial text."""
        super().__init__("the run was aborted")
        self.partial = partial


class Suspended(ControlSignal):
    """Report an attempt waiting for an unresolved external signal.

    A `ControlSignal` so a tool that suspends the attempt from inside its own body still stops the
    attempt, instead of being reported to the model as a tool that failed.
    """

    def __init__(self, signal: str) -> None:
        """Initialize the suspension with its signal identifier."""
        super().__init__(f"waiting for signal: {signal}")
        self.signal = signal


class AgentSuspended(Suspended):
    """Report an agent tool call suspended with a persisted continuation."""

    def __init__(self, pending_id: str, tool_call_id: str) -> None:
        """Initialize the suspension with external and tool-call identifiers."""
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
    """Coordinate one durable workflow attempt under an exclusive run lease.

    The lease renews at step boundaries, and every protected write carries its fencing token.
    Reuse the same ``run_id`` when resuming the workflow.
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
        """Initialize one durable run attempt and its execution controls."""
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
        """Publish input metadata without exposing unscreened content."""
        if self._emit is not None and item.kind in {"user_prompt", "user_steer"}:
            await self._emit(
                EventType.USER_PROMPT_SUBMIT,
                {"input_id": item.origin_id, "source": item.kind},
            )

    async def claim_inputs(self, represented: set[str] | None = None) -> list[PendingInput]:
        """Claim unrepresented inputs in dependency and arrival order."""
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
        """Acquire the run lease and return this orchestrator."""
        self._token = await self._log.acquire(self.run_id, self.owner, self._ttl)
        if not self._token:
            raise Contended(self.run_id)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Release the run lease when the attempt exits."""
        await self._log.release(self.run_id, self.owner)
        self._token = 0

    async def _renew(self) -> None:
        """Extend the lease at a step boundary, if one is held at all."""
        if self._token:
            self._token = await self._log.acquire(self.run_id, self.owner, self._ttl) or self._token

    async def run(self, step: str, fn: Callable[[], Awaitable[Any] | Any]) -> Any:
        """Execute or replay one durable step.

        Args:
            step: Stable step identifier and idempotency key.
            fn: Synchronous or asynchronous effect to execute when no result exists.

        Returns:
            Newly produced or previously recorded result.

        Raises:
            Indeterminate: If execution intent exists without a recorded result.
        """
        record = await self._begin(step)
        if record is not None:
            if record.status == "done":
                return record.value
            raise Indeterminate(self.run_id, step)
        try:
            result = fn()
            if isinstance(result, Awaitable):
                result = await result
        except asyncio.CancelledError:
            # Cancellation is not the step's report about itself. The decision was made above it —
            # a shutdown, a cancelled parent, `cancel_task` on the subagent that launched this —
            # and it can land after the effect already has. So the intent stands, and the next
            # attempt reads `running` and raises `Indeterminate`: the same fact a crash leaves,
            # because it is the same fact. Clearing it here made a graceful stop more dangerous
            # than `kill -9`, which is the one comparison that has to come out the other way.
            raise
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

    async def invoke_model(
        self,
        step: str,
        factory: ModelStreamFactory,
    ) -> AsyncIterator[AIMessageChunk]:
        """Execute or replay one streamed model request as a durable effect step.

        A provider failure before its first chunk is known not to have exposed output, so its
        intent is cleared for the loop's bounded retry policy. Once a chunk was visible, an
        interrupted request remains ``running`` and recovery raises ``Indeterminate`` rather than
        risking a duplicate model charge and a different answer.
        """
        try:
            record = await self._begin(step)
        except Exception as error:
            raise ModelStepError(error) from error

        if record is not None and record.status == "done":
            try:
                yield_chunks = _decode_model_chunks(record.value)
            except Exception as error:
                raise ModelStepError(error) from error
            for chunk in yield_chunks:
                yield chunk
            return
        if record is not None:
            raise ModelStepError(Indeterminate(self.run_id, step))

        chunks: list[dict[str, Any]] = []
        try:
            stream = factory()
            try:
                async for chunk in stream:
                    chunks.append(message_to_dict(chunk))
                    yield chunk
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        except ControlSignal:
            raise
        except Exception:
            if not chunks:
                try:
                    await self._clear(step)
                    self._seen.discard(step)
                except Exception as error:
                    raise ModelStepError(error) from error
            raise

        try:
            await self._log.finish(
                self.run_id,
                step,
                {"type": "model_result", "chunks": chunks},
                self._token,
            )
        except Exception as error:
            raise ModelStepError(error) from error

    async def _begin(self, step: str) -> Step | None:
        """Claim an absent step atomically, or return the record that won the race."""
        self._claim(step)
        await self._renew()
        record = await self._log.read(self.run_id, step)
        if record.status != "absent":
            return record
        if await self._log.start(self.run_id, step, self._token):
            return None
        record = await self._log.read(self.run_id, step)
        if record.status == "absent":
            raise RuntimeError(f"step {step!r} disappeared while recording its intent")
        return record

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
        """Execute one model-issued tool round through the durable boundary.

        Calls are recorded by identifier and execute sequentially unless the complete batch is
        declared concurrency-safe.
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
            await publisher(EventType.PRE_TOOL_USE, tool_payload(context, call))

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
                        tool_payload(context, call, reason=result, source="on_resume"),
                    )
                resolved = Resolved(call, result, refused=True)
            case Suspend(new_request):
                if publisher is not None:
                    await publisher(
                        EventType.PERMISSION_REQUEST,
                        tool_payload(context, call, request=new_request, source="on_resume"),
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
        messages = [
            *context.messages,
            *round_.answers,
            *(message for _, message in round_.images),
            *(message for _, message in round_.context),
        ]
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
            subject=context.subject,
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
        """Recover the latest unanswered tool round without replaying its model call.

        Completed results are restored, absent calls are executed, and running calls are either
        retried with their original idempotency keys or reported as indeterminate.

        Args:
            history: Model history containing the unanswered tool round.
            tools: Raw tool executor.
            aborted: Cancellation predicate.
            emit: Optional event publisher.
            turn: Optional turn override.
            controls: Runtime control plane.
            retry_running: Whether to retry ambiguous running steps.

        Returns:
            Reconstructed history and ordered tool outcomes.

        Raises:
            Indeterminate: If a running step exists and retry is disabled.
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
            [
                *history,
                *absorbed.answers,
                *(message for _, message in absorbed.images),
                *(message for _, message in absorbed.context),
            ],
            resolved,
            absorbed.completed,
            absorbed.suspended,
        )

    async def _clear(self, step: str) -> None:
        if isinstance(self._log, ClearableSteps):
            await self._log.forget(self.run_id, step)

    async def force_retry(self, step: str) -> None:
        """Clear an indeterminate step so a subsequent attempt can execute it again."""
        if not isinstance(self._log, ClearableSteps):
            raise NotImplementedError(f"{type(self._log).__name__} cannot clear a step")
        await self._clear(step)
        self._seen.discard(step)

    async def signal(self, name: str) -> Any:
        """Return a recorded signal or suspend the current attempt."""
        self._claim(name)
        record = await self._log.read(self.run_id, _signal_key(name))
        if record.status != "done":
            raise Suspended(name)
        return record.value

    async def suspend(self, key: str, payload: dict[str, Any]) -> None:
        """Persist an opaque continuation payload under an idempotent key."""
        await self._log.finish(self.run_id, _suspend_key(key), payload, self._token)

    async def persist_suspension(
        self,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
        *,
        turn: int,
        subject: str = "",
    ) -> None:
        """Persist a tool-round continuation before terminating the attempt."""
        call_id = call["id"] or ""
        continuation = encode_continuation(
            call,
            request,
            snapshot,
            completed,
            self._rules_version,
            turn=turn,
            subject=subject,
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

    async def compact_suspension(
        self, call_id: str, *, conversation_id: str, leaf_uuid: str | None
    ) -> None:
        """Replace a durable snapshot with its transcript cursor after transcript sync."""
        active = await self.active_continuation()
        if (
            active is None
            or active.get("state") != "waiting"
            or active.get("call_id") != call_id
            or not isinstance(active.get("continuation"), dict)
        ):
            raise RuntimeError(f"cannot compact inactive suspension {call_id!r}")
        continuation = compact_continuation(
            active["continuation"],
            conversation_id=conversation_id,
            leaf_uuid=leaf_uuid,
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


def _decode_model_chunks(value: Any) -> list[AIMessageChunk]:
    """Validate and decode a model-step payload stored as JSON-compatible message mappings."""
    if not isinstance(value, dict) or value.get("type") != "model_result":
        raise RuntimeError("durable model step has an invalid result envelope")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not all(isinstance(item, dict) for item in chunks):
        raise RuntimeError("durable model step has invalid chunks")
    decoded = messages_from_dict(chunks)
    if not all(isinstance(message, AIMessageChunk) for message in decoded):
        raise RuntimeError("durable model step contains a non-chunk message")
    return cast(list[AIMessageChunk], decoded)
