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

`StepLog` answers one narrow question for opaque keys: absent, running, or done. It does not claim
exactly-once execution. A crash between an external effect and its committed result remains
`running`/`Indeterminate` until an idempotency key or reconciliation resolves the ambiguity.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, NamedTuple, Protocol, cast, runtime_checkable
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from .contracts.events import EventType, RuntimeEvents
from .contracts.types import Aborted, BaseMessage, Emit, OnSuspend, PendingInput, ToolCall, Tools
from .controls import Continue, Controls, Ctx, Deny, ResumeInput, Suspend
from .history import (
    SuspensionKind,
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
    a_tool_result,
    absorb_round,
    announce_batch,
    execute_calls,
    record_resolved,
    tool_payload,
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
    "PermissionChain",
    "RecoveredTools",
    "Step",
    "StepLog",
    "Suspended",
    "run_agent",
]


class PermissionChain:
    """The stages a tool call passes to be allowed, in order. Where middleware lives.

    Composed at this layer — a supervisor stacks as many stages as it wants — but it is a **call
    chain, not a subscription**. `resolve` returns a decision the caller holds and acts on, and
    the *position* of each stage is the policy:

        PermissionChain(
            hook,             # the caller's own opinion, first and not trusted
            deny_rules,       # ↓ a deny here beats the hook's allow
            tool_check,
            safety_check,     # ↑ nothing below can lift a decision made above
            bypass,           # only from here on may anything answer "allow"
            allow_rules,
        )

    That ordering cannot be expressed with subscribers: published events arrive in whatever order
    a dispatcher chooses, so "the bypass entry sits after the deny rules" — the invariant the
    whole model rests on — would not be an invariant at all. Events are for telling a UI and an
    audit log what happened, after the fact.

    Precedence, so it is readable in one place:

    * **deny short-circuits.** No later stage runs; nothing can lift it.
    * **ask is remembered, and the chain keeps going.** A deny after an ask still wins, so a
      stage that only wants confirmation cannot accidentally shield a call from a refusal.
    * **allow does not end the chain.** A stage answering `allow` is an opinion, re-checked
      against every stage after it. This is what stops a permissive hook from being the last word.
    * **nothing matched → allow.** Fail-open at the *end* of the chain is a deliberate choice:
      a caller that wants the opposite adds a final stage that asks.
    """

    def __init__(self, *stages: Callable[[ToolCall], Awaitable[Any]], record: Any = None) -> None:
        self._stages = stages
        self._record = record

    async def resolve(self, call: ToolCall) -> dict[str, Any] | None:
        """The decision: `None` to allow, an `error` result to deny, a `suspend` result to ask."""
        asked: dict[str, Any] | None = None
        for stage in self._stages:
            answer = await stage(call)
            if not isinstance(answer, dict):
                continue
            kind = answer.get("type")
            if kind == "error":
                return answer
            if kind == "suspend" and asked is None:
                asked = answer
        return asked

    async def record(self, call: ToolCall, result: dict[str, Any]) -> None:
        """The durable note that this call resolved. Raises through — fail-closed.

        Not an event: `EventStream` logs a failing sink and carries on, and a run that outran its
        own record cannot tell on resume which calls already ran.
        """
        if self._record is not None:
            await self._record(call, result)


class AgentFailed(Exception):
    """A run ended in `error`. Raised so a step cannot record a failure as an outcome."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


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
            raise AgentFailed(event["message"])
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


class Fenced(Exception):
    """A write arrived from a worker whose lease has been taken over. Refused.

    The failure this exists for: a worker stalls — GC pause, a hung socket, a suspended VM — past
    its lease TTL, another worker takes the run over, and then the first one wakes up and finishes
    what it was doing. It still believes it holds the lease, so no amount of renewal helps. Only
    the token does: it is stale, and the store says no.
    """

    def __init__(self, run_id: str, presented: int, issued: int) -> None:
        super().__init__(
            f"run {run_id!r} moved on: write presented token {presented}, current is {issued}"
        )
        self.run_id = run_id
        self.presented = presented
        self.issued = issued


class Contended(Exception):
    """Someone else holds this run's lease. Not an error — try later, or let them finish."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id!r} is held by another worker")
        self.run_id = run_id


class Indeterminate(Exception):
    """A step was started and never finished. Whether its effect happened is unknown.

    Raised instead of quietly re-running, because the two safe answers are opposite and only the
    caller knows which applies: re-run (the effect is idempotent, or provably did not happen), or
    reconcile against the external system. Guessing is how a payment goes out twice.

    `force_retry` clears the intent for exactly this decision.
    """

    def __init__(self, run_id: str, step: str) -> None:
        super().__init__(f"step {step!r} of run {run_id!r} may or may not have happened")
        self.run_id = run_id
        self.step = step


class Step(NamedTuple):
    """What the log knows about one step. Three states, not two.

    `running` is the state a two-state log cannot express, and it is the one that matters: a crash
    between the effect and its result leaves exactly this, and a log that only says
    "recorded / not recorded" reports it as *not recorded* and replays the effect.
    """

    status: Literal["absent", "running", "done"]
    value: Any = None


class RecoveredTools(NamedTuple):
    """A persisted assistant tool round completed without replaying its model call."""

    history: list[BaseMessage]
    """The supplied history plus reconstructed `ToolMessage`s, ready for `react_loop`."""
    resolved: list[Resolved]
    completed: list[dict[str, Any]]
    suspended: tuple[ToolCall, dict[str, Any]] | None


class InputRecord(NamedTuple):
    """One durable queue row, ordered within a run."""

    input_id: str
    status: Literal["pending", "claimed", "admitted"]
    value: dict[str, Any]
    sequence: int


@runtime_checkable
class StepLog(Protocol):
    """Where step results and signal answers are written down.

    **What this gives, named precisely:** durable intent, ambiguity detection, and exclusive
    execution. *Not* exactly-once — no ledger can promise that for an arbitrary external effect.
    A crash after the pharmacy accepted the request and before `finish` committed leaves the log
    unable to say whether it went out; it declines to guess (`Indeterminate`) rather than
    repeating it. So: **at-most-once automatic attempt, with the ambiguity surfaced.** Real
    exactly-once needs the receiver's own idempotency key, a transactional outbox, or a
    reconciliation query — all outside this file.

    Three write calls, because the ledger records an *intent* before the effect runs. `start` must
    be committed before `Orchestrator.run` calls the function and `finish` after it returns; a
    store that batches or defers either one gives back the guarantee they exist to provide.

    The lease is the other half. Two workers replaying one run both see `absent` and both run the
    effect, and no per-step record can prevent that. `acquire` returns a **fencing token**, not a
    boolean: renewal alone cannot make a lease safe, because a worker can stall past its TTL and
    wake up still believing it holds the run. Carrying the token on every write is what makes a
    stalled worker harmless — its writes are refused, whatever it believes.
    """

    async def read(self, run_id: str, key: str) -> Step: ...

    async def start(self, run_id: str, key: str, token: int = 0) -> None:
        """Record the intent. Committed before the effect runs. Refused on a stale token."""
        ...

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        """Record the outcome. The step is `done` only after this. Refused on a stale token."""
        ...

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float) -> int:
        """A fencing token if this owner now holds the run, 0 if someone else does.

        The token increases every time the lease changes hands, so the newest holder always has
        the highest one. A write carrying anything lower is from a worker that has been replaced.
        """
        ...

    async def release(self, run_id: str, owner: str) -> None: ...

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently. This write comes from outside the run lease."""
        ...

    async def list_inputs(self, run_id: str) -> list[InputRecord]: ...

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None: ...

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None: ...


class MemorySteps:
    """A `StepLog` in a dict. Correct semantics, no durability — it dies with the process.

    Useful for tests and for finding out whether a workflow's step boundaries are right before
    paying for a database. The lease is real within one process, which is enough to catch a
    workflow that accidentally runs itself twice concurrently.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Step] = {}
        self._leases: dict[str, tuple[str, int]] = {}
        self._tokens: dict[str, int] = {}
        self._inputs: dict[tuple[str, str], InputRecord] = {}
        self._input_sequence: dict[str, int] = {}

    async def read(self, run_id: str, key: str) -> Step:
        return self._entries.get((run_id, key), Step("absent"))

    async def start(self, run_id: str, key: str, token: int = 0) -> None:
        self._fence(run_id, token)
        self._entries[run_id, key] = Step("running")

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        self._fence(run_id, token)
        self._entries[run_id, key] = Step("done", value)

    async def forget(self, run_id: str, key: str) -> None:
        self._entries.pop((run_id, key), None)

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        held = self._leases.get(run_id)
        if held is not None and held[0] != owner:
            return 0
        if held is not None:
            return held[1]  # the holder renewing keeps its token
        self._tokens[run_id] = self._tokens.get(run_id, 0) + 1
        self._leases[run_id] = (owner, self._tokens[run_id])
        return self._tokens[run_id]

    async def release(self, run_id: str, owner: str) -> None:
        held = self._leases.get(run_id)
        if held is not None and held[0] == owner:
            del self._leases[run_id]

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        key = (run_id, input_id)
        if key in self._inputs:
            return False
        sequence = self._input_sequence.get(run_id, 0)
        self._input_sequence[run_id] = sequence + 1
        self._inputs[key] = InputRecord(input_id, "pending", value, sequence)
        return True

    async def list_inputs(self, run_id: str) -> list[InputRecord]:
        return sorted(
            (record for (item_run, _), record in self._inputs.items() if item_run == run_id),
            key=lambda record: record.sequence,
        )

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None:
        self._fence(run_id, token)
        key = (run_id, input_id)
        record = self._inputs[key]
        if record.status != "admitted":
            self._inputs[key] = InputRecord(input_id, "claimed", record.value, record.sequence)

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        self._fence(run_id, token)
        for input_id in input_ids:
            key = (run_id, input_id)
            record = self._inputs.get(key)
            if record is not None:
                self._inputs[key] = InputRecord(
                    input_id, "admitted", record.value, record.sequence
                )

    def _fence(self, run_id: str, token: int) -> None:
        """Refuse a write from a worker that has been replaced.

        `token=0` means "I hold no lease" and is not fenced. Two kinds of caller present it: a
        single-process user who never took one, and a writer from outside the run — a webhook
        answering a signal holds no lease, and fencing it would refuse the writes it exists for.
        A worker that *did* take a lease always has a non-zero token, so it cannot land here by
        accident.
        """
        issued = self._tokens.get(run_id, 0)
        if token and token < issued:
            raise Fenced(run_id, token, issued)


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
        input_id = item.origin_id or str(uuid4())
        message = item.message.model_copy(update={"id": input_id})
        normalized = PendingInput(item.kind, message, input_id)
        inserted = await self._log.enqueue_input(
            self.run_id, input_id, encode_pending_input(normalized)
        )
        if inserted and self._emit is not None and item.kind in {"user_prompt", "user_steer"}:
            await self._emit(
                EventType.USER_PROMPT_SUBMIT,
                {"input_id": input_id, "prompt": message.content, "source": item.kind},
            )
        return normalized

    async def claim_inputs(self, history: list[BaseMessage] | None = None) -> list[PendingInput]:
        """Claim every queued input absent from this attempt's transcript, in arrival order."""
        represented = {message.id for message in history or [] if message.id is not None}
        claimed: list[PendingInput] = []
        for record in await self._log.list_inputs(self.run_id):
            if record.input_id in represented or record.input_id in self._seen_inputs:
                continue
            await self._log.claim_input(self.run_id, record.input_id, self._token)
            self._seen_inputs.add(record.input_id)
            claimed.append(decode_pending_input(record.value))
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
        await announce_batch(publisher, turn, items)
        if self._on_agent_event is not None:
            event = a_tool_result(call, resolved.result)
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
        kind: SuspensionKind = "effect_approval" if suspended_result.refused else "elicitation"
        if controls is not None:
            await controls.on_suspend(context, call, request, snapshot, round_.completed)
        await self.persist_suspension(
            call,
            request,
            snapshot,
            round_.completed,
            kind=kind,
            turn=turn,
        )

        if emit is not None and not suspended_result.refused:
            # A gate suspension already announced this while deciding. A tool-originated
            # suspension reaches the same approval path, so announce it here after persistence.
            await emit(
                EventType.PERMISSION_REQUEST,
                tool_payload(turn, call, request=request, source="tool_result"),
            )
        await announce_batch(emit, turn, resolved)
        if self._on_agent_event is not None:
            for item in resolved:
                event = a_tool_result(item.call, item.result)
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
        await announce_batch(publisher, recovery_turn, resolved)
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
        kind: SuspensionKind,
        turn: int,
    ) -> None:
        """Commit the waiting record owned by this execution layer.

        The execution boundary calls this before terminating a suspended round. No agent
        generator or worker remains alive while approval is outstanding; only this durable
        record does.
        """
        await self.suspend(
            call["id"] or "",
            encode_continuation(
                call,
                request,
                snapshot,
                completed,
                self._rules_version,
                kind=kind,
                turn=turn,
            ),
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
