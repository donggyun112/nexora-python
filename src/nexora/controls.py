"""Control points: the places an engine asks before acting, and what composes at each one.

A **control point** is awaited before anything happens and its return value changes the flow. An
**event** is a fact published after the decision, for a UI or an audit log. Keeping them apart is
the whole reason this file exists: a permission check that arrives as a subscriber has neither a
guaranteed order nor a return value, so the order of the stages — which *is* the policy — stops
being expressible. The control method and observation event use the same lifecycle name.

    await emit(EventType.PRE_TOOL_USE, payload(ctx.turn, call))
    decision = await controls.pre_tool_use(ctx, call)
    match decision:
        case Continue(): ...
        case Deny(result): ...
        case Suspend(request): ...

**Every control point composes differently**, which is why there is no one generic gate:

| point | rule |
|---|---|
| `on_inputs` | screens chain in order, each seeing the last one's output; any may halt |
| `pre_tool_use` | a deny beats a suspend; an `allow` does not short-circuit |
| `before_model` | steers accumulate in order; any gate may halt |
| `after_tool_call` | every writer runs, and a raise stops the run — fail-closed |
| `before_finish` | any verifier may veto completion and inject steering |
| `on_resume` | the human answer is revalidated under current policy |
| `on_suspend` | all must succeed before `suspended` is published |

A `dict[name, list[callable]]` cannot hold those rules. Adding a *gate* means registering it
in an existing pipeline; adding a *control point* means a new method here, a planner or
orchestrator boundary that calls it, and a contract test.

**A stage must be safe to run again.** Recovery re-enters every one of them. The two tool-boundary
points are re-entered from the ledger — `Orchestrator.recover_pending` calls `after_tool_call`
again for a result it restored from a completed step, deliberately, to close the crash window
between committing a tool result and committing its journal entry. The rest are re-entered because
the round itself runs again. So a stage that accumulates — a spend counter, a non-idempotent
external write — double-counts on recovery. Derive from `ctx.messages` or the ledger instead of
adding up, and key any write by the `call_id` the stage already receives.

**A stage is not a durable boundary.** What it changes is what the model and the transcript see;
the ledger was written first. The seams that do reach the durable copy:

*A tool result whose PII must never reach the ledger* — wrap the executor, not the control plane:

    await runtime.run(run_id, model, Masking(my_tools), prompt)

`Masking` is any `Tools`, and the orchestrator wraps what you pass in `Concurrent(Stepped(...))`,
so your executor runs *inside* the step and the recorded value is already the masked one.

*A prompt whose PII must never reach the input ledger* — mask at your edge, before `submit()` or
`run(prompt=...)`. `on_inputs` is the defence for inputs you could not pre-screen, such as results
arriving from the background.

*A journal entry exactly once* — make the writer idempotent on `call_id`; recovery calls it again
by design. *A budget or verification count* — derive it, never accumulate. *A record that must not
be lost* — `after_tool_call`, which raises through, and never the event channel, which logs a
failing sink and carries on.

What no seam gives you: making a side effect *inside* a control stage its own durable step. The
`Orchestrator` holds the run's lease, so a stage cannot open a second one on the same run. Use
your own store with your own idempotency key.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

from .contracts.types import BaseMessage, PendingInput, StopReason, ToolCall

__all__ = [
    "Continue",
    "ControlPlane",
    "Controls",
    "Ctx",
    "Deny",
    "FinishPolicy",
    "Halt",
    "Ingress",
    "Journal",
    "Permissions",
    "Proceed",
    "ResumeInput",
    "Steering",
    "Suspend",
    "Suspending",
    "ToolDecision",
    "TurnDecision",
]


@dataclass(frozen=True, slots=True)
class Ctx:
    """Where the run is. Passed to every control point.

    One context rather than one per point: `turn` and the transcript are what a gate reads, and six
    near-identical classes would be six places to add a field. The point-specific value — the call,
    the results — is a separate argument, so the signature still says which point it is.
    """

    turn: int
    messages: list[BaseMessage] = field(default_factory=list)
    calls_made: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    """The assistant text of the round so far, for a gate that budgets on output."""


class Continue(NamedTuple):
    """Nothing to say. The call runs."""


class Deny(NamedTuple):
    """Refused. `result` stands in for the call, so the model sees a failed call and can react."""

    result: dict[str, Any]


class Suspend(NamedTuple):
    """Stop and wait for a person. `request` is whole — it carries any handle the tool minted."""

    request: dict[str, Any]


ToolDecision = Continue | Deny | Suspend


@dataclass(frozen=True, slots=True)
class ResumeInput:
    """The external answer plus the policy window a resume must revalidate."""

    answer: dict[str, Any]
    request: dict[str, Any]
    suspended_rules_version: str
    current_rules_version: str


class Proceed(NamedTuple):
    """Carry on, optionally with messages to inject first — this is what a steer looks like."""

    steers: list[BaseMessage] = []  # noqa: RUF012 — NamedTuple default, not shared mutable state


class Halt(NamedTuple):
    """End the run, with the reason that goes on the terminal event."""

    reason: StopReason


ModelAction = Proceed | Halt
TurnDecision = Proceed | Halt


@runtime_checkable
class Controls(Protocol):
    """What an engine asks. Implementations own the policy; engines own only the positions."""

    async def on_inputs(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        """Admission for everything about to enter model context: rewrite, drop, or halt.

        What a screen changes here is **what the model and the transcript see** — and only that.
        It is not a redaction boundary for the durable stores: `Orchestrator.enqueue_input` has
        already written the submitted text to the input ledger, and a tool result was recorded by
        `Stepped` before this point exists. Masking the durable copy is a different seam; see the
        recipes at the top of this module.

        Screens scope themselves by `kind`; dropping a `tool_result` input breaks the transcript
        the model needs, so screens that only care about people should leave other kinds untouched.
        """
        ...

    async def before_model(self, ctx: Ctx) -> ModelAction: ...

    async def pre_tool_use(self, ctx: Ctx, call: ToolCall) -> ToolDecision: ...

    async def after_tool_call(self, ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        """Durable record and validation. **Raises through** — a run must not outlive its record."""
        ...

    async def before_finish(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        """The last word before a run ends. `Proceed` vetoes the finish and goes around again.

        Asked on a round where the model requested no tools, after the engine has checked for a
        late steer — an arriving input and a policy objection are different reasons to continue.
        This is where a verifier says "not done yet"; its `Proceed` steers are admitted beside the
        next model call like any other input.

        A gate cannot relabel an ending it did not object to: `FinishPolicy` returns the engine's
        own reason, so the only thing a verifier decides is whether the run continues. Renaming
        needs a `before_finish` implementation of your own, not a gate inside that one.

        Nothing caps the vetoes. A gate that never lets go never ends, the same way a
        `drain_inputs` that always yields never ends — bound it with `should_stop_after_turn`
        accounting or with your own counter, which must be derived rather than accumulated.
        """
        ...

    async def on_resume(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> ToolDecision:
        """Re-decide a parked call under the rules in force *now*.

        A suspension's window is days and rules move inside it, so the answer a person gave is an
        input to the decision and never the decision. Hosts inject an answer-aware revalidator;
        the original approval-request gate is not blindly replayed because that would ask forever.
        """
        ...

    async def on_suspend(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        """Park the continuation. Runs **before** the event, so a consumer that raises on
        `suspended` cannot abandon the generator before the write happened."""
        ...


def gate(
    answer: Callable[[ToolCall], Awaitable[dict[str, Any] | None]],
) -> Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]:
    """A plain `(call) -> result | None` predicate as a `pre_tool_use` stage.

    Most gates are exactly that shape — look at the call, refuse or don't — and making each one
    take a context it ignores and build a decision type would be ceremony. The mapping is fixed:
    `None` and an `allow` result both let the chain continue, a `suspend` result asks, anything
    else denies.

    `allow` maps to `Continue` and not to a decision because that is `Permissions`' rule — a stage
    saying yes is an opinion, and every stage after it still runs. A host wiring an external hook
    that answers `{"type": "allow"}` would otherwise have its permissive answer silently inverted
    into a refusal, which is the one direction a permission bug must never go.
    """

    async def stage(ctx: Ctx, call: ToolCall) -> ToolDecision:
        decision = await answer(call)
        if decision is None or decision.get("type") == "allow":
            return Continue()
        return Suspend(decision) if decision.get("type") == "suspend" else Deny(decision)

    return stage


def writer(
    record: Callable[[ToolCall, dict[str, Any]], Awaitable[None]],
) -> Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]:
    """A plain `(call, result)` writer as an `after_tool_call` stage. Same reason as `gate`."""

    async def stage(ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        await record(call, result)

    return stage


# ── the composition rules, one per control point ─────────────────────────────


class Ingress:
    """Screens for `on_inputs`. They chain: each sees the previous one's output; any may halt.

    A pipeline and not a vote, because this point exists to transform — once a screen has masked
    a prompt, every later screen and the model must see the masked text, never the original.
    """

    def __init__(
        self,
        *screens: Callable[[Ctx, list[PendingInput]], Awaitable[list[PendingInput] | Halt]],
    ) -> None:
        self._screens = screens

    async def __call__(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        for screen in self._screens:
            match await screen(ctx, inputs):
                case Halt() as stop:
                    return stop
                case screened:
                    inputs = screened
        return inputs


class Permissions:
    """Ordered stages for `pre_tool_use`. A deny wins; an `allow` decides nothing.

    That second half is the load-bearing one: a stage answering `Continue` is an opinion, and every
    stage after it still runs. It is what stops a permissive hook from being the last word, and it
    is why the rule layer can sit behind a hook without needing a second copy of itself.
    """

    def __init__(self, *stages: Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]) -> None:
        self._stages = stages

    async def __call__(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        asked: Suspend | None = None
        for stage in self._stages:
            match await stage(ctx, call):
                case Deny() as denied:
                    return denied
                case Suspend() as request if asked is None:
                    asked = request
                case _:
                    pass
        return asked or Continue()


class Journal:
    """Writers for `after_tool_call`. All of them run; the first raise stops the run.

    Fail-closed on purpose. A run that carried on past a failed write cannot tell, on resume, which
    calls already happened — and that is the one question the record exists to answer.
    """

    def __init__(
        self, *writers: Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._writers = writers

    async def __call__(self, ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        for write in self._writers:
            await write(ctx, call, result)


class FinishPolicy:
    """Gates for `before_finish`. Any one of them may veto the ending; silence lets it end.

    Proceed steers from every verifier accumulate in registration order. Without an objection,
    the original stop reason is returned as a Halt decision.
    """

    def __init__(
        self, *gates: Callable[[Ctx, StopReason], Awaitable[TurnDecision]]
    ) -> None:
        self._gates = gates

    async def __call__(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        keep_going: Proceed | None = None
        for gate in self._gates:
            match await gate(ctx, reason):
                case Proceed(steers):
                    keep_going = Proceed([*(keep_going.steers if keep_going else []), *steers])
                case _:
                    pass
        return keep_going or Halt(reason)


class Steering:
    """Sources for `before_model`. Messages accumulate in arrival order; any source may halt."""

    def __init__(self, *sources: Callable[[Ctx], Awaitable[ModelAction]]) -> None:
        self._sources = sources

    async def __call__(self, ctx: Ctx) -> ModelAction:
        steers: list[BaseMessage] = []
        for source in self._sources:
            match await source(ctx):
                case Halt() as stop:
                    return stop
                case Proceed(more):
                    steers += more
        return Proceed(steers)


class Suspending:
    """Persisters for `on_suspend`. All must succeed before the suspension is announced."""

    def __init__(
        self,
        *persisters: Callable[
            [Ctx, ToolCall, dict[str, Any], list[BaseMessage], list[dict[str, Any]]],
            Awaitable[None],
        ],
    ) -> None:
        self._persisters = persisters

    async def __call__(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        for persist in self._persisters:
            await persist(ctx, call, request, snapshot, completed)


class ControlPlane:
    """The assembled control points. Every one is optional and defaults to letting the run through.

    An engine holds one of these and knows nothing about what is registered in it.
    """

    def __init__(
        self,
        *,
        on_inputs: Any = None,
        before_model: Any = None,
        pre_tool_use: Any = None,
        after_tool_call: Any = None,
        before_finish: Any = None,
        on_resume: Any = None,
        on_suspend: Any = None,
    ) -> None:
        self._on_inputs = on_inputs
        self._before_model = before_model
        self._pre_tool_use = pre_tool_use
        self._after_tool_call = after_tool_call
        self._before_finish = before_finish
        self._on_resume = on_resume
        self._on_suspend = on_suspend

    async def on_inputs(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        if self._on_inputs is None:
            return inputs
        admitted: list[PendingInput] | Halt = await self._on_inputs(ctx, inputs)
        return admitted

    async def before_model(self, ctx: Ctx) -> ModelAction:
        if self._before_model is None:
            return Proceed()
        action: ModelAction = await self._before_model(ctx)
        return action

    async def pre_tool_use(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        if self._pre_tool_use is None:
            return Continue()
        decision: ToolDecision = await self._pre_tool_use(ctx, call)
        return decision

    async def after_tool_call(self, ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        if self._after_tool_call is not None:
            await self._after_tool_call(ctx, call, result)

    async def before_finish(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        if self._before_finish is None:
            return Halt(reason)
        decision: TurnDecision = await self._before_finish(ctx, reason)
        return decision

    async def on_resume(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> ToolDecision:
        """Apply the human answer, then the separately injected latest-policy revalidator."""
        if resume.answer.get("type") == "error":
            return Deny(resume.answer)
        if self._on_resume is None:
            return Continue()
        decision: ToolDecision = await self._on_resume(ctx, call, resume)
        return decision

    async def on_suspend(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        if self._on_suspend is not None:
            await self._on_suspend(ctx, call, request, snapshot, completed)
