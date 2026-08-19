"""Define runtime control points and their composition policies.

Control points return load-bearing decisions before the runtime acts. Unlike observation events,
their ordering and failure behavior are part of the execution contract. Stages may run again
during recovery and therefore must be idempotent.
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
    "gate",
    "writer",
]


@dataclass(frozen=True, slots=True)
class Ctx:
    """Provide shared run context to every control point."""

    turn: int
    messages: list[BaseMessage] = field(default_factory=list)
    calls_made: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    """Assistant text accumulated in the current round."""
    subject: str = ""
    """Who the run acts for, as the host names them. **Never interpreted here.**

    A user id, a service account, a tenant-scoped pair, whatever an external directory calls a
    principal — this runtime cannot know which, so it carries the string and reads nothing out of
    it. Empty means the host did not say, which is the honest default: a framework that invented a
    subject would be putting a name it made up into an audit record.

    It reaches the record two ways, and both matter. A stage may decide with it — a gate that asks
    a directory per call needs to know who is asking. And it is stamped onto every tool event and
    onto a suspension, so "who was this denied for" and "whose authority was this parked under" have
    answers that do not depend on correlating by run id afterwards.
    """


class Continue(NamedTuple):
    """Allow control flow to continue without changes."""


class Deny(NamedTuple):
    """Deny a tool call and provide its model-visible result."""

    result: dict[str, Any]


class Suspend(NamedTuple):
    """Suspend a tool call with its complete external request."""

    request: dict[str, Any]


ToolDecision = Continue | Deny | Suspend


@dataclass(frozen=True, slots=True)
class ResumeInput:
    """Carry an external answer and the policy versions needed for revalidation."""

    answer: dict[str, Any]
    request: dict[str, Any]
    suspended_rules_version: str
    current_rules_version: str


class Proceed(NamedTuple):
    """Continue, optionally injecting steering messages first."""

    steers: list[BaseMessage] = []  # noqa: RUF012 — NamedTuple default, not shared mutable state


class Halt(NamedTuple):
    """End the run with a terminal stop reason."""

    reason: StopReason


ModelAction = Proceed | Halt
TurnDecision = Proceed | Halt


@runtime_checkable
class Controls(Protocol):
    """Define the policy decisions requested at runtime execution boundaries."""

    async def on_inputs(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        """Rewrite, drop, or halt inputs before they enter model context."""
        ...

    async def before_model(self, ctx: Ctx) -> ModelAction:
        """Return steering messages or halt before a model call."""
        ...

    async def pre_tool_use(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        """Allow, deny, or suspend a requested tool call."""
        ...

    async def after_tool_call(self, ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        """Record and validate a tool result, propagating failures."""
        ...

    async def before_finish(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        """Accept completion or veto it with steering for another round."""
        ...

    async def on_resume(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> ToolDecision:
        """Revalidate a parked call and its answer under the current policy."""
        ...

    async def on_suspend(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        """Persist a continuation before its suspension is announced."""
        ...


def gate(
    answer: Callable[[ToolCall], Awaitable[dict[str, Any] | None]],
) -> Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]:
    """Adapt a simple tool predicate to a ``pre_tool_use`` control stage."""

    async def stage(ctx: Ctx, call: ToolCall) -> ToolDecision:
        decision = await answer(call)
        if decision is None or decision.get("type") == "allow":
            return Continue()
        return Suspend(decision) if decision.get("type") == "suspend" else Deny(decision)

    return stage


def writer(
    record: Callable[[ToolCall, dict[str, Any]], Awaitable[None]],
) -> Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]:
    """Adapt a simple result writer to an ``after_tool_call`` control stage."""

    async def stage(ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        await record(call, result)

    return stage


# ── the composition rules, one per control point ─────────────────────────────


class Ingress:
    """Chain input screens so each sees the previous screen's output."""

    def __init__(
        self,
        *screens: Callable[[Ctx, list[PendingInput]], Awaitable[list[PendingInput] | Halt]],
    ) -> None:
        """Initialize the ordered input screens."""
        self._screens = screens

    async def __call__(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        """Apply input screens until one halts or all complete."""
        for screen in self._screens:
            match await screen(ctx, inputs):
                case Halt() as stop:
                    return stop
                case screened:
                    inputs = screened
        return inputs


class Permissions:
    """Compose tool gates so denial wins and allowance does not short-circuit."""

    def __init__(self, *stages: Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]) -> None:
        """Initialize the ordered permission stages."""
        self._stages = stages

    async def __call__(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        """Evaluate all stages needed to produce the final tool decision."""
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
    """Run result writers in order and propagate the first failure."""

    def __init__(
        self, *writers: Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Initialize the ordered result writers."""
        self._writers = writers

    async def __call__(self, ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        """Write a tool result through every registered writer."""
        for write in self._writers:
            await write(ctx, call, result)


class FinishPolicy:
    """Compose completion verifiers and accumulate steering from vetoes."""

    def __init__(
        self, *gates: Callable[[Ctx, StopReason], Awaitable[TurnDecision]]
    ) -> None:
        """Initialize the ordered completion verifiers."""
        self._gates = gates

    async def __call__(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        """Return accumulated veto steering or preserve the original stop reason."""
        keep_going: Proceed | None = None
        for gate in self._gates:
            match await gate(ctx, reason):
                case Proceed(steers):
                    keep_going = Proceed([*(keep_going.steers if keep_going else []), *steers])
                case _:
                    pass
        return keep_going or Halt(reason)


class Steering:
    """Accumulate pre-model steering in order unless a source halts."""

    def __init__(self, *sources: Callable[[Ctx], Awaitable[ModelAction]]) -> None:
        """Initialize the ordered steering sources."""
        self._sources = sources

    async def __call__(self, ctx: Ctx) -> ModelAction:
        """Collect steering messages or return the first halt."""
        steers: list[BaseMessage] = []
        for source in self._sources:
            match await source(ctx):
                case Halt() as stop:
                    return stop
                case Proceed(more):
                    steers += more
        return Proceed(steers)


class Suspending:
    """Run all suspension persisters before suspension is announced."""

    def __init__(
        self,
        *persisters: Callable[
            [Ctx, ToolCall, dict[str, Any], list[BaseMessage], list[dict[str, Any]]],
            Awaitable[None],
        ],
    ) -> None:
        """Initialize the ordered continuation persisters."""
        self._persisters = persisters

    async def __call__(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[BaseMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        """Persist a suspension through every registered persister."""
        for persist in self._persisters:
            await persist(ctx, call, request, snapshot, completed)


class ControlPlane:
    """Assemble optional control points with permissive defaults."""

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
        """Initialize the configured control-point implementations."""
        self._on_inputs = on_inputs
        self._before_model = before_model
        self._pre_tool_use = pre_tool_use
        self._after_tool_call = after_tool_call
        self._before_finish = before_finish
        self._on_resume = on_resume
        self._on_suspend = on_suspend

    async def on_inputs(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        """Apply input admission or return inputs unchanged."""
        if self._on_inputs is None:
            return inputs
        admitted: list[PendingInput] | Halt = await self._on_inputs(ctx, inputs)
        return admitted

    async def before_model(self, ctx: Ctx) -> ModelAction:
        """Apply pre-model controls or proceed without steering."""
        if self._before_model is None:
            return Proceed()
        action: ModelAction = await self._before_model(ctx)
        return action

    async def pre_tool_use(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        """Apply tool permission controls or allow the call."""
        if self._pre_tool_use is None:
            return Continue()
        decision: ToolDecision = await self._pre_tool_use(ctx, call)
        return decision

    async def after_tool_call(self, ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        """Apply configured result writers and validators."""
        if self._after_tool_call is not None:
            await self._after_tool_call(ctx, call, result)

    async def before_finish(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        """Apply completion controls or preserve the stop reason."""
        if self._before_finish is None:
            return Halt(reason)
        decision: TurnDecision = await self._before_finish(ctx, reason)
        return decision

    async def on_resume(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> ToolDecision:
        """Apply the human answer and current-policy revalidation."""
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
        """Persist a suspension through the configured control."""
        if self._on_suspend is not None:
            await self._on_suspend(ctx, call, request, snapshot, completed)
