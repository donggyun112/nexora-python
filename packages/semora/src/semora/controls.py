"""Define runtime control points and their composition policies.

Control points return load-bearing decisions before the runtime acts. Unlike observation events,
their ordering and failure behavior are part of the execution contract. Stages may run again
during recovery and therefore must be idempotent.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

from pydantic_ai.messages import ModelMessage, ModelRequestPart

from .contracts import PendingInput, StopReason, ToolCall

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
    "controls_of",
    "gate",
    "writer",
]


@dataclass(frozen=True, slots=True)
class Ctx:
    """Provide shared run context to every control point."""

    turn: int
    messages: list[ModelMessage] = field(default_factory=list)
    calls_made: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    """Assistant text of the current round."""
    subject: str = ""
    """Who the run acts for, as the host names them, carried verbatim and never interpreted."""


class Continue(NamedTuple):
    """Allow control flow to continue without changes."""


class Deny(NamedTuple):
    """Deny a tool call and provide its model-visible result."""

    result: Any


class Suspend(NamedTuple):
    """Suspend a tool call with its complete external request. Must carry a `pending_id`."""

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
    """Continue, optionally injecting steering parts into the next model request first."""

    steers: Sequence[ModelRequestPart] = ()


class Halt(NamedTuple):
    """End the run with a terminal stop reason."""

    reason: StopReason


TurnDecision = Proceed | Halt

OnInputs = Callable[[Ctx, list[PendingInput]], Awaitable[list[PendingInput] | Halt]]
BeforeModel = Callable[[Ctx], Awaitable[TurnDecision]]
PreToolUse = Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]
PostToolUse = Callable[[Ctx, ToolCall, Any], Awaitable[None]]
BeforeFinish = Callable[[Ctx, StopReason], Awaitable[TurnDecision]]
OnResume = Callable[[Ctx, ToolCall, ResumeInput], Awaitable[ToolDecision]]
OnSuspend = Callable[
    [Ctx, ToolCall, dict[str, Any], list[ModelMessage], list[dict[str, Any]]], Awaitable[None]
]


@runtime_checkable
class Controls(Protocol):
    """Define the policy decisions requested at runtime execution boundaries."""

    async def on_inputs(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        """Rewrite, drop, or halt inputs before they enter model context."""
        ...

    async def before_model(self, ctx: Ctx) -> TurnDecision:
        """Return steering parts or halt before a model call."""
        ...

    async def pre_tool_use(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        """Allow, deny, or suspend a requested tool call."""
        ...

    async def post_tool_use(self, ctx: Ctx, call: ToolCall, result: Any) -> None:
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
        snapshot: list[ModelMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        """Persist a continuation before its suspension is announced."""
        ...


def gate(
    answer: Callable[[ToolCall], Awaitable[dict[str, Any] | None]],
) -> PreToolUse:
    """Adapt a simple tool predicate to a ``pre_tool_use`` control stage.

    ``None`` or ``{"type": "allow"}`` allows, ``{"type": "suspend", "pending_id": ...}`` parks,
    anything else denies with itself as the model-visible result.
    """

    async def stage(ctx: Ctx, call: ToolCall) -> ToolDecision:
        decision = await answer(call)
        if decision is None or decision.get("type") == "allow":
            return Continue()
        return Suspend(decision) if decision.get("type") == "suspend" else Deny(decision)

    return stage


def writer(record: Callable[[ToolCall, Any], Awaitable[None]]) -> PostToolUse:
    """Adapt a simple result writer to a ``post_tool_use`` control stage."""

    async def stage(ctx: Ctx, call: ToolCall, result: Any) -> None:
        await record(call, result)

    return stage


# ── the composition rules, one per control point ─────────────────────────────


class Ingress:
    """Chain input screens so each sees the previous screen's output."""

    def __init__(self, *screens: OnInputs) -> None:
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

    def __init__(self, *stages: PreToolUse) -> None:
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

    def __init__(self, *writers: PostToolUse) -> None:
        """Initialize the ordered result writers."""
        self._writers = writers

    async def __call__(self, ctx: Ctx, call: ToolCall, result: Any) -> None:
        """Write a tool result through every registered writer."""
        for write in self._writers:
            await write(ctx, call, result)


class FinishPolicy:
    """Compose completion verifiers and accumulate steering from vetoes."""

    def __init__(self, *gates: BeforeFinish) -> None:
        """Initialize the ordered completion verifiers."""
        self._gates = gates

    async def __call__(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        """Return accumulated veto steering or preserve the original stop reason."""
        keep_going: Proceed | None = None
        for verify in self._gates:
            match await verify(ctx, reason):
                case Proceed(steers):
                    keep_going = Proceed([*(keep_going.steers if keep_going else []), *steers])
                case _:
                    pass
        return keep_going or Halt(reason)


class Steering:
    """Accumulate pre-model steering in order unless a source halts."""

    def __init__(self, *sources: BeforeModel) -> None:
        """Initialize the ordered steering sources."""
        self._sources = sources

    async def __call__(self, ctx: Ctx) -> TurnDecision:
        """Collect steering parts or return the first halt."""
        steers: list[ModelRequestPart] = []
        for source in self._sources:
            match await source(ctx):
                case Halt() as stop:
                    return stop
                case Proceed(more):
                    steers += more
        return Proceed(steers)


class Suspending:
    """Run all suspension persisters before suspension is announced."""

    def __init__(self, *persisters: OnSuspend) -> None:
        """Initialize the ordered continuation persisters."""
        self._persisters = persisters

    async def __call__(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[ModelMessage],
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
        on_inputs: OnInputs | None = None,
        before_model: BeforeModel | None = None,
        pre_tool_use: PreToolUse | None = None,
        post_tool_use: PostToolUse | None = None,
        before_finish: BeforeFinish | None = None,
        on_resume: OnResume | None = None,
        on_suspend: OnSuspend | None = None,
    ) -> None:
        """Initialize the configured control-point implementations."""
        self._on_inputs = on_inputs
        self._before_model = before_model
        self._pre_tool_use = pre_tool_use
        self._post_tool_use = post_tool_use
        self._before_finish = before_finish
        self._on_resume = on_resume
        self._on_suspend = on_suspend

    async def on_inputs(self, ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        """Apply input admission or return inputs unchanged."""
        if self._on_inputs is None:
            return inputs
        return await self._on_inputs(ctx, inputs)

    async def before_model(self, ctx: Ctx) -> TurnDecision:
        """Apply pre-model controls or proceed without steering."""
        if self._before_model is None:
            return Proceed()
        return await self._before_model(ctx)

    async def pre_tool_use(self, ctx: Ctx, call: ToolCall) -> ToolDecision:
        """Apply tool permission controls or allow the call."""
        if self._pre_tool_use is None:
            return Continue()
        return await self._pre_tool_use(ctx, call)

    async def post_tool_use(self, ctx: Ctx, call: ToolCall, result: Any) -> None:
        """Apply configured result writers and validators."""
        if self._post_tool_use is not None:
            await self._post_tool_use(ctx, call, result)

    async def before_finish(self, ctx: Ctx, reason: StopReason) -> TurnDecision:
        """Apply completion controls or preserve the stop reason."""
        if self._before_finish is None:
            return Halt(reason)
        return await self._before_finish(ctx, reason)

    async def on_resume(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> ToolDecision:
        """Apply the human answer and current-policy revalidation. A human denial stands."""
        if resume.answer.get("type") == "error":
            return Deny(resume.answer)
        if self._on_resume is None:
            return Continue()
        return await self._on_resume(ctx, call, resume)

    async def on_suspend(
        self,
        ctx: Ctx,
        call: ToolCall,
        request: dict[str, Any],
        snapshot: list[ModelMessage],
        completed: list[dict[str, Any]],
    ) -> None:
        """Persist a suspension through the configured control."""
        if self._on_suspend is not None:
            await self._on_suspend(ctx, call, request, snapshot, completed)


CONTROL_POINTS = (
    "on_inputs",
    "before_model",
    "pre_tool_use",
    "post_tool_use",
    "before_finish",
    "on_resume",
    "on_suspend",
)
"""The seven decision points, in loop order."""


def controls_of(agent: object) -> Controls | None:
    """The control plane an agent carries as methods, or `None` when it defines none.

    A class that subclasses `pydantic_ai.Agent` and defines `pre_tool_use`, `before_finish` and
    the rest as methods is its own default policy. An explicit `controls=` at run time replaces
    it whole, so one agent can run under different rules.
    """
    found = {name: getattr(agent, name) for name in CONTROL_POINTS if hasattr(agent, name)}
    return ControlPlane(**found) if found else None
