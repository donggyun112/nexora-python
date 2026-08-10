"""Reading tool definitions and tool results.

Everything the loop needs to know *about* a tool, as opposed to how to run one. Pure — the
loop's two trickiest rules (exclusive calls, terminating calls) are verifiable here without
driving a turn.
"""

import asyncio
import json
from typing import Any, NamedTuple, Protocol

from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from nexora_store import Contended, Fenced, Indeterminate

from .contracts.events import EventType
from .contracts.types import (
    Aborted,
    BaseMessage,
    BatchTools,
    ControlSignal,
    Emit,
    ToolCall,
    Tools,
)
from .controls import Continue, Controls, Ctx, Deny, Suspend, ToolDecision


class Resolved(NamedTuple):
    """How one call ended, and enough about it for the engine to publish the right event.

    `refused` is the part the result alone cannot tell: an `error` from the gate and an `error`
    from a tool look identical as results, but one is `permission_denied` and the other is
    `post_tool_use_failure`. The engine publishes, so the engine needs to know which.
    """

    call: ToolCall
    result: dict[str, Any]
    refused: bool
    """The gate stood in for the call. The tool never ran."""


class RoundSuspended(ControlSignal):
    """Report a tool round suspended without an orchestrator to persist it."""

    def __init__(self, resolved: list[Resolved]) -> None:
        """Initialize the error with results completed before suspension."""
        super().__init__(
            "tool round suspended, and nothing here can park it: a suspension has to commit its "
            "continuation. Drive the loop through `AgentRuntime`, pass "
            "`execute_round=orchestrator.execute_round`, or let the gate deny instead of suspend"
        )
        self.resolved = resolved


class InvalidToolResult(RuntimeError):
    """A tool returned a control-plane result reserved for permission gates."""


class InvalidToolCall(RuntimeError):
    """A requested round cannot be keyed, so none of it may run."""


def require_call_ids(calls: list[ToolCall]) -> list[ToolCall]:
    """Refuse a round whose calls cannot be told apart. Checked before any effect.

    The call id is the idempotency key, so it is also the step name. LangChain types it
    as optional and a provider or adapter can hand over a round with a missing or repeated one —
    and then the ledger cannot say which call a result belongs to.

    The whole round is refused rather than the offending call filtered out, for two reasons. A
    filtered call would sit in the assistant message forever with no `ToolMessage` answering it,
    which is exactly what `_unanswered_tool_calls` reads as "still pending". And a result cannot be
    returned for a call with no id at all: `ToolMessage` needs one. Refusing early is also what
    keeps the round from half-executing — the failure this replaced ran the first tool and raised
    `duplicate step name` on the second, after the effect.
    """
    seen: set[str] = set()
    for position, call in enumerate(calls):
        call_id = call.get("id")
        if not call_id:
            raise InvalidToolCall(
                f"tool call {position} ({call.get('name')!r}) has no id; "
                "the id is the idempotency key and the durable step name"
            )
        if call_id in seen:
            raise InvalidToolCall(
                f"tool call id {call_id!r} appears twice in one round; "
                "a ledger keyed by call id cannot tell the two apart"
            )
        seen.add(call_id)
    return calls


def validate_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Reject approval suspension after execution; approval belongs to ``pre_tool_use``."""
    if result.get("type") == "suspend":
        raise InvalidToolResult(
            f"tool {name!r} returned reserved result type 'suspend'; "
            "request approval from pre_tool_use before executing the tool"
        )
    return result


class ExecuteRound(Protocol):
    """Define execution ownership for one requested tool round."""

    async def __call__(
        self,
        tools: Tools,
        calls: list[ToolCall],
        aborted: Aborted,
        emit: Emit | None = None,
        turn: int = 0,
        controls: Controls | None = None,
        ctx: Ctx | None = None,
    ) -> list[Resolved]:
        """Execute a tool round and return results in model call order."""
        ...


async def execute_calls(
    tools: Tools,
    calls: list[ToolCall],
    aborted: Aborted,
    emit: Emit | None = None,
    turn: int = 0,
    controls: Controls | None = None,
    ctx: Ctx | None = None,
) -> list[Resolved]:
    """Run a round of calls and return their outcomes in the order the model issued them.

    `controls` is the engine's contract with whoever owns policy — `pre_tool_use` decides,
    `after_tool_call` records. Both are awaited calls whose return value this function acts on,
    which is what lets the *order* of the stages behind them be the policy. Events are published
    after each decision so a UI and an audit log can see it; answering one changes nothing.

    A `BatchTools` executor gets the allowed calls at once and applies its own concurrency
    policy — only it knows which of its tools are safe to run together, since safety is a
    property of a pair of calls and not of one tool. Otherwise the calls run one at a time,
    which is the fail-closed default.

    Either way the round is cut short by a suspend or an abort: the remaining calls are simply
    absent from the result, and the model re-issues them on resume.
    """
    here = ctx if ctx is not None else Ctx(turn=turn)
    if isinstance(tools, BatchTools):
        batch_resolved = await _execute_batched(
            tools.execute_batch, calls, aborted, emit, here, controls
        )
        if any(item.result.get("type") == "suspend" for item in batch_resolved):
            raise RoundSuspended(batch_resolved)
        return batch_resolved

    resolved: list[Resolved] = []
    for call in calls:
        if aborted():
            break
        decision = await decide_tool_call(controls, emit, here, call)
        stood_in = _stands_in_for(decision)
        result = (
            stood_in
            if stood_in is not None
            else await _execute_validated(tools, call["name"], call["id"] or "", call["args"])
        )
        # A policy result stands in for an effect; it must not claim that the tool ran. Actual
        # executions are recorded before append so a failed record leaves the call unresolved.
        if stood_in is None:
            await record_resolved(controls, emit, here, call, result)
        resolved.append(Resolved(call, result, refused=stood_in is not None))
        if result.get("type") == "suspend":
            break
    if any(item.result.get("type") == "suspend" for item in resolved):
        raise RoundSuspended(resolved)
    return resolved


def _stands_in_for(decision: ToolDecision) -> dict[str, Any] | None:
    """The result a refusal supplies, or None when the call may run.

    A `Deny` and a `Suspend` both carry a tool-result-shaped dict, so downstream needs no special
    case: the model sees a failed call either way. `Resolved.refused` keeps the difference the
    result cannot, because the *event* differs even when the result does not.
    """
    match decision:
        case Deny(result) | Suspend(result):
            return result
        case _:
            return None


async def decide_tool_call(
    controls: Controls | None, emit: Emit | None, ctx: Ctx, call: ToolCall
) -> ToolDecision:
    """Ask the control point, then say publicly what it decided."""
    if emit is not None:
        await emit(EventType.PRE_TOOL_USE, tool_payload(ctx, call))
    decision: ToolDecision = (
        await controls.pre_tool_use(ctx, call) if controls is not None else Continue()
    )
    if emit is not None:
        match decision:
            case Deny(result):
                await emit(
                    EventType.PERMISSION_DENIED,
                    tool_payload(ctx, call, reason=result, source="pre_tool_use"),
                )
            case Suspend(request):
                await emit(
                    EventType.PERMISSION_REQUEST,
                    tool_payload(ctx, call, request=request, source="pre_tool_use"),
                )
            case _:
                pass
    return decision


async def record_resolved(
    controls: Controls | None,
    emit: Emit | None,
    ctx: Ctx,
    call: ToolCall,
    result: dict[str, Any],
) -> None:
    """The durable record first — it may raise — then the observing event."""
    if controls is not None:
        await controls.after_tool_call(ctx, call, result)
    if emit is None:
        return
    failed = result.get("type") == "error"
    await emit(
        EventType.POST_TOOL_USE_FAILURE if failed else EventType.POST_TOOL_USE,
        tool_payload(ctx, call, result=result),
    )


class Stepped:
    """Wrap each tool call in a durable step keyed by its call identifier."""

    def __init__(self, tools: Tools, orchestrator: Any) -> None:
        """Initialize the wrapper with an executor and orchestrator."""
        self._tools = tools
        self._orchestrator = orchestrator

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Execute or replay a tool call through its durable step."""
        result: dict[str, Any] = await self._orchestrator.run(
            call_id,
            lambda: _execute_validated(self._tools, name, call_id, arguments),
        )
        return result

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a wrapped tool definition by name."""
        return self._tools.get(name)

    def list(self) -> list[dict[str, Any]]:
        """Return all wrapped tool definitions."""
        return self._tools.list()


class Concurrent:
    """Execute a batch concurrently only when every call opts in to concurrency."""

    def __init__(self, tools: Tools, aborted: Aborted = lambda: False) -> None:
        """Initialize the wrapper with an executor and cancellation predicate."""
        self._tools = tools
        self._aborted = aborted
        # Who turns a raise into an error result. `Stepped` already did it around the user's tool,
        # inside the step, which is the only place that can tell a failed tool from a failed
        # runtime. Catching a second time out here would convert exactly what it let through — a
        # lost lease, an indeterminate step, a store that died mid-write — into a sentence the
        # model reads as a bad file path and moves past.
        self._converts_failures = not isinstance(tools, Stepped)

    async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute a batch concurrently when safe and sequentially otherwise."""
        if concurrency_safe_batch(self._tools, [_as_tool_call(call) for call in calls]):
            if self._aborted():
                return []
            # A tool's own failure is an error result by the time it gets here, so what still
            # raises is a runtime signal or a broken tool result — either way the round ends.
            # `return_exceptions` so that ending does not abandon the calls beside it. Plain
            # `gather` propagates the first exception and leaves the rest running: their effects
            # still land, after the round is over and with nothing awaiting them. Cancelling
            # instead would be worse — a tool cut mid-write leaves the ledger saying it never ran.
            # So the round is awaited whole, every finished call reaches its step, and then the
            # failure is raised.
            together: list[dict[str, Any]] = []
            for outcome in await asyncio.gather(
                *(self._one(call) for call in calls), return_exceptions=True
            ):
                if isinstance(outcome, BaseException):
                    raise outcome
                together.append(outcome)
            return together
        resolved: list[dict[str, Any]] = []
        for call in calls:
            if self._aborted():
                break
            item = await self._one(call)
            resolved.append(item)
            if item["result"].get("type") == "suspend":
                break
        return resolved

    async def _one(self, call: dict[str, Any]) -> dict[str, Any]:
        result = await self.execute(call["name"], call["call_id"], call["input"])
        return {"call_id": call["call_id"], "result": result}

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Execute one validated tool call."""
        if self._converts_failures:
            return await _execute_validated(self._tools, name, call_id, arguments)
        return validate_tool_result(name, await self._tools.execute(name, call_id, arguments))

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a wrapped tool definition by name."""
        return self._tools.get(name)

    def list(self) -> list[dict[str, Any]]:
        """Return all wrapped tool definitions."""
        return self._tools.list()


class Absorbed(NamedTuple):
    """A finished round read into the four things the loop does next.

    Pulled out of the loop because it is the only part of a round that is pure — given the
    outcomes, everything here is derivable, so it is testable without driving a turn and the
    `while` stays about control flow. The live and recovery paths both call it, so a round has one
    interpretation.
    """

    answers: list[BaseMessage]
    """`ToolMessage`s for the calls that resolved. A suspension contributes none — its answer
    arrives on resume."""
    completed: list[dict[str, Any]]
    suspended: tuple[ToolCall, dict[str, Any]] | None
    """The first suspension, if any. The round is still absorbed whole before it is reported."""
    ended_by_tool: bool
    """A terminating tool succeeded. Only on success — a failed one gets a recovery round."""


def absorb_round(tools: Tools, resolved: list[Resolved]) -> Absorbed:
    """Read a round's outcomes. Does not stop at a suspension.

    Calls that already finished are external facts, so dropping them because a *different* call
    suspended means the resumed run re-issues them — a second write, a second charge. react.ts
    absorbs the whole batch and stops after (`suspended ??=`, inside its loop over `toolResults`);
    so does this.
    """
    answers: list[BaseMessage] = []
    completed: list[dict[str, Any]] = []
    suspended: tuple[ToolCall, dict[str, Any]] | None = None
    ended_by_tool = False

    for call, result, _refused in resolved:
        if result.get("type") == "suspend":
            suspended = suspended or (call, result)
            continue
        failed = result.get("type") == "error"
        rendered = render_for_model(result)
        completed.append({"id": call["id"], "content": rendered, "is_error": failed})
        answers.append(
            ToolMessage(
                content=rendered,
                tool_call_id=call["id"] or "",
                status="error" if failed else "success",
            )
        )
        if not failed and terminates_loop(tools, call):
            ended_by_tool = True

    return Absorbed(answers, completed, suspended, ended_by_tool)


def tool_result(call: ToolCall, result: dict[str, Any]) -> dict[str, Any]:
    """The one definition of the public `tool_result` event."""
    return {
        "type": "tool_result",
        "id": call["id"],
        "name": call["name"],
        "result": result,
        "is_error": result.get("type") == "error",
    }


def as_model_tools(available: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our `{name, description, parameters}` definitions in the shape `bind_tools` accepts.

    `convert_to_openai_tool` is LangChain's own normalizer and already recognises this shape, so
    there is nothing here to write. It replaced two hand-rolled copies of the conversion, one per
    engine, which had **already drifted** — they disagreed about what an empty `parameters` should
    default to. One definition is the point; the fact that it is a one-liner is the reward.

    **Sorted by name** (`registry.ts`'s `assemble`). Providers cache on a prefix of the request,
    and the tool
    schemas sit in it, so a `Tools.list()` that returns the same set in a different order is a cache
    miss for every turn after it — paid in tokens, invisible in behaviour. `list()` is the host's to
    implement and cannot be made to promise an order, so the guarantee belongs at the one place the
    definitions cross into a model call rather than in each implementation.
    """
    return [
        convert_to_openai_tool(definition)
        for definition in sorted(available, key=lambda d: str(d.get("name", "")))
    ]


def tool_payload(ctx: Ctx, call: ToolCall, **extra: Any) -> dict[str, Any]:
    """The payload shape every tool event shares. One definition, so the engines cannot drift.

    Takes the whole `Ctx` and not a bare `turn` so that `subject` rides along everywhere at once.
    Passed per field, an audit stamp is only ever on the events somebody remembered to add it to,
    and the one it is missing from is the one an incident asks about. Omitted when the host named
    nobody, rather than reported as an empty string that reads like a finding.
    """
    return {
        "turn": ctx.turn,
        "call_id": call["id"],
        "name": call["name"],
        "input": call["args"],
        **({"subject": ctx.subject} if ctx.subject else {}),
        **extra,
    }


async def _execute_batched(
    run_batch: Any,
    calls: list[ToolCall],
    aborted: Aborted,
    emit: Emit | None,
    ctx: Ctx,
    controls: Controls | None,
) -> list[Resolved]:
    """Decide every call first, then hand the executor only what it may run.

    Gating has to finish before the batch starts: once the executor is running calls
    concurrently there is no way to stop the ones a later answer would have refused.
    """
    decided: list[tuple[ToolCall, dict[str, Any] | None]] = []
    for call in calls:
        if aborted():
            break
        decision = _stands_in_for(await decide_tool_call(controls, emit, ctx, call))
        decided.append((call, decision))
        if decision is not None and decision.get("type") == "suspend":
            break

    allowed = [call for call, decision in decided if decision is None]
    batch = [{"call_id": c["id"], "name": c["name"], "input": c["args"]} for c in allowed]
    ran = await run_batch(batch) if allowed else []
    by_id = {
        call["id"]: result for call, result in _in_call_order(allowed, ran, incomplete_ok=aborted())
    }

    resolved: list[Resolved] = []
    for call, decision in decided:
        if decision is not None:
            resolved.append(Resolved(call, decision, refused=True))
        elif call["id"] in by_id:
            resolved.append(Resolved(call, by_id[call["id"]], refused=False))
    # In call order, not completion order: the record reads the same sequence a retry replays.
    for call, result, refused in resolved:
        if not refused:
            await record_resolved(controls, emit, ctx, call, result)
    return resolved


def _in_call_order(
    calls: list[ToolCall],
    batch_results: list[dict[str, Any]],
    *,
    incomplete_ok: bool = False,
) -> list[tuple[ToolCall, dict[str, Any]]]:
    """Re-sort a batch executor's results, keeping only what it actually answered.

    A suspend means the executor stopped early, so calls it never reported are dropped rather
    than faked. A call reported as missing when nothing suspended is a broken executor, and
    surfaces as an error result instead of vanishing.
    """
    by_id = {result["call_id"]: result for result in batch_results}
    suspended = any(r["result"].get("type") == "suspend" for r in batch_results)
    answered = [c for c in calls if c["id"] in by_id] if suspended or incomplete_ok else calls

    ordered: list[tuple[ToolCall, dict[str, Any]]] = []
    for call in answered:
        result = by_id.get(call["id"])
        if result is None:
            missing = f"Missing tool result: {call['id']}"
            ordered.append((call, {"type": "error", "message": missing}))
        else:
            ordered.append((call, validate_tool_result(call["name"], result["result"])))
    return ordered


_PASSES_THROUGH = (ControlSignal, Contended, Fenced, Indeterminate)
"""The runtime's own signals. Everything else a tool raises is that tool's failure."""


async def _execute_validated(
    tools: Tools, name: str, call_id: str, arguments: Any
) -> dict[str, Any]:
    """Run one tool. A raise becomes the error result the model can read.

    Every path that reaches a tool comes through here, so this is the only place that has to
    catch. react.ts catches per call (`executeBatch` in `tool-executor.ts`) and so does this: a
    tool that raises
    is one failed effect, not a failed round. Letting the exception escape instead ended the run
    in the caller's event loop and discarded the sibling calls' results, which had already been
    computed — the model was never told what went wrong, so it could never try something else.

    Caught here rather than at each call site because the exception is also a fact the *ledger*
    needs. Inside the durable step, an error result commits as `done`, and the call is not
    re-run. That is the honest record: we watched the tool fail, which is a different thing from
    the crash `Indeterminate` exists for, where nobody watched anything.

    Validation stays outside the `try`. `InvalidToolResult` is our contract with the tool author,
    not the tool's report about the world, and answering a broken tool with `[ERROR]` would hide
    the bug behind a message the model reads as a bad file path.
    """
    try:
        result = await tools.execute(name, call_id, arguments)
    except _PASSES_THROUGH:
        raise
    except Exception as failure:
        result = {"type": "error", "message": f"{type(failure).__name__}: {failure}"}
    return validate_tool_result(name, result)


def select_for_execution(tools: Tools, calls: list[ToolCall]) -> list[ToolCall]:
    """An exclusive call runs alone; the model re-issues the others next round.

    Ids are checked here because this runs before the engine appends the assistant message or
    announces a single call — the earliest point where refusing costs nothing.
    """
    require_call_ids(calls)
    for call in calls:
        if _flag(tools, call, "is_exclusive"):
            return [call]
    return calls


def concurrency_safe_batch(tools: Tools, calls: list[ToolCall]) -> bool:
    """Whether the whole batch explicitly opted into concurrent execution."""
    return len(calls) > 1 and all(_flag(tools, call, "is_concurrency_safe") for call in calls)


def terminates_loop(tools: Tools, call: ToolCall) -> bool:
    """Whether this call ends the run when it succeeds.

    Only on success: a failed submit/finish tool gets a recovery round instead.
    """
    return _flag(tools, call, "terminates_loop")


def _as_tool_call(call: dict[str, Any]) -> ToolCall:
    return {
        "id": call["call_id"],
        "name": call["name"],
        "args": call["input"],
        "type": "tool_call",
    }


def _flag(tools: Tools, call: ToolCall, name: str) -> bool:
    """Read a loop flag, which may be a bool or a predicate over the call's arguments."""
    definition = tools.get(call["name"]) or {}
    flag = definition.get(name, False)
    return bool(flag(call["args"]) if callable(flag) else flag)


def render_for_model(result: dict[str, Any]) -> str:
    """Flatten a tool result to the text the model sees.

    Images are summarized, never inlined: the caller attaches them as separate image blocks,
    and serializing them here would leak base64 into the transcript.
    """
    match result:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "error", "message": str(message)}:
            return f"[ERROR] {message}"
        case {"type": "image"}:
            return "[image]"
        case {"type": "content", "blocks": list(blocks)}:
            return "\n".join(block.get("text", "[image]") for block in blocks)
        case _:
            return json.dumps(result, default=str)
