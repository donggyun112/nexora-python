"""Tool definition normalization, execution, gating, and result handling."""

import asyncio
import json
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from semora_store import Contended, Fenced, Indeterminate

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
    """Tool call outcome with permission-gate provenance."""

    call: ToolCall
    result: dict[str, Any]
    refused: bool


class RoundSuspended(ControlSignal):
    """Report a tool round suspended without an orchestrator to persist it."""

    def __init__(self, resolved: list[Resolved]) -> None:
        """Initialize the error with results completed before suspension."""
        super().__init__(
            "tool round suspended, and nothing here can park it: a suspension has to commit its "
            "continuation. Attach `DurableRuntimeOrchestrator`, use `AgentRuntime(store=...)`, "
            "or let the gate deny instead of suspend"
        )
        self.resolved = resolved


class InvalidToolResult(RuntimeError):
    """A tool returned a control-plane result reserved for permission gates."""


class InvalidToolCall(RuntimeError):
    """A requested round cannot be keyed, so none of it may run."""


def require_call_ids(calls: list[ToolCall]) -> list[ToolCall]:
    """Validate that every call in a round has a unique identifier.

    Args:
        calls: Tool calls to validate before execution.

    Returns:
        The original call list.

    Raises:
        InvalidToolCall: If an identifier is missing or duplicated.
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
    """Execute a gated tool round and preserve model call order."""
    return await _advance_calls(tools, calls, aborted, emit, turn, controls, ctx, replayed={})


async def _advance_calls(
    tools: Tools,
    calls: list[ToolCall],
    aborted: Aborted,
    emit: Emit | None,
    turn: int,
    controls: Controls | None,
    ctx: Ctx | None,
    *,
    replayed: Mapping[str, dict[str, Any]],
) -> list[Resolved]:
    """Advance live or recovering calls through one lifecycle path.

    Batch-capable executors own their concurrency policy. Other executors run calls sequentially.
    Suspension or cancellation ends the round without synthesizing results for unexecuted calls.
    Completed results supplied by recovery skip their gate and effect, but cross the same ordered
    post-effect boundary as results produced by the live attempt.
    """
    here = ctx if ctx is not None else Ctx(turn=turn)
    if isinstance(tools, BatchTools):
        batch_resolved = await _execute_batched(
            tools.execute_batch, calls, aborted, emit, here, controls, replayed
        )
        if any(item.result.get("type") == "suspend" for item in batch_resolved):
            raise RoundSuspended(batch_resolved)
        return batch_resolved

    resolved: list[Resolved] = []
    collecting = False
    for call in calls:
        if aborted():
            break
        call_id = call["id"] or ""
        if call_id in replayed:
            result = replayed[call_id]
            await record_resolved(controls, emit, here, call, result)
            resolved.append(Resolved(call, result, refused=False))
            continue
        if collecting:
            suspended = await _collect_suspension(controls, emit, here, call)
            if suspended is not None:
                resolved.append(Resolved(call, suspended, refused=True))
            continue
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
            collecting = True
    if any(item.result.get("type") == "suspend" for item in resolved):
        raise RoundSuspended(resolved)
    return resolved


def _stands_in_for(decision: ToolDecision) -> dict[str, Any] | None:
    """Return the synthetic tool result supplied by a denial or suspension."""
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
        await emit(EventType.PRE_TOOL_USE, tool_payload(ctx, call, event=EventType.PRE_TOOL_USE))
    decision = await _ask_tool_call(controls, ctx, call)
    await _emit_tool_decision(emit, ctx, call, decision)
    return decision


async def _ask_tool_call(controls: Controls | None, ctx: Ctx, call: ToolCall) -> ToolDecision:
    """Return the load-bearing gate decision without publishing an observation."""
    return await controls.pre_tool_use(ctx, call) if controls is not None else Continue()


async def _emit_tool_decision(
    emit: Emit | None, ctx: Ctx, call: ToolCall, decision: ToolDecision
) -> None:
    """Publish the observable consequence of one already-made gate decision."""
    if emit is not None:
        match decision:
            case Deny(result):
                await emit(
                    EventType.PERMISSION_DENIED,
                    tool_payload(
                        ctx,
                        call,
                        event=EventType.PERMISSION_DENIED,
                        reason=result,
                        source="pre_tool_use",
                    ),
                )
            case Suspend(request):
                await emit(
                    EventType.PERMISSION_REQUEST,
                    tool_payload(
                        ctx,
                        call,
                        event=EventType.PERMISSION_REQUEST,
                        request=request,
                        source="pre_tool_use",
                    ),
                )
            case _:
                pass


async def _collect_suspension(
    controls: Controls | None, emit: Emit | None, ctx: Ctx, call: ToolCall
) -> dict[str, Any] | None:
    """Keep and announce only a suspension after an earlier call already stopped the round."""
    decision = await _ask_tool_call(controls, ctx, call)
    result = _stands_in_for(decision)
    if result is None or result.get("type") != "suspend":
        return None
    if emit is not None:
        await emit(EventType.PRE_TOOL_USE, tool_payload(ctx, call, event=EventType.PRE_TOOL_USE))
    await _emit_tool_decision(emit, ctx, call, decision)
    return result


async def record_resolved(
    controls: Controls | None,
    emit: Emit | None,
    ctx: Ctx,
    call: ToolCall,
    result: dict[str, Any],
) -> None:
    """The durable record first — it may raise — then the observing event."""
    if controls is not None:
        await controls.post_tool_use(ctx, call, result)
    if emit is None:
        return
    failed = result.get("type") == "error"
    outcome = EventType.POST_TOOL_USE_FAILURE if failed else EventType.POST_TOOL_USE
    await emit(outcome, tool_payload(ctx, call, event=outcome, result=result))


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
    """Normalized tool-round state consumed by the planner.

    Attributes:
        answers: One tool answer per completed call, in call order.
        completed: Rendered results carried into a suspension record.
        suspended: Every call that parked the round, in model order; empty when none did.
        ended_by_tool: Whether a terminating call succeeded.
        images: ``(call id, message)`` pairs re-entering images the answers could only name.
        context: Extra model context emitted after a tool answer, such as an invoked skill body.
    """

    answers: list[BaseMessage]
    completed: list[dict[str, Any]]
    suspended: list[tuple[ToolCall, dict[str, Any]]]
    ended_by_tool: bool
    images: list[tuple[str, BaseMessage]]
    context: list[tuple[str, BaseMessage]]


def absorb_round(tools: Tools, resolved: list[Resolved]) -> Absorbed:
    """Normalize a complete tool round without dropping results before suspension."""
    answers: list[BaseMessage] = []
    completed: list[dict[str, Any]] = []
    suspended: list[tuple[ToolCall, dict[str, Any]]] = []
    ended_by_tool = False
    images: list[tuple[str, BaseMessage]] = []
    context: list[tuple[str, BaseMessage]] = []

    for call, result, _refused in resolved:
        if result.get("type") == "suspend":
            suspended.append((call, result))
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
        if blocks := image_blocks(result):
            images.append((call["id"] or "", image_message(call["name"], call["id"] or "", blocks)))
        context.extend((call["id"] or "", message) for message in context_messages(result))
        if not failed and terminates_loop(tools, call):
            ended_by_tool = True

    return Absorbed(answers, completed, suspended, ended_by_tool, images, context)


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
    """Normalize and sort tool definitions for LangChain model binding.

    Sorting stabilizes provider request prefixes and improves prompt-cache reuse.
    """
    return [
        convert_to_openai_tool(definition)
        for definition in sorted(available, key=lambda d: str(d.get("name", "")))
    ]


def tool_payload(ctx: Ctx, call: ToolCall, event: str = "", **extra: Any) -> dict[str, Any]:
    """Build the common payload for tool lifecycle events.

    When ``event`` names the event type, the payload carries a deterministic ``event_id``: the
    same logical event keeps the same identity across recovery re-emission, so consumers can
    deduplicate replays. A reissued permission request has a new pending id and therefore a new
    identity.
    """
    payload = {
        "turn": ctx.turn,
        "call_id": call["id"],
        "name": call["name"],
        "input": call["args"],
        **({"subject": ctx.subject} if ctx.subject else {}),
        **extra,
    }
    if event:
        identity = [str(event), str(ctx.turn), str(call["id"])]
        if extra.get("source"):
            identity.append(str(extra["source"]))
        request = extra.get("request")
        if isinstance(request, dict) and request.get("pending_id"):
            identity.append(str(request["pending_id"]))
        # The unit separator keeps the key canonical: a ":" inside a host-authored pending id
        # must not collide two different identities.
        payload["event_id"] = "\x1f".join(identity)
    return payload


async def _execute_batched(
    run_batch: Any,
    calls: list[ToolCall],
    aborted: Aborted,
    emit: Emit | None,
    ctx: Ctx,
    controls: Controls | None,
    replayed: Mapping[str, dict[str, Any]],
) -> list[Resolved]:
    """Gate a batch before dispatch, replay completed calls, and record in call order."""
    decided: list[
        tuple[ToolCall, dict[str, Any] | None, dict[str, Any] | None]
    ] = []
    collecting = False
    for call in calls:
        if aborted():
            break
        call_id = call["id"] or ""
        if call_id in replayed:
            decided.append((call, None, replayed[call_id]))
            continue
        if collecting:
            suspension = await _collect_suspension(controls, emit, ctx, call)
            if suspension is not None:
                decided.append((call, suspension, None))
            continue
        decision = _stands_in_for(await decide_tool_call(controls, emit, ctx, call))
        decided.append((call, decision, None))
        if decision is not None and decision.get("type") == "suspend":
            collecting = True

    allowed = [
        call for call, decision, recovered in decided if decision is None and recovered is None
    ]
    batch = [{"call_id": c["id"], "name": c["name"], "input": c["args"]} for c in allowed]
    ran = await run_batch(batch) if allowed else []
    by_id = {
        call["id"]: result for call, result in _in_call_order(allowed, ran, incomplete_ok=aborted())
    }

    resolved: list[Resolved] = []
    for call, decision, recovered in decided:
        if recovered is not None:
            resolved.append(Resolved(call, recovered, refused=False))
        elif decision is not None:
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
    """Reorder batch results and handle missing executor responses."""
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
    """Execute one tool and convert tool exceptions into model-visible errors.

    Runtime control signals and invalid result contracts propagate to the caller.
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


def is_read_only(tools: Tools, call: ToolCall) -> bool:
    """Whether this call only reads. Claude Code's `Tool.isReadOnly(input)` is the reference."""
    return _flag(tools, call, "is_read_only")


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


def image_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every image a tool result carries, as LangChain image content blocks.

    A ``content`` result yields one block per image it holds; an ``image`` result yields one.
    """
    match result:
        case {"type": "content", "blocks": list(blocks)}:
            return [block for item in blocks if (block := _image_block(item)) is not None]
        case _:
            single = _image_block(result)
            return [single] if single is not None else []


def _image_block(result: Any) -> dict[str, Any] | None:
    """Translate one tool-result image into a content block, or None if it is not an image."""
    match result:
        case {"type": "image", "data": str(data), "mime_type": str(mime_type)}:
            return {"type": "image", "base64": data, "mime_type": mime_type}
        case _:
            return None


def context_messages(result: dict[str, Any]) -> list[BaseMessage]:
    """Decode trusted tool-provided context that follows, but never replaces, its answer."""
    raw = result.get("context_messages")
    if not isinstance(raw, list):
        return []
    messages: list[BaseMessage] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), str):
            continue
        metadata = item.get("metadata")
        messages.append(
            HumanMessage(
                content=item["content"],
                additional_kwargs={
                    "semora_context": dict(metadata) if isinstance(metadata, Mapping) else {}
                },
            )
        )
    return messages


def image_message(name: str, call_id: str, blocks: list[dict[str, Any]]) -> HumanMessage:
    """Re-enter a tool's images as visual context, since its answer can only name them.

    A user message rather than blocks inside the tool answer: providers disagree on whether a
    tool result may carry images at all, and this is the shape all of them accept.
    """
    one = len(blocks) == 1
    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Tool {name} returned {'an image' if one else f'{len(blocks)} images'} "
                    f"for call {call_id}. Use {'this image' if one else 'these images'} "
                    "as visual context for the current task."
                ),
            },
            *blocks,
        ]
    )


def render_for_model(result: dict[str, Any]) -> str:
    """Flatten a tool result to the text the model sees.

    Images are summarized, never inlined: `image_message` re-attaches them as their own blocks,
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
