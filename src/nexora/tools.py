"""Reading tool definitions and tool results.

Everything the loop needs to know *about* a tool, as opposed to how to run one. Pure — the
loop's two trickiest rules (exclusive calls, terminating calls) are verifiable here without
driving a turn.
"""

import json
from typing import Any

from .contracts.events import EventType
from .contracts.types import Aborted, BeforeToolCall, Emit, ToolCall, Tools


async def execute_calls(
    tools: Tools,
    calls: list[ToolCall],
    aborted: Aborted,
    before_tool_call: BeforeToolCall | None = None,
    emit: Emit | None = None,
    turn: int = 0,
) -> list[tuple[ToolCall, dict[str, Any]]]:
    """Run a round of calls and return their results in the order the model issued them.

    `before_tool_call` is the policy gate. It returns a tool result to stand in for the call,
    or None to let it run — which makes its three outcomes the ones the loop already knows:

        None                      allow  → the tool runs
        {"type": "error", ...}    deny   → the model sees a failed call and can react
        {"type": "suspend", ...}  ask    → the turn is checkpointed and stops here

    `ask` deliberately does not block waiting for a human. A gate that awaits an answer holds
    the worker for as long as the person takes, which caps approvals at whatever timeout the
    transport allows. Suspending instead costs nothing while stopped, so a policy approval can
    take days — the same machinery an agent-initiated handraise already uses.

    An executor exposing `execute_batch` gets the allowed calls at once and applies its own
    concurrency policy — only it knows which of its tools are safe to run together. Otherwise
    the calls run one at a time.

    Either way the round is cut short by a suspend or an abort: the remaining calls are simply
    absent from the result, and the model re-issues them on resume.
    """
    gate = _Gate(before_tool_call, emit, turn)
    run_batch = getattr(tools, "execute_batch", None)

    if run_batch is not None:
        results = await _execute_batched(run_batch, calls, gate)
    else:
        results = []
        for call in calls:
            if aborted():
                break
            decision = await gate.decide(call)
            result = (
                decision
                if decision is not None
                else await tools.execute(call["name"], call["id"] or "", call["args"])
            )
            results.append((call, result))
            if result.get("type") == "suspend":
                break

    await _announce(emit, turn, results)
    return results


class _Gate:
    """The policy gate plus the events that record what it decided."""

    def __init__(self, before_tool_call: BeforeToolCall | None, emit: Emit | None, turn: int):
        self._before_tool_call = before_tool_call
        self._emit = emit
        self._turn = turn

    async def decide(self, call: ToolCall) -> dict[str, Any] | None:
        """None to run the call, or the result that stands in for it."""
        await self._publish(EventType.PRE_TOOL_USE, call, {})
        if self._before_tool_call is None:
            return None

        decision = await self._before_tool_call(call)
        if decision is None:
            return None

        kind = decision.get("type")
        if kind == "error":
            await self._publish(EventType.PERMISSION_DENIED, call, {"reason": decision})
        elif kind == "suspend":
            await self._publish(EventType.PERMISSION_REQUEST, call, {"request": decision})
        return decision

    async def _publish(self, event: EventType, call: ToolCall, extra: dict[str, Any]) -> None:
        if self._emit is None:
            return
        await self._emit(
            event,
            {
                "turn": self._turn,
                "call_id": call["id"],
                "name": call["name"],
                "input": call["args"],
                **extra,
            },
        )


async def _announce(
    emit: Emit | None,
    turn: int,
    results: list[tuple[ToolCall, dict[str, Any]]],
) -> None:
    """Report each finished call, then the round as a whole."""
    if emit is None:
        return
    for call, result in results:
        failed = result.get("type") == "error"
        await emit(
            EventType.POST_TOOL_USE_FAILURE if failed else EventType.POST_TOOL_USE,
            {
                "turn": turn,
                "call_id": call["id"],
                "name": call["name"],
                "result": result,
            },
        )
    await emit(
        EventType.POST_TOOL_BATCH,
        {"turn": turn, "calls": [{"call_id": c["id"], "name": c["name"]} for c, _ in results]},
    )


async def _execute_batched(
    run_batch: Any,
    calls: list[ToolCall],
    gate: _Gate,
) -> list[tuple[ToolCall, dict[str, Any]]]:
    """Gate every call first, then hand the executor only what it may run.

    Gating has to finish before the batch starts: once the executor is running calls
    concurrently there is no way to stop the ones a later gate would have refused.
    """
    decided: list[tuple[ToolCall, dict[str, Any] | None]] = []
    for call in calls:
        decision = await gate.decide(call)
        decided.append((call, decision))
        if decision is not None and decision.get("type") == "suspend":
            break

    allowed = [call for call, decision in decided if decision is None]
    batch = [{"call_id": c["id"], "name": c["name"], "input": c["args"]} for c in allowed]
    ran = await run_batch(batch) if allowed else []
    by_id = {call["id"]: result for call, result in _in_call_order(allowed, ran)}

    results: list[tuple[ToolCall, dict[str, Any]]] = []
    for call, decision in decided:
        if decision is not None:
            results.append((call, decision))
        elif call["id"] in by_id:
            results.append((call, by_id[call["id"]]))
    return results


def _in_call_order(
    calls: list[ToolCall],
    batch_results: list[dict[str, Any]],
) -> list[tuple[ToolCall, dict[str, Any]]]:
    """Re-sort a batch executor's results, keeping only what it actually answered.

    A suspend means the executor stopped early, so calls it never reported are dropped rather
    than faked. A call reported as missing when nothing suspended is a broken executor, and
    surfaces as an error result instead of vanishing.
    """
    by_id = {result["call_id"]: result for result in batch_results}
    suspended = any(r["result"].get("type") == "suspend" for r in batch_results)
    answered = [c for c in calls if c["id"] in by_id] if suspended else calls

    ordered: list[tuple[ToolCall, dict[str, Any]]] = []
    for call in answered:
        result = by_id.get(call["id"])
        if result is None:
            missing = f"Missing tool result: {call['id']}"
            ordered.append((call, {"type": "error", "message": missing}))
        else:
            ordered.append((call, result["result"]))
    return ordered


def select_for_execution(tools: Tools, calls: list[ToolCall]) -> list[ToolCall]:
    """An exclusive call runs alone; the model re-issues the others next round."""
    for call in calls:
        if _flag(tools, call, "is_exclusive"):
            return [call]
    return calls


def terminates_loop(tools: Tools, call: ToolCall) -> bool:
    """Whether this call ends the run when it succeeds.

    Only on success: a failed submit/finish tool gets a recovery round instead.
    """
    return _flag(tools, call, "terminates_loop")


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
