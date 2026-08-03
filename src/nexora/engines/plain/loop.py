"""The ReAct loop, ported from Nexora's `packages/architectures/src/react.ts`.

Control flow and nothing else: the model, the tools, and every policy hook are injected. The
loop runs until something tells it to stop — there is no built-in iteration cap, because how
long an agent may run is the caller's decision (see `ShouldStopAfterTurn`).

The model is a LangChain `BaseChatModel`, so provider differences, tool binding, and the
reassembly of tool arguments that arrive as JSON fragments all happen below this file.
"""

from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ...contracts.events import EventType
from ...contracts.types import (
    Aborted,
    BaseMessage,
    BeforeToolCall,
    DrainSteers,
    Emit,
    OnSuspend,
    ShouldStopAfterTurn,
    StopReason,
    Tools,
)
from ...history import suspend_history_snapshot
from ...tools import execute_calls, render_for_model, select_for_execution, terminates_loop


async def react_loop(
    model: Any,
    tools: Tools,
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[BaseMessage] | None = None,
    aborted: Aborted = lambda: False,
    before_tool_call: BeforeToolCall | None = None,
    emit: Emit | None = None,
    drain_steers: DrainSteers | None = None,
    should_stop_after_turn: ShouldStopAfterTurn | None = None,
    on_suspend: OnSuspend | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Reason, act, repeat. Yields events as they happen."""

    messages: list[BaseMessage] = [
        *([SystemMessage(system_prompt)] if system_prompt else []),
        *(history or []),
        HumanMessage(prompt),
    ]
    available = tools.list()
    bound = model.bind_tools(_as_openai_tools(available)) if available else model
    calls_made: list[dict[str, Any]] = []
    spent: Counter[str] = Counter()
    last_text = ""
    turn = -1

    while True:
        turn += 1
        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return
        if drain_steers:
            messages += drain_steers()

        # ── Reason ───────────────────────────────────────────────────────────
        reply: AIMessageChunk | None = None
        try:
            async for chunk in bound.astream(messages):
                # Chunks add, and the sum reassembles tool arguments that arrived as JSON
                # fragments — the one part of streaming worth not writing ourselves.
                reply = chunk if reply is None else reply + chunk
                if isinstance(chunk.content, str) and chunk.content:
                    yield {"type": "text", "text": chunk.content}
        except Exception as failure:
            # A provider failure ends the run as a reported error rather than an exception
            # escaping into the caller's event loop. Cancellation is not an `Exception`, so it
            # still propagates.
            if aborted():
                yield await _done(emit, last_text, calls_made, "aborted", spent)
            else:
                # A failed run has to reach the event log too, or the audit record just stops.
                if emit is not None:
                    await emit(
                        EventType.STOP_FAILURE, {"reason": "error", "message": str(failure)}
                    )
                yield {"type": "error", "message": str(failure)}
            return

        turn_text = _text_of(reply)
        last_text = turn_text
        spent.update(_usage_of(reply))

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        requested = select_for_execution(tools, list(reply.tool_calls) if reply else [])

        if not requested:
            messages.append(AIMessage(turn_text))
            # A steer that landed while the turn was finishing cancels the stop.
            if drain_steers and (steers := drain_steers()):
                messages += steers
                continue
            yield await _done(emit, turn_text, calls_made, "completed", spent)
            return

        # ── Act ──────────────────────────────────────────────────────────────
        messages.append(AIMessage(content=turn_text, tool_calls=list(requested)))
        for call in requested:
            calls_made.append({"name": call["name"], "input": call["args"]})
            yield {
                "type": "tool_call",
                "id": call["id"],
                "name": call["name"],
                "input": call["args"],
            }

        completed: list[dict[str, Any]] = []
        a_tool_ended_the_run = False

        for call, result in await execute_calls(
            tools, requested, aborted, before_tool_call, emit, turn
        ):
            failed = result.get("type") == "error"
            yield {
                "type": "tool_result",
                "id": call["id"],
                "name": call["name"],
                "result": result,
                "is_error": failed,
            }

            if result.get("type") == "suspend":
                yield {
                    "type": "suspended",
                    "pending_id": result["pending_id"],
                    "tool_call_id": call["id"],
                }
                if on_suspend:
                    snapshot = suspend_history_snapshot(
                        messages, call["id"] or "", [c["id"] for c in completed]
                    )
                    await on_suspend(call, result, snapshot, completed)
                return

            rendered = render_for_model(result)
            completed.append({"id": call["id"], "content": rendered, "is_error": failed})
            messages.append(
                ToolMessage(
                    content=rendered,
                    tool_call_id=call["id"] or "",
                    status="error" if failed else "success",
                )
            )
            if not failed and terminates_loop(tools, call):
                a_tool_ended_the_run = True

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        # ── Stop? ────────────────────────────────────────────────────────────
        # Always asked, even when a tool already ended the run: the hook is where budget and
        # verification accounting lives, and it must see every completed round.
        policy_says_stop = should_stop_after_turn is not None and await should_stop_after_turn(
            turn, turn_text, calls_made
        )
        if a_tool_ended_the_run or policy_says_stop:
            reason: StopReason = "tool" if a_tool_ended_the_run else "policy"
            yield await _done(
                emit, last_text or "(stopped after turn)", calls_made, reason, spent
            )
            return


def _as_openai_tools(available: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our tool descriptions in the shape `bind_tools` accepts."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for tool in available
    ]


def _text_of(reply: AIMessageChunk | None) -> str:
    if reply is None:
        return ""
    return reply.content if isinstance(reply.content, str) else reply.text()


def _usage_of(reply: AIMessageChunk | None) -> dict[str, int]:
    """Token counts if the provider reported any. Absent and zero are different facts."""
    usage = getattr(reply, "usage_metadata", None) if reply is not None else None
    if not usage:
        return {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


async def _done(
    emit: Emit | None,
    content: str,
    calls_made: list[dict[str, Any]],
    reason: StopReason,
    spent: Counter[str],
) -> dict[str, Any]:
    """The single terminal event. Every exit but `suspended` goes through here.

    `stop_reason` exists so an abort leaves a record: without it a cancelled run and a finished
    one both look like a stream that simply ended.
    """
    if emit is not None:
        await emit(EventType.STOP, {"reason": reason, "content": content})
    done: dict[str, Any] = {
        "type": "done",
        "content": content,
        "tool_calls": calls_made,
        "stop_reason": reason,
    }
    if spent:
        done["usage"] = dict(spent)
    return done
