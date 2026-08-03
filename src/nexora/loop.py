"""The ReAct loop, ported from Nexora's `packages/architectures/src/react.ts`.

Control flow and nothing else: the model, the tools, and every policy hook are injected. The
loop runs until something tells it to stop — there is no built-in iteration cap, because how
long an agent may run is the caller's decision (see `ShouldStopAfterTurn`).
"""

from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from .events import EventType
from .history import suspend_history_snapshot
from .model_turn import ModelTurn
from .tools import execute_calls, render_for_model, select_for_execution, terminates_loop
from .types import (
    LLM,
    Aborted,
    BeforeToolCall,
    DrainSteers,
    Emit,
    LLMMessage,
    OnSuspend,
    ShouldStopAfterTurn,
    StopReason,
    ToolCall,
    Tools,
)


async def react_loop(
    llm: LLM,
    tools: Tools,
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[LLMMessage] | None = None,
    aborted: Aborted = lambda: False,
    before_tool_call: BeforeToolCall | None = None,
    emit: Emit | None = None,
    drain_steers: DrainSteers | None = None,
    should_stop_after_turn: ShouldStopAfterTurn | None = None,
    on_suspend: OnSuspend | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Reason, act, repeat. Yields events as they happen."""

    # A `system` role rather than a separate argument to the provider: adapters that need it
    # split out (Anthropic) can lift it back out, and the loop stays provider-neutral.
    preamble: list[LLMMessage] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    messages: list[LLMMessage] = [
        *preamble,
        *(history or []),
        {"role": "user", "content": prompt},
    ]
    calls_made: list[dict[str, Any]] = []
    spent: Counter[str] = Counter()
    """Token usage summed across every round of this run. Counter adds keys for us."""
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
        model_turn = ModelTurn()
        try:
            async for chunk in llm.stream(messages):
                event = model_turn.absorb(chunk)
                if event is not None:
                    yield event
        except Exception as failure:
            # A provider failure ends the run as a reported error, not as an exception
            # escaping into the caller's event loop. Cancellation is not an Exception, so it
            # still propagates.
            if aborted():
                yield await _done(emit, last_text, calls_made, "aborted", spent)
            else:
                yield {"type": "error", "message": str(failure)}
            return
        last_text = model_turn.text
        spent.update(model_turn.usage)

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        requested = select_for_execution(tools, model_turn.tool_calls())

        if not requested:
            messages.append({"role": "assistant", "content": model_turn.text})
            # A steer that landed while the turn was finishing cancels the stop.
            if drain_steers and (steers := drain_steers()):
                messages += steers
                continue
            yield await _done(emit, model_turn.text, calls_made, "completed", spent)
            return

        # ── Act ──────────────────────────────────────────────────────────────
        messages.append(_assistant_turn(model_turn.text, requested))
        for call in requested:
            calls_made.append({"name": call.name, "input": call.arguments})
            yield {
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }

        result_blocks: list[dict[str, Any]] = []
        a_tool_ended_the_run = False

        executed = await execute_calls(tools, requested, aborted, before_tool_call, emit, turn)
        for call, result in executed:
            failed = result.get("type") == "error"
            yield {
                "type": "tool_result",
                "id": call.id,
                "name": call.name,
                "result": result,
                "is_error": failed,
            }

            if result.get("type") == "suspend":
                yield {
                    "type": "suspended",
                    "pending_id": result["pending_id"],
                    "tool_call_id": call.id,
                }
                if on_suspend:
                    snapshot = suspend_history_snapshot(
                        messages, call.id, [b["id"] for b in result_blocks]
                    )
                    await on_suspend(call, result, snapshot, result_blocks)
                return

            result_blocks.append(
                {
                    "type": "tool_result",
                    "id": call.id,
                    "content": render_for_model(result),
                    "is_error": failed,
                }
            )
            if not failed and terminates_loop(tools, call):
                a_tool_ended_the_run = True

        messages.append({"role": "tool_result", "content": result_blocks})
        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        # ── Stop? ────────────────────────────────────────────────────────────
        # Always asked, even when a tool already ended the run: the hook is where budget and
        # verification accounting lives, and it must see every completed round.
        policy_says_stop = should_stop_after_turn is not None and await should_stop_after_turn(
            turn, model_turn.text, calls_made
        )
        if a_tool_ended_the_run or policy_says_stop:
            reason: StopReason = "tool" if a_tool_ended_the_run else "policy"
            yield await _done(emit, last_text or "(stopped after turn)", calls_made, reason, spent)
            return


def _assistant_turn(text: str, calls: list[ToolCall]) -> LLMMessage:
    """The assistant message recording what was said and what was asked for."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    content += [
        {"type": "tool_call", "id": c.id, "name": c.name, "arguments": c.arguments} for c in calls
    ]
    return {"role": "assistant", "content": content}


async def _done(
    emit: Emit | None,
    content: str,
    calls_made: list[dict[str, Any]],
    reason: StopReason,
    spent: Counter[str],
) -> dict[str, Any]:
    """The single terminal event. Every exit but `suspended` goes through here.

    `stop_reason` exists so an abort leaves a record: without it a cancelled run and a finished
    one both look like a stream that simply ended. `usage` is omitted rather than zeroed when
    the provider reported none — nothing spent and nothing measured are different facts.
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
