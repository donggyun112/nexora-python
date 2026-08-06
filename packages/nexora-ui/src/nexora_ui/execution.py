"""Bridge a Nexora attempt to newline-delimited UI frames."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from nexora_orchestrator import AgentAborted, AgentFailed, AgentSuspended

from nexora.runtime import AgentRuntime

from .state import STATE, RuntimeState, SimulatedWorkerCrash
from .tools import DemoTools

AgentEvent = Callable[[dict[str, Any]], Awaitable[None]]
Attempt = Callable[[AgentRuntime, DemoTools, AgentEvent], Awaitable[dict[str, Any]]]


def frame(kind: str, **payload: Any) -> str:
    return json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str) + "\n"


async def capped(turn: int, _text: str, _calls: list[dict[str, Any]]) -> bool:
    return turn >= 7


async def stream_attempt(
    run_id: str, attempt: Attempt, state: RuntimeState = STATE
) -> AsyncIterator[str]:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    session = state.session(run_id)

    async def publish(event_type: str, payload: dict[str, Any]) -> None:
        await queue.put({"kind": "lifecycle", "type": str(event_type), "payload": payload})

    async def on_event(event: dict[str, Any]) -> None:
        # The lifecycle rail is the canonical audit stream. The agent stream also needs the
        # planner-facing call and result pair so an operator can read the conversation the model
        # actually saw. `done` stays represented by the outcome frame.
        # A refused call has a tool-result-shaped stand-in for the model, but it is not an effect
        # result. The lifecycle `permission_request`/`permission_denied` frame renders that gate.
        if event.get("type") == "tool_result" and event.get("executed") is False:
            return
        if event.get("type") in {"text", "tool_call", "tool_result"}:
            await queue.put({"kind": "agent", "event": event})

    async def run_attempt() -> None:
        runtime = AgentRuntime(store=state.step_store, emit=publish)
        await queue.put({"kind": "meta", "run_id": run_id})
        try:
            outcome = await attempt(runtime, session.tools, on_event)
            await queue.put({"kind": "outcome", "outcome": outcome})
        except AgentSuspended as stopped:
            await queue.put(
                {
                    "kind": "suspended",
                    "pending_id": stopped.pending_id,
                    "tool_call_id": stopped.tool_call_id,
                }
            )
        except SimulatedWorkerCrash as failure:
            await queue.put(
                {
                    "kind": "recoverable",
                    "message": str(failure),
                    "tool_call_id": failure.step,
                }
            )
        except (AgentAborted, AgentFailed) as failure:
            await queue.put({"kind": "error", "message": str(failure)})
        except Exception as failure:
            await queue.put({"kind": "error", "message": str(failure)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_attempt())
    try:
        while (item := await queue.get()) is not None:
            yield frame(**item)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
