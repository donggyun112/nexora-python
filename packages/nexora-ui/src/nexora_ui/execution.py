"""Bridge a Nexora attempt to newline-delimited UI frames."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from loguru import logger
from nexora import Subagents
from nexora.contracts import EventType, Tools
from nexora.orchestrator import AgentAborted, AgentFailed, AgentSuspended
from nexora.runtime import AgentRuntime

from .children import subagents
from .config import SETTINGS
from .state import STATE, RuntimeState, Session, SimulatedWorkerCrash

AgentEvent = Callable[[dict[str, Any]], Awaitable[None]]
Attempt = Callable[[AgentRuntime, Tools, AgentEvent], Awaitable[dict[str, Any]]]


def frame(kind: str, **payload: Any) -> str:
    """Encode one newline-delimited UI frame."""
    return json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str) + "\n"


def _remember_opened(session: Session, payload: dict[str, Any]) -> None:
    """Record the address announced for an independent child run."""
    if payload.get("mode") != "none":
        return
    child = str(payload.get("run_id", ""))
    if child and all(item["run_id"] != child for item in session.opened):
        session.opened.append({"run_id": child, "agent": str(payload.get("agent_id", ""))})


async def capped(turn: int, _text: str, _calls: list[dict[str, Any]]) -> bool:
    """Stop the demo loop after its eighth turn."""
    return turn >= 7


async def stream_attempt(
    run_id: str, attempt: Attempt, state: RuntimeState = STATE, *, model: str = ""
) -> AsyncIterator[str]:
    """Run one attempt and stream its events as UI frames."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    session = state.session(run_id)

    async def publish(event_type: str, payload: dict[str, Any]) -> None:
        """Queue a lifecycle event and record independent child addresses."""
        if event_type == EventType.SUBAGENT_START.value:
            _remember_opened(session, payload)
        await queue.put({"kind": "lifecycle", "type": str(event_type), "payload": payload})

    def on_child_event(agent: str, event: dict[str, Any]) -> None:
        """Forward observable child events to the UI stream."""
        if event.get("type") in {"text", "tool_call", "tool_result", "done", "error"}:
            queue.put_nowait({"kind": "child", "agent": agent, "event": event})

    async def on_event(event: dict[str, Any]) -> None:
        """Queue model-visible planner events for the conversation rail."""
        # The lifecycle rail is the canonical audit stream. The agent stream also needs the
        # planner-facing call and result pair so an operator can read the conversation the model
        # actually saw. `done` stays represented by the outcome frame.
        # A refused call has a tool-result-shaped stand-in for the model, but it is not an effect
        # result. The lifecycle `permission_request`/`permission_denied` frame renders that gate.
        if event.get("type") == "tool_result" and event.get("executed") is False:
            return
        # `thinking` rides the same rail as `text`: it is what the model did before answering, and
        # reading the answer without it is reading half the turn.
        if event.get("type") in {"text", "thinking", "tool_call", "tool_result"}:
            await queue.put({"kind": "agent", "event": event})

    async def run_attempt() -> None:
        """Execute the attempt and translate its terminal state into a frame."""
        runtime = AgentRuntime(store=state.step_store, emit=publish)
        toolbox = Subagents(
            session.tools,
            subagents(model or SETTINGS.default_model, state.step_store, state.transcripts),
            deliver=runtime.background_sink(run_id),
            registry=session.tasks,
            run_id=run_id,
            emit=publish,
            on_child_event=on_child_event,
        )
        await queue.put({"kind": "meta", "run_id": run_id})
        try:
            outcome = await attempt(runtime, toolbox, on_event)
            await queue.put({"kind": "outcome", "outcome": outcome})
        except AgentSuspended as stopped:
            await queue.put(
                {
                    "kind": "suspended",
                    "pending_id": stopped.pending_id,
                    "tool_call_id": stopped.tool_call_id,
                    "pending": [list(pair) for pair in stopped.pending],
                }
            )
        except SimulatedWorkerCrash as failure:
            # Logged as well as framed. A frame reaches whoever is watching the browser; the crash
            # itself is not an event, so without this line an effect that committed and then lost
            # its worker leaves nothing behind but `POST /api/run 200 OK`.
            logger.warning("run {} crashed after a committed step: {}", run_id, failure)
            await queue.put(
                {
                    "kind": "recoverable",
                    "message": str(failure),
                    "tool_call_id": failure.step,
                }
            )
        except (AgentAborted, AgentFailed) as failure:
            logger.warning("run {} ended without an outcome: {}", run_id, failure)
            await queue.put({"kind": "error", "message": str(failure)})
        except Exception as failure:
            # With a stack, because this branch catches what nobody anticipated. Turning an
            # unknown failure into a one-line frame and dropping the traceback is how a bug here
            # becomes unreproducible.
            logger.exception("run {} failed", run_id)
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
