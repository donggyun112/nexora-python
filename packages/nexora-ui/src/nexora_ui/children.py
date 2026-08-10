"""Subagent roster and runner adapters for the local test console.

Each child uses ``AgentRuntime`` with the parent's step and transcript stores. The runner adapter
exposes runtime callbacks as the event stream required by ``Subagents``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from nexora import Answering, Compiled, Subagent
from nexora.runtime import AgentRuntime
from nexora.subagents import Reply

from .config import SYSTEM_PROMPT
from .provider import openrouter_model
from .recording import Recorder, history_of
from .tools import DemoTools

CHILD_PROMPT = (
    f"{SYSTEM_PROMPT}\n\n"
    "You are a subagent. Another agent delegated this task to you. Do the work with your tools, "
    "then call `respond_to_parent` exactly once with what you found. That call is how your answer "
    "reaches the agent waiting on it, and it ends your run."
)

async def _capped(turn: int, _text: str, _calls: list[dict[str, Any]]) -> bool:
    """Stop a console child after four model turns."""
    return turn >= 3


ROSTER: tuple[tuple[str, str], ...] = (
    ("note-keeper", "Stores and recalls notes. Give it a key and a value, or a key to look up."),
    ("echoer", "Echoes text back through a durable tool effect. Useful for testing a round trip."),
    ("flaky", "Calls an API that always fails. Use it to watch a child report a failure."),
)
"""Names and model-visible descriptions for console subagents."""


def subagents(model: str, store: Any, transcripts: Any) -> list[Subagent]:
    """Build compiled child definitions for the console roster."""
    return [
        Compiled(name, description, _runner(model, store, transcripts))
        for name, description in ROSTER
    ]


def _runner(model: str, store: Any, transcripts: Any) -> Any:
    """Create a runner backed by durable step and transcript stores."""

    async def run(prompt: str, reply: Reply, run_id: str) -> AsyncIterator[dict[str, Any]]:
        async for event in _stream_run(
            AgentRuntime(store=store),
            run_id,
            model,
            Answering(DemoTools(), reply),
            prompt,
            transcripts,
        ):
            yield event

    return run


async def _stream_run(
    runtime: AgentRuntime,
    run_id: str,
    model: str,
    tools: Any,
    prompt: str,
    transcripts: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Run a child and yield its planner events in publication order."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    already_said = await history_of(transcripts, run_id)
    recorder = await Recorder.open(transcripts, run_id, prompt)

    async def on_event(event: dict[str, Any]) -> None:
        await recorder.observe(event)
        await queue.put(event)

    async def drive() -> dict[str, Any]:
        try:
            # Whatever this child already said. Empty on its first turn, and the reason a second
            # turn on the same run id continues rather than starting a stranger.
            return await runtime.run(
                run_id,
                openrouter_model(model),
                tools,
                prompt,
                on_event=on_event,
                system_prompt=CHILD_PROMPT,
                should_stop_after_turn=_capped,
                history=already_said,
            )
        finally:
            await recorder.closed()
            await queue.put(None)

    task = asyncio.create_task(drive())
    terminal = False
    try:
        while (item := await queue.get()) is not None:
            terminal = terminal or item.get("type") in {"done", "error"}
            yield item
    finally:
        # Reached on a cancel too — `cancel_task` closes this generator, and a child left running
        # after its leash was pulled is the thing that cancel exists to prevent.
        if not task.done():
            task.cancel()
    try:
        outcome = await task
    except asyncio.CancelledError:
        raise
    except Exception as failure:
        # The one thing the stream cannot carry: a run that raised instead of ending. `_drain`
        # reads this as a failed child, so the parent is told rather than the exception crossing
        # into its tool round.
        yield {"type": "error", "message": f"{type(failure).__name__}: {failure}"}
        return
    # `on_event` already carried the terminal event, so re-yielding the outcome would show the
    # child finishing twice. Yielded only when nothing terminal came through at all.
    if not terminal:
        yield outcome
