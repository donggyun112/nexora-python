"""Adapter that consumes an agent engine stream into one terminal outcome."""

from collections.abc import Awaitable, Callable
from typing import Any

from .contracts.types import Tools
from .orchestrator import run_agent

__all__ = ["drive"]


async def drive(
    engine: Any,
    model: Any,
    tools: Tools,
    *,
    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    **hooks: Any,
) -> dict[str, Any]:
    """Run an engine and return its completed terminal outcome.

    Args:
        engine: Agent engine callable.
        model: LangChain chat model.
        tools: Tool executor exposed to the engine.
        on_event: Optional event observer.
        **hooks: Engine-specific collaborators and hooks.

    Returns:
        Completed terminal event.
    """
    return await run_agent(engine(model, tools, **hooks), on_event)
