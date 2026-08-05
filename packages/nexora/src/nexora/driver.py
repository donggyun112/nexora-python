"""One attempt at an agent, driven until it returns a value.

The runtime owns fresh/resume/recover and turns each external input into a queue item. This adapter
only consumes the already-wired engine stream and returns its terminal value.
"""

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
    """Drive `engine` to a finished outcome, or raise if it did not finish.

    `run_agent` supplies the raising: `error`, `suspended` and `aborted` are not outcomes, so a step
    that records what this returns cannot record a non-answer.

    `hooks` goes to the engine untouched (`permissions`, `emit`, `aborted`, …). Passing them
    through rather than naming them keeps this from becoming a second place engine arguments are
    declared, which is a place they can be forgotten.
    """
    return await run_agent(engine(model, tools, **hooks), on_event)
