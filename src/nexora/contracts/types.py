"""What the engines and their collaborators agree on.

Messages and tool calls are LangChain's (`BaseMessage`, `ToolCall`) rather than ours. Owning
those types bought a translation layer in every direction and a provider adapter that
re-implemented `ChatOpenAI`; the types themselves were never the interesting part. What stays
ours is the loop's control-flow contract — the hooks below, and the event vocabulary in
`events.py`.

Tool results stay plain tagged dicts. Their `type` values:

    text | image | content | error | suspend
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from langchain_core.messages import BaseMessage, ToolCall

__all__ = [
    "Aborted",
    "BaseMessage",
    "BeforeToolCall",
    "DrainSteers",
    "Emit",
    "OnSuspend",
    "ShouldStopAfterTurn",
    "StopReason",
    "ToolCall",
    "Tools",
]

StopReason = Literal["completed", "aborted", "tool", "policy"]
"""Why a run ended, carried on the terminal `done` event.

`aborted` matters most: a cancelled run must leave a record, not just a stream that stops.
"""


class Tools(Protocol):
    """Runs tool calls and describes them. Implementations own permissions and sandboxing."""

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]: ...

    def get(self, name: str) -> dict[str, Any] | None:
        """The tool's definition, read for its `is_exclusive` / `terminates_loop` flags."""
        ...

    def list(self) -> list[dict[str, Any]]:
        """Every available tool as `{name, description, parameters}`, for binding to a model."""
        ...


# ── Hooks ────────────────────────────────────────────────────────────────────

Aborted = Callable[[], bool]
"""True once the run has been cancelled. Checked at every round boundary."""

DrainSteers = Callable[[], list[BaseMessage]]
"""Pops user messages injected mid-run. Returns [] when nothing is queued."""

ShouldStopAfterTurn = Callable[[int, str, list[dict[str, Any]]], Awaitable[bool]]
"""Asked after every tool round with (turn, text, calls_made). True ends the run.

This is also where an iteration cap lives — the loop does not impose one.
"""

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
"""Publishes an observing event. See `nexora.contracts.events` for the vocabulary.

Observing only — nothing emitted here can change what the loop does next. Anything that
should be able to is a hook with a return value, like `BeforeToolCall`.
"""

BeforeToolCall = Callable[[ToolCall], Awaitable[dict[str, Any] | None]]
"""The policy gate, consulted before every tool call.

Returns a tool result to stand in for the call, or None to allow it:

    None                      allow
    {"type": "error", ...}    deny
    {"type": "suspend", ...}  ask — checkpoint the turn, do not block on an answer

There is one approval path, not two. An agent asking a human (`handraise`) and a policy
refusing to let a tool run both end as a suspension, so both can take days and neither holds
a worker while waiting. Gates that block synchronously can only ever support approvals that
fit inside a transport timeout.
"""

OnSuspend = Callable[
    [ToolCall, dict[str, Any], list[BaseMessage], list[dict[str, Any]]], Awaitable[None]
]
"""Called with (call, suspend_result, history_snapshot, completed_results) when a tool suspends.

The whole suspend result is handed over, not just its `pending_id`: what a suspension record
needs beyond the key — who suspended, and any external handle the tool minted — lives in
fields the loop has no business naming. Persisting it is the caller's job.
"""
