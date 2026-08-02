"""What the loop and its collaborators agree on.

Provider chunks, tool results and emitted events stay as tagged dicts — they cross a wire and
their shape is the provider's or the tool author's, not ours. Their `type` values:

    chunk    text_delta | thinking_delta | tool_call_start | tool_call_delta | done
    result   text | image | content | error | suspend
    event    text | thinking | tool_call | tool_result | suspended | done
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool_result"]
    content: str | list[dict[str, Any]]


StopReason = Literal["completed", "aborted", "tool", "policy"]
"""Why a run ended, carried on the terminal `done` event.

`aborted` matters most: a cancelled run must leave a record, not just a stream that stops.
"""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Any


class LLM(Protocol):
    def stream(self, messages: list[LLMMessage]) -> AsyncIterator[dict[str, Any]]: ...


class Tools(Protocol):
    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]: ...

    def get(self, name: str) -> dict[str, Any] | None:
        """The tool's definition, read for its `is_exclusive` / `terminates_loop` flags."""
        ...

    def list(self) -> list[dict[str, Any]]:
        """Every available tool as `{name, description, parameters}`.

        The `while` loop never needs this — the model was already told what it may call. An
        engine that builds the graph from the tool set does.
        """
        ...


# ── Hooks ────────────────────────────────────────────────────────────────────

Aborted = Callable[[], bool]
"""True once the run has been cancelled. Checked at every round boundary."""

DrainSteers = Callable[[], list[LLMMessage]]
"""Pops user messages injected mid-run. Returns [] when nothing is queued."""

ShouldStopAfterTurn = Callable[[int, str, list[dict[str, Any]]], Awaitable[bool]]
"""Asked after every tool round with (turn, text, calls_made). True ends the run.

This is also where an iteration cap lives — the loop does not impose one.
"""

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]
"""Publishes an observing event. See `nexora.events` for the vocabulary and the envelope.

Observing only — nothing the loop emits here can change what it does next. Anything that
should be able to is a hook with a return value, like `BeforeToolCall`.
"""

BeforeToolCall = Callable[["ToolCall"], Awaitable[dict[str, Any] | None]]
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
    ["ToolCall", dict[str, Any], list[LLMMessage], list[dict[str, Any]]], Awaitable[None]
]
"""Called with (call, suspend_result, history_snapshot, completed_results) when a tool suspends.

The whole suspend result is handed over, not just its `pending_id`: what a suspension record
needs beyond the key — who suspended, and any external handle the tool minted — lives in
fields the loop has no business naming. Persisting it is the caller's job.
"""
