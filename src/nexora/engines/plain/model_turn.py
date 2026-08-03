"""Folding a provider's chunk stream into one assistant turn.

Separate from the loop because the streaming protocol has its own edge cases — tool arguments
that arrive in fragments, snapshot providers that emit no deltas at all — and they are worth
testing without running a whole turn.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from ...contracts.types import ToolCall


@dataclass
class ModelTurn:
    """Accumulates one assistant turn, chunk by chunk.

    Everything the provider reported is kept, including fields nothing reads yet. A provider
    only says these once: `thinking` in particular cannot be recovered later, and Anthropic
    expects a turn's reasoning blocks echoed back when the conversation continues through a
    tool call.
    """

    text: str = ""
    thinking: str = ""
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    _partial: dict[str, list[str]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def absorb(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Take one chunk. Returns an event to emit, or None when there is nothing to show."""
        kind = chunk["type"]

        if kind == "text_delta" and chunk["delta"]:
            self.text += chunk["delta"]
            return {"type": "text", "text": chunk["delta"]}

        if kind == "thinking_delta" and chunk["delta"]:
            self.thinking += chunk["delta"]
            return {"type": "thinking", "content": chunk["delta"]}

        if kind == "tool_call_start" and chunk["id"] not in self._partial:
            self._partial[chunk["id"]] = [chunk["name"], ""]
            self._order.append(chunk["id"])

        elif kind == "tool_call_delta" and chunk["id"] in self._partial:
            self._partial[chunk["id"]][1] += chunk["delta"]

        elif kind == "done":
            # A snapshot-style provider emits no deltas at all, so `done.content` wins.
            if chunk.get("content"):
                self.text = chunk["content"]
            self.stop_reason = chunk.get("stop_reason", self.stop_reason)
            self.usage = chunk.get("usage") or self.usage

        return None

    def tool_calls(self) -> list[ToolCall]:
        """The calls this turn requested, in the order the model issued them."""
        calls = []
        for call_id in self._order:
            name, raw_arguments = self._partial[call_id]
            calls.append(ToolCall(call_id, name, parse_arguments(raw_arguments)))
        return calls


def parse_arguments(raw: str) -> Any:
    """Arguments arrive as JSON fragments; malformed JSON degrades to no-args.

    One bad tool call becomes a call with empty arguments rather than a crashed turn.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
