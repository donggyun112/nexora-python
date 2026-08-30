"""Recover DeepSeek DSML tool markup that a provider left as assistant text.

DeepSeek V4 writes its tool calls as a markup block using fullwidth bars, and some
gateways (OpenRouter among them) fail to lift that into OpenAI ``tool_calls``,
streaming it as assistant content instead. Repairing it belongs to the provider
client: the planner speaks LangChain tool calls and should never learn a vendor's
markup dialect.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk

_BAR = "\uff5c"  # FULLWIDTH VERTICAL LINE, what DeepSeek actually emits
_DSML = rf"(?:\|DSML\||{_BAR}DSML{_BAR})"
_INVOKE = re.compile(
    rf"<{_DSML}invoke\s+name=\"([^\"]+)\"\s*>(.*?)</{_DSML}invoke>",
    re.DOTALL,
)
_PARAM = re.compile(
    rf"<{_DSML}parameter\s+name=\"([^\"]+)\"(?:\s+string=\"(true|false)\")?\s*>"
    rf"(.*?)</{_DSML}parameter>",
    re.DOTALL,
)
_OPENS = (
    "<|DSML|tool_calls>",
    "<|DSML|function_calls>",
    "<|DSML|invoke ",
    "<|DSML|invoke>",
)


def _ascii(text: str) -> str:
    """Normalise the fullwidth bars so one set of tags matches both spellings."""
    return text.replace(_BAR, "|")


def parse_dsml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Turn DSML markup into LangChain tool_call dicts. Empty if none."""
    calls: list[dict[str, Any]] = []
    for name, body in _INVOKE.findall(text or ""):
        args: dict[str, Any] = {}
        for key, is_str, raw in _PARAM.findall(body):
            value: Any = raw
            if is_str == "false":
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value = raw
            args[key] = value
        calls.append(
            {
                "name": name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }
        )
    return calls


def strip_dsml(text: str) -> str:
    """Drop DSML markup, including an unfinished open tag at the end."""
    if not text:
        return ""
    start = _open_index(text)
    if start is not None:
        return text[:start].rstrip()
    held = _prefix_len(text)
    return text[:-held].rstrip() if held else text


def _open_index(text: str) -> int | None:
    hits = [_ascii(text).find(tag) for tag in _OPENS]
    found = [index for index in hits if index >= 0]
    return min(found) if found else None


def _prefix_len(text: str) -> int:
    """Longest suffix of ``text`` that is a prefix of a DSML open tag."""
    norm = _ascii(text)
    best = 0
    for tag in _OPENS:
        limit = min(len(tag), len(norm))
        for size in range(1, limit + 1):
            if norm.endswith(tag[:size]):
                best = max(best, size)
    return best


class DsmlFilter:
    """Hold streamed deltas until they are either ordinary text or DSML markup."""

    def __init__(self) -> None:
        """Start with nothing held and nothing swallowed."""
        self._held = ""
        self._swallow = False

    @property
    def swallowed(self) -> bool:
        """True once a real open tag arrived and the rest is markup."""
        return self._swallow

    @property
    def markup(self) -> str:
        """The swallowed block, for parsing. Empty while nothing has been swallowed."""
        return self._held if self._swallow else ""

    def push(self, delta: str) -> str:
        """Return the visible slice of ``delta``; empty when it belongs to DSML."""
        if not delta:
            return ""
        if self._swallow:
            self._held += delta
            return ""
        self._held += delta
        start = _open_index(self._held)
        if start is not None:
            visible = self._held[:start]
            self._held = self._held[start:]
            self._swallow = True
            return visible
        held = _prefix_len(self._held)
        if not held:
            visible, self._held = self._held, ""
            return visible
        visible, self._held = self._held[:-held], self._held[-held:]
        return visible

    def finish(self) -> str:
        """Flush a prefix that never became markup. Swallowed DSML stays hidden."""
        if self._swallow or "DSML" in _ascii(self._held):
            self._swallow = True
            return ""
        visible, self._held = self._held, ""
        return visible


def _chunk_text(chunk: Any) -> str:
    """The plain text of a chunk, or empty when its content is not a string."""
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(chunk, "content", "")
    return content if isinstance(content, str) else ""


def _has_native_calls(chunk: Any) -> bool:
    """True when the gateway did lift the call, leaving nothing to recover."""
    return bool(getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None))


async def recover_dsml_chunks(
    chunks: AsyncIterator[AIMessageChunk],
) -> AsyncIterator[AIMessageChunk]:
    """Hide leaked markup mid-stream and close with the calls parsed out of it.

    A delta that is entirely the start of an open tag is held back, because the next
    one decides whether it was markup or a stray character. Held chunks are dropped
    once later output carries them, and released unchanged when the block never
    completed — a sentence ending in ``<`` is a sentence, not a tool call.
    """
    pending: list[AIMessageChunk] = []
    filt = DsmlFilter()
    last: AIMessageChunk | None = None
    native = False
    async for chunk in chunks:
        last = chunk
        if _has_native_calls(chunk):
            held = filt.finish()
            if held:
                yield AIMessageChunk(content=held)
            pending.clear()
            native = True
            yield chunk
            continue
        piece = _chunk_text(chunk)
        visible = filt.push(piece)
        if visible:
            pending.clear()
            yield chunk.model_copy(update={"content": visible}) if piece != visible else chunk
        elif piece:
            pending.append(chunk)
    leftover = filt.finish()
    if leftover:
        yield AIMessageChunk(content=leftover)
        return
    if native:
        return
    calls = parse_dsml_tool_calls(filt.markup)
    if not calls:
        for held_chunk in pending:
            yield held_chunk
        return
    yield AIMessageChunk(
        content="",
        tool_calls=calls,
        id=getattr(last, "id", None),
        response_metadata=getattr(last, "response_metadata", {}) or {},
    )
