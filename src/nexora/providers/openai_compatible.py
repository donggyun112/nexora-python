"""An `LLM` for any OpenAI-compatible chat completions endpoint (OpenAI, OpenRouter, vLLM).

The normalization the loop depends on happens here, not above it. Two pieces of it are the
provider's shape rather than ours:

* tool calls stream keyed by **index**, and only the first fragment of each carries the `id`
  and name. The adapter remembers index→id so the loop sees a stable id from the first chunk.
* the arguments arrive as JSON fragments that only parse once concatenated. The adapter
  forwards the fragments; `ModelTurn` does the joining.
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ..contracts.types import LLMMessage


class OpenAICompatible:
    """Streams a chat completion and yields the loop's chunk vocabulary."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._tools = tools or []
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )

    async def _chunks(self, messages: list[LLMMessage]) -> AsyncIterator[dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai(m) for m in messages],
            "stream": True,
        }
        if self._tools:
            request["tools"] = [{"type": "function", "function": t} for t in self._tools]

        names: dict[int, str] = {}
        ids: dict[int, str] = {}
        content = ""
        finish = "end_turn"

        stream = await self._client.chat.completions.create(**request)
        async for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta.content:
                content += delta.content
                yield {"type": "text_delta", "delta": delta.content}
            for call in delta.tool_calls or []:
                index = call.index
                if call.id and index not in ids:
                    ids[index] = call.id
                    names[index] = (call.function.name if call.function else None) or ""
                    yield {
                        "type": "tool_call_start",
                        "id": ids[index],
                        "name": names[index],
                    }
                fragment = call.function.arguments if call.function else None
                if fragment and index in ids:
                    yield {"type": "tool_call_delta", "id": ids[index], "delta": fragment}

        yield {"type": "done", "content": content, "stop_reason": finish}

    def stream(self, messages: list[LLMMessage]) -> AsyncIterator[dict[str, Any]]:
        return self._chunks(messages)


def _to_openai(message: LLMMessage) -> dict[str, Any]:
    """One of our messages in the provider's shape."""
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}

    if message["role"] == "tool_result":
        # The provider wants one `tool` message per result, but our history groups a round's
        # results into a single message. Only the first survives this shape; callers with
        # parallel tool calls need `_to_openai_many`.
        first = content[0]
        return {
            "role": "tool",
            "tool_call_id": first["id"],
            "content": first["content"],
        }

    text = "".join(b["text"] for b in content if b["type"] == "text")
    calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b["arguments"])},
        }
        for b in content
        if b["type"] == "tool_call"
    ]
    out: dict[str, Any] = {"role": message["role"], "content": text or None}
    if calls:
        out["tool_calls"] = calls
    return out


def to_openai_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Flatten our history, splitting a grouped tool_result message into one each."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "tool_result" and not isinstance(content, str):
            out += [
                {"role": "tool", "tool_call_id": b["id"], "content": b["content"]}
                for b in content
                if b["type"] == "tool_result"
            ]
        else:
            out.append(_to_openai(message))
    return out
