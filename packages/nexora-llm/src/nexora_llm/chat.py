"""OpenAI Chat Completions client that streams LangChain chunks.

The loop binds tools and consumes ``astream``. This class is that surface over the
official ``openai`` SDK. HTTP, SSE, retries-off, and error types stay the SDK's.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openai import AsyncOpenAI

from .dsml import recover_dsml_chunks

__all__ = ["ChatModel"]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
XAI_URL = "https://api.x.ai/v1"


@dataclass(frozen=True, slots=True)
class ChatModel:
    """Stream chat completions from any OpenAI-compatible ``/v1`` endpoint.

    ``bind_tools`` / ``astream`` / ``_identifying_params`` are the planner contract.
    Construct with ``base_url`` for OpenRouter, xAI, Groq, vLLM, Ollama, and the rest
    of the OpenAI-compatible wire. Native Anthropic/Google APIs are not this class.
    """

    model: str
    api_key: str | None = None
    base_url: str | None = None
    default_headers: Mapping[str, str] | None = None
    extra_body: Mapping[str, Any] | None = None
    tools: tuple[dict[str, Any], ...] = ()
    client: Any = None
    timeout: float | None = None
    """Seconds before a request is abandoned. None leaves the SDK's own default, which
    is generous: a hung provider holds a worker for ten minutes without one."""
    recover_dsml: bool = False
    """Repair tool markup a gateway left in assistant content. A no-op for a provider
    that does not leak it, so presets turn it on per gateway rather than per model."""

    def bind_tools(self, tools: Sequence[Any], **_kwargs: Any) -> ChatModel:
        """Return a copy that sends these OpenAI-format tool definitions."""
        encoded = tuple(_as_openai_tool(tool) for tool in tools)
        return ChatModel(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers,
            extra_body=self.extra_body,
            tools=encoded,
            client=self.client,
            timeout=self.timeout,
            recover_dsml=self.recover_dsml,
        )

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Stable identity used by the durable model request key."""
        return {
            "model": self.model,
            "base_url": self.base_url or "",
            "tools": list(self.tools),
        }

    async def astream(
        self, messages: Iterable[BaseMessage], **_kwargs: Any
    ) -> AsyncIterator[AIMessageChunk]:
        """Yield one LangChain chunk per provider delta, repaired where a gateway leaks."""
        deltas = self._deltas(messages)
        repaired = recover_dsml_chunks(deltas) if self.recover_dsml else deltas
        async for chunk in repaired:
            yield chunk

    async def _deltas(
        self, messages: Iterable[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        """The provider's own stream, one LangChain chunk per delta."""
        stream = await self._openai().chat.completions.create(
            model=self.model,
            messages=encode_messages(list(messages)),
            stream=True,
            stream_options={"include_usage": True},
            **self._create_options(),
        )
        async for chunk in stream:
            yield decode_chunk(chunk)

    def _openai(self) -> Any:
        """Return the injected client or construct the official async SDK client."""
        if self.client is not None:
            return self.client
        options: dict[str, Any] = {}
        if self.timeout is not None:
            options["timeout"] = self.timeout
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=dict(self.default_headers) if self.default_headers else None,
            # Retries stay off here. A retried model call is a second turn the ledger
            # never saw, and deciding whether that is safe is the caller's, not ours.
            max_retries=0,
            **options,
        )

    def _create_options(self) -> dict[str, Any]:
        """Keyword arguments that only exist when the caller set them."""
        options: dict[str, Any] = {}
        if self.tools:
            options["tools"] = list(self.tools)
        if self.extra_body:
            options["extra_body"] = dict(self.extra_body)
        return options


def encode_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Turn LangChain messages into Chat Completions message dicts."""
    return [_encode_one(message) for message in messages]


def decode_chunk(chunk: Any) -> AIMessageChunk:
    """Turn one SDK stream chunk into the chunk type the loop already adds."""
    choice = (getattr(chunk, "choices", None) or [None])[0]
    delta = getattr(choice, "delta", None) if choice is not None else None
    text = getattr(delta, "content", None) if delta is not None else None
    content: str | list[dict[str, Any]] = text or ""
    reasoning = _extra(delta, "reasoning")
    if reasoning:
        blocks: list[dict[str, Any]] = [
            {"type": "reasoning", "reasoning": str(reasoning), "index": 0},
        ]
        if text:
            blocks.append({"type": "text", "text": str(text)})
        content = blocks
    kwargs: dict[str, Any] = {}
    details = _extra(delta, "reasoning_details")
    if details:
        kwargs["reasoning_details"] = details
    usage = getattr(chunk, "usage", None)
    usage_metadata = _usage(usage)
    model_name = getattr(chunk, "model", None)
    return AIMessageChunk(
        content=content,
        additional_kwargs=kwargs,
        tool_call_chunks=_tool_call_chunks(delta),
        usage_metadata=usage_metadata or None,
        response_metadata={"model_name": model_name} if model_name else {},
    )


def _encode_one(message: BaseMessage) -> dict[str, Any]:
    """Encode one LangChain message to the Chat Completions shape."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": _text(message)}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": _text(message)}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text(message),
        }
    if isinstance(message, AIMessage):
        body: dict[str, Any] = {"role": "assistant", "content": _text(message) or None}
        if message.tool_calls:
            body["tool_calls"] = [_encode_tool_call(call) for call in message.tool_calls]
        if details := message.additional_kwargs.get("reasoning_details"):
            body["reasoning_details"] = details
        return body
    return {"role": "user", "content": _text(message)}


def _encode_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """Encode one LangChain tool call as an OpenAI function tool call."""
    arguments = call.get("args", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call.get("id") or "",
        "type": "function",
        "function": {"name": call.get("name") or "", "arguments": arguments},
    }


def _as_openai_tool(tool: Any) -> dict[str, Any]:
    """Accept already-normalized OpenAI tool dicts from ``as_model_tools``."""
    if isinstance(tool, Mapping) and tool.get("type") == "function":
        return dict(tool)
    if isinstance(tool, Mapping) and "name" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or tool.get("schema") or {},
            },
        }
    raise TypeError(f"unsupported tool definition: {type(tool)!r}")


def _tool_call_chunks(delta: Any) -> list[dict[str, Any]]:
    """Extract streaming tool-call fragments from a delta."""
    calls = getattr(delta, "tool_calls", None) if delta is not None else None
    if not calls:
        return []
    chunks: list[dict[str, Any]] = []
    for call in calls:
        function = getattr(call, "function", None)
        chunks.append(
            {
                "name": getattr(function, "name", None) or "",
                "args": getattr(function, "arguments", None) or "",
                "id": getattr(call, "id", None) or "",
                "index": getattr(call, "index", None) or 0,
            }
        )
    return chunks


def _usage(usage: Any) -> dict[str, int]:
    """Map SDK usage onto the names the loop already reads."""
    if usage is None:
        return {}
    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    completion = (
        getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
    )
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    return {
        "input_tokens": int(prompt),
        "output_tokens": int(completion),
        "total_tokens": int(total),
    }


def _extra(delta: Any, key: str) -> Any:
    """Read an undocumented delta field without depending on one SDK attribute layout."""
    if delta is None:
        return None
    value = getattr(delta, key, None)
    if value:
        return value
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, Mapping):
        return extra.get(key)
    return None


def _text(message: BaseMessage) -> str:
    """Flatten message content to the string Chat Completions still accepts."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text") or block.get("reasoning") or "")
            for block in content
            if isinstance(block, dict)
        ]
        return "".join(parts)
    return str(content or "")


def openrouter(model: str, *, api_key: str | None = None, **options: Any) -> ChatModel:
    """OpenRouter preset: OpenAI wire plus the attribution headers their docs ask for."""
    return ChatModel(
        model,
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_URL,
        default_headers={
            "HTTP-Referer": options.pop("referer", "https://nexora.dev"),
            "X-Title": options.pop("title", "Nexora"),
        },
        extra_body=options.pop("extra_body", None),
        # OpenRouter is the gateway observed dropping DeepSeek's markup into content.
        # The repair costs a suffix scan per delta and does nothing to a clean stream.
        recover_dsml=options.pop("recover_dsml", True),
        **options,
    )


def xai(model: str, *, api_key: str | None = None, **options: Any) -> ChatModel:
    """XAI preset: Grok over the OpenAI-compatible ``api.x.ai`` endpoint."""
    return ChatModel(
        model,
        api_key=api_key or os.environ.get("XAI_API_KEY"),
        base_url=XAI_URL,
        **options,
    )
