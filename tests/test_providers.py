from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from nexora.providers import FallbackChatModel, ModelProvider


class StreamModel:
    def __init__(self, *items: AIMessageChunk | Exception) -> None:
        self.items = items
        self.calls = 0
        self.bound = False

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> "StreamModel":
        self.bound = True
        return self

    async def astream(self, _messages: Any, **_kwargs: Any) -> AsyncIterator[AIMessageChunk]:
        self.calls += 1
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item


class ServerError(RuntimeError):
    status_code = 503


async def test_fallback_uses_the_next_model_after_a_provider_failure() -> None:
    failed = StreamModel(ServerError("unavailable"))
    healthy = StreamModel(AIMessageChunk(content="done"))
    model = FallbackChatModel(
        [ModelProvider("primary", failed), ModelProvider("secondary", healthy)],
        transient_retry_seconds=0,
    )

    chunks = [chunk async for chunk in model.astream([HumanMessage("hello")])]

    assert [chunk.content for chunk in chunks] == ["done"]
    assert failed.calls == healthy.calls == 1


async def test_fallback_never_duplicates_partial_output() -> None:
    partial = StreamModel(AIMessageChunk(content="visible"), ServerError("dropped"))
    unused = StreamModel(AIMessageChunk(content="duplicate"))
    model = FallbackChatModel(
        [ModelProvider("primary", partial), ModelProvider("secondary", unused)],
        transient_retry_seconds=0,
    )

    stream = model.astream([HumanMessage("hello")])
    assert (await anext(stream)).content == "visible"
    with pytest.raises(ServerError):
        await anext(stream)
    assert unused.calls == 0


async def test_fallback_binds_tools_to_every_candidate() -> None:
    first = StreamModel(RuntimeError("failed"))
    second = StreamModel(AIMessageChunk(content="done"))

    FallbackChatModel(
        [ModelProvider("first", first), ModelProvider("second", second)]
    ).bind_tools([{"name": "read"}])

    assert first.bound and second.bound
