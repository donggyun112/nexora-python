"""ChatModel speaks the planner contract over a scripted OpenAI client."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import nexora
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from nexora import ChatModel
from nexora.contracts import PendingInput
from nexora.engines.plain import react_loop
from nexora.tools import as_model_tools

from tests.test_loop import Tools, a_call, cap


@dataclass
class _Fn:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _ToolCall:
    index: int = 0
    id: str | None = None
    function: _Fn = field(default_factory=_Fn)


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None
    reasoning: str | None = None
    model_extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Chunk:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage | None = None
    model: str | None = None


class Scripted:
    """An OpenAI-shaped client that replays turns and records the request."""

    def __init__(self, turns: list[list[_Chunk]]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    @property
    def chat(self) -> "Scripted":
        return self

    @property
    def completions(self) -> "Scripted":
        return self

    async def create(self, **kwargs: Any) -> AsyncIterator[_Chunk]:
        self.calls.append(kwargs)

        async def gen() -> AsyncIterator[_Chunk]:
            for chunk in self._turns.pop(0):
                yield chunk

        return gen()


def test_package_exports_chat_model() -> None:
    """The default install's model is ChatModel, not a LangChain provider class."""
    assert "ChatModel" in nexora.__all__
    assert nexora.ChatModel is ChatModel


async def test_chat_model_streams_text_and_usage() -> None:
    """Deltas add, and usage lands on the names the loop already reads."""
    client = Scripted(
        [
            [
                _Chunk(choices=[_Choice(_Delta(content="hello"))], model="gpt-test"),
                _Chunk(choices=[_Choice(_Delta(content=" world"))]),
                _Chunk(choices=[], usage=_Usage(3, 2, 5), model="gpt-test"),
            ]
        ]
    )
    model = ChatModel("gpt-test", client=client)
    parts = [chunk async for chunk in model.astream([HumanMessage("hi")])]

    assert "".join(chunk.text for chunk in parts) == "hello world"
    assert parts[-1].usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert parts[0].response_metadata["model_name"] == "gpt-test"


async def test_chat_model_reassembles_fragmented_tool_calls() -> None:
    """Tool-call deltas are the fragments LangChain already knows how to add."""
    client = Scripted(
        [
            [
                _Chunk(
                    choices=[
                        _Choice(
                            _Delta(
                                tool_calls=[
                                    _ToolCall(0, "c1", _Fn("read", "")),
                                ]
                            )
                        )
                    ]
                ),
                _Chunk(
                    choices=[
                        _Choice(
                            _Delta(
                                tool_calls=[
                                    _ToolCall(0, None, _Fn(None, '{"path":')),
                                ]
                            )
                        )
                    ]
                ),
                _Chunk(
                    choices=[
                        _Choice(
                            _Delta(
                                tool_calls=[
                                    _ToolCall(0, None, _Fn(None, '"a.md"}')),
                                ]
                            )
                        )
                    ]
                ),
            ]
        ]
    )
    model = ChatModel("gpt-test", client=client)
    reply = None
    async for chunk in model.astream([HumanMessage("read it")]):
        reply = chunk if reply is None else reply + chunk

    assert reply is not None
    assert reply.tool_calls == [a_call("c1", "read", {"path": "a.md"})]


async def test_chat_model_sends_bound_tools_and_prior_tool_results() -> None:
    """The next request carries the tool schema and the previous call's result."""
    client = Scripted([[_Chunk(choices=[_Choice(_Delta(content="done"))])]])
    tools = as_model_tools([{"name": "read", "description": "read a file", "schema": {}}])
    model = ChatModel("gpt-test", client=client).bind_tools(tools)
    await anext(
        model.astream(
            [
                HumanMessage("go"),
                AIMessage("", tool_calls=[a_call("c1", "read", {"path": "a.md"})]),
                ToolMessage("ok", tool_call_id="c1"),
            ]
        )
    )

    sent = client.calls[0]
    assert sent["model"] == "gpt-test"
    assert sent["stream"] is True
    assert sent["tools"][0]["function"]["name"] == "read"
    roles = [message["role"] for message in sent["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert sent["messages"][2]["tool_call_id"] == "c1"


async def test_the_loop_drives_chat_model_through_one_tool_round() -> None:
    """A ChatModel is a planner model: bind, stream, execute, stream again."""
    client = Scripted(
        [
            [
                _Chunk(
                    choices=[
                        _Choice(
                            _Delta(tool_calls=[_ToolCall(0, "c1", _Fn("read", '{"path":"a.md"}'))])
                        )
                    ]
                )
            ],
            [_Chunk(choices=[_Choice(_Delta(content="the file is empty"))])],
        ]
    )
    pending = [PendingInput("user_prompt", HumanMessage("what is in a.md?"), "p1")]

    async def drain() -> list[PendingInput]:
        items, pending[:] = pending[:], []
        return items

    collected = [
        event
        async for event in react_loop(
            ChatModel("gpt-test", client=client),
            Tools(),
            drain_inputs=drain,
            should_stop_after_turn=cap(10),
        )
    ]

    assert collected[-1]["stop_reason"] == "completed"
    assert collected[-1]["content"] == "the file is empty"
    assert len(client.calls) == 2
