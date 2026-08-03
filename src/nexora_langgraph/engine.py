"""The same agent loop, driven by LangChain's `create_agent` instead of a `while`.

This exists to keep ADR-001 falsifiable. The claim there is that the reference's semantics do
not all fit LangChain's six middleware hooks; the way to find out is to write it and run both
engines against one conformance suite (`tests/test_engine_conformance.py`).

Where a semantic has no hook to live in, that is recorded here as a `ponytail:` comment rather
than worked around silently — the gaps are the point.
"""

from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool

from nexora.tools import render_for_model, select_for_execution
from nexora.types import (
    Aborted,
    BeforeToolCall,
    DrainSteers,
    Emit,
    LLMMessage,
    OnSuspend,
    ShouldStopAfterTurn,
    ToolCall,
    Tools,
)


async def langgraph_loop(
    model: Any,
    tools: Tools,
    prompt: str,
    *,
    history: list[LLMMessage] | None = None,
    aborted: Aborted = lambda: False,
    before_tool_call: BeforeToolCall | None = None,
    emit: Emit | None = None,
    drain_steers: DrainSteers | None = None,
    should_stop_after_turn: ShouldStopAfterTurn | None = None,
    on_suspend: OnSuspend | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Same inputs and same event stream as `nexora.loop.react_loop`.

    `model` is a LangChain chat model here rather than a `nexora.types.LLM`: `create_agent`
    owns the provider call, so the engines cannot share that one argument.

    ponytail: `before_tool_call`, `emit`, `should_stop_after_turn` and `on_suspend` are
    accepted and ignored. The gate and the suspend handoff have homes here (`wrap_tool_call`
    and `after_model`'s `interrupt`) and are simply not written yet; the stop policy does not
    — see the note on `_Steering`. The conformance suite records which is which.
    """
    agent = create_agent(model, _as_langchain_tools(tools), middleware=[_Steering(drain_steers)])

    state: dict[str, Any] = {"messages": _to_langchain(history or [], prompt)}
    seen: set[str] = set()
    calls_made: list[dict[str, Any]] = []
    spent: Counter[str] = Counter()
    text = ""

    if aborted():
        yield _done(text, calls_made, "aborted", spent)
        return

    async for chunk in agent.astream(state, stream_mode="values"):  # type: ignore[call-overload]
        for message in chunk["messages"]:
            key = str(getattr(message, "id", None) or id(message))
            if key in seen:
                continue
            seen.add(key)
            async for event in _translate(message, tools, calls_made, spent, before_tool_call):
                if event["type"] == "text":
                    text = event["text"]
                yield event

    yield _done(text, calls_made, "completed", spent)


class _Steering(AgentMiddleware):
    """`drain_steers` at the top of a round — react.ts L116."""

    def __init__(self, drain_steers: DrainSteers | None) -> None:
        super().__init__()
        self._drain = drain_steers

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._drain is None:
            return None
        steers = self._drain()
        if not steers:
            return None
        return {"messages": _to_langchain(steers, None)}

    # ponytail: react.ts L152 — a steer arriving as the turn finishes must cancel the stop.
    # `after_model` can `jump_to: "model"`, but only after the model already produced its
    # final answer, so the cancelled stop is observable in the stream. Not implemented.


def _as_langchain_tools(tools: Tools) -> list[StructuredTool]:
    """Wrap our executor's tools so `create_agent`'s ToolNode can call them."""
    wrapped: list[StructuredTool] = []
    for summary in tools.list():
        name = summary["name"]

        async def run(_name: str = name, **kwargs: Any) -> str:
            return render_for_model(await tools.execute(_name, _name, kwargs))

        wrapped.append(
            StructuredTool.from_function(
                coroutine=run,
                name=name,
                description=summary["description"],
                args_schema=summary["parameters"],
            )
        )
    return wrapped


def _to_langchain(messages: list[LLMMessage], prompt: str | None) -> list[BaseMessage]:
    from langchain_core.messages import HumanMessage

    out: list[BaseMessage] = [
        HumanMessage(content=m["content"] if isinstance(m["content"], str) else "")
        for m in messages
    ]
    if prompt is not None:
        out.append(HumanMessage(content=prompt))
    return out


async def _translate(
    message: BaseMessage,
    tools: Tools,
    calls_made: list[dict[str, Any]],
    spent: Counter[str],
    before_tool_call: BeforeToolCall | None,
) -> AsyncIterator[dict[str, Any]]:
    """One LangChain message into zero or more Nexora events."""
    if isinstance(message, AIMessage):
        if message.usage_metadata:
            spent.update(
                {
                    "prompt_tokens": message.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": message.usage_metadata.get("output_tokens", 0),
                }
            )
        if isinstance(message.content, str) and message.content:
            yield {"type": "text", "text": message.content}
        requested = [ToolCall(c["id"] or "", c["name"], c["args"]) for c in message.tool_calls]
        for call in select_for_execution(tools, requested):
            calls_made.append({"name": call.name, "input": call.arguments})
            yield {
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
    elif isinstance(message, ToolMessage):
        yield {
            "type": "tool_result",
            "id": message.tool_call_id,
            "name": message.name or "",
            "result": {"type": "text", "text": str(message.content)},
            "is_error": message.status == "error",
        }


def _done(
    content: str,
    calls_made: list[dict[str, Any]],
    reason: str,
    spent: Counter[str],
) -> dict[str, Any]:
    done: dict[str, Any] = {
        "type": "done",
        "content": content,
        "tool_calls": calls_made,
        "stop_reason": reason,
    }
    if spent:
        done["usage"] = dict(spent)
    return done
