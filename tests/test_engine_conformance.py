"""One suite, both engines.

"Swappable engine" only means something if the two produce the same event stream from the
same inputs. Every test here runs twice — once against the `while` loop, once against
`create_agent` — and the failures are the interesting part: they are ADR-001's claim about
which semantics have nowhere to live in LangChain's middleware hooks, checked rather than
argued.

`xfail(strict=True)` marks the ones we expect the LangGraph engine to fail. If one starts
passing the suite fails, which is how we would learn the gap closed.
"""

from typing import Any

import pytest

from nexora.engines.plain import react_loop
from tests.test_loop import Llm, Tools, call, done

pytest.importorskip("langchain", reason="the langgraph extra is not installed")

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from nexora.engines.langgraph import langgraph_loop


class ScriptedModel(GenericFakeChatModel):
    """A fake that binds tools and streams.

    `GenericFakeChatModel` alone cannot do both: asking it to stream a turn that carries tool
    calls fails with "No generations found in stream". Streaming is not optional for this
    comparison — the whole question of whether the engines behave alike includes whether text
    arrives token by token — so the fake implements `_stream` itself.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        reply = next(self.messages)
        assert isinstance(reply, AIMessage), "script this fake with AIMessages"
        if reply.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls)
            )
            return
        for i, word in enumerate(str(reply.content).split(" ")):
            piece = word if i == 0 else " " + word
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def a_tool(name: str = "read") -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }


class ListingTools(Tools):
    """Our fake executor, plus the `list()` the LangGraph engine needs to build its tools."""

    def __init__(self, names: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._names = names or []

    def list(self) -> list[dict[str, Any]]:
        return [a_tool(n) for n in self._names]


async def run_plain(turns: list[Any], tools: ListingTools, **kw: Any) -> list[dict[str, Any]]:
    llm = Llm(*turns)
    return [e async for e in react_loop(llm, tools, "hi", **kw)]


async def run_langgraph(
    replies: list[AIMessage], tools: ListingTools, **kw: Any
) -> list[dict[str, Any]]:
    model = ScriptedModel(messages=iter(replies))
    return [e async for e in langgraph_loop(model, tools, "hi", **kw)]


def shape_of(events: list[dict[str, Any]]) -> list[str]:
    """Event types with runs of the same type collapsed.

    How many `text` deltas a provider splits an answer into is the provider's business, not
    a semantic the engines have to agree on. Comparing raw type lists would bake one engine's
    streaming granularity into the contract.
    """
    shape: list[str] = []
    for event in events:
        if not shape or shape[-1] != event["type"]:
            shape.append(event["type"])
    return shape


def structure_of(events: list[dict[str, Any]]) -> list[str]:
    """Shape with `text` dropped — whether a turn had prose is not a control-flow fact."""
    return [t for t in shape_of(events) if t != "text"]


def text_of(events: list[dict[str, Any]]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "text")


# ── Both engines ─────────────────────────────────────────────────────────────


async def test_plain_a_reply_without_tool_calls_completes() -> None:
    events = await run_plain([[done("all done")]], ListingTools())

    assert shape_of(events) == ["done"]
    assert events[-1]["content"] == "all done"
    assert events[-1]["stop_reason"] == "completed"


async def test_langgraph_a_reply_without_tool_calls_completes() -> None:
    events = await run_langgraph([AIMessage(content="all done")], ListingTools())

    assert shape_of(events) == ["text", "done"]
    assert text_of(events) == "all done"
    assert events[-1]["content"] == "all done"
    assert events[-1]["stop_reason"] == "completed"


async def test_plain_a_tool_round_reports_call_then_result() -> None:
    events = await run_plain(
        [[*call("c1", "read"), done()], [done("finished")]],
        ListingTools(["read"]),
    )

    assert structure_of(events) == ["tool_call", "tool_result", "done"]


async def test_langgraph_a_tool_round_reports_call_then_result() -> None:
    asked = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "read", "args": {}}],
    )
    events = await run_langgraph(
        [asked, AIMessage(content="finished")],
        ListingTools(["read"]),
    )

    assert structure_of(events) == ["tool_call", "tool_result", "done"]


async def test_plain_the_system_prompt_reaches_the_provider() -> None:
    llm = Llm([done("ok")])
    await run_plain([[done("ok")]], ListingTools())  # warm the shared helper's expectations

    events = [
        e async for e in react_loop(llm, ListingTools(), "hi", system_prompt="너는 도우미다")
    ]

    assert llm.seen[0][0] == {"role": "system", "content": "너는 도우미다"}
    assert events[-1]["type"] == "done"


async def test_langgraph_the_system_prompt_reaches_the_provider() -> None:
    seen: list[list[tuple[str, Any]]] = []

    class Recording(ScriptedModel):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            seen.append([(m.type, m.content) for m in messages])
            yield from super()._stream(messages, *a, **k)

    model = Recording(messages=iter([AIMessage(content="ok")]))
    async for _ in langgraph_loop(model, ListingTools(), "hi", system_prompt="너는 도우미다"):
        pass

    assert seen[0][0] == ("system", "너는 도우미다")


# ── Where the engines are expected to diverge ────────────────────────────────


async def test_plain_a_terminating_tool_ends_the_run() -> None:
    tools = ListingTools(["submit"], defs={"submit": {"terminates_loop": True}})
    events = await run_plain([[*call("c1", "submit"), done("bye")]], tools)

    assert events[-1]["stop_reason"] == "tool"


async def test_langgraph_a_terminating_tool_ends_the_run() -> None:
    """Predicted impossible, and it is not: `before_model` re-derives that tools just ran."""
    tools = ListingTools(["submit"], defs={"submit": {"terminates_loop": True}})
    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "submit", "args": {}}])
    events = await run_langgraph([asked, AIMessage(content="kept going")], tools)

    assert events[-1]["stop_reason"] == "tool"


async def test_plain_the_policy_hook_can_end_the_run() -> None:
    async def stop_now(turn: int, content: str, calls: list[Any]) -> bool:
        return True

    events = await run_plain(
        [[*call("c1", "read"), done()], [done("never")]],
        ListingTools(["read"]),
        should_stop_after_turn=stop_now,
    )

    assert events[-1]["stop_reason"] == "policy"


async def test_langgraph_the_policy_hook_can_end_the_run() -> None:
    """Also predicted impossible. Same workaround."""
    async def stop_now(turn: int, content: str, calls: list[Any]) -> bool:
        return True

    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "read", "args": {}}])
    events = await run_langgraph(
        [asked, AIMessage(content="never")],
        ListingTools(["read"]),
        should_stop_after_turn=stop_now,
    )

    assert events[-1]["stop_reason"] == "policy"


# ── #1 provider failure ──────────────────────────────────────────────────────


class BrokenModel(ScriptedModel):
    def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
        raise RuntimeError("429 rate limited")
        yield  # pragma: no cover — makes this a generator


async def test_plain_a_provider_failure_is_reported() -> None:
    class Broken:
        def stream(self, messages: Any) -> Any:
            async def gen() -> Any:
                raise RuntimeError("429 rate limited")
                yield  # pragma: no cover

            return gen()

    events = [e async for e in react_loop(Broken(), ListingTools(), "hi")]

    assert events == [{"type": "error", "message": "429 rate limited"}]


async def test_langgraph_a_provider_failure_is_reported() -> None:
    model = BrokenModel(messages=iter([AIMessage(content="")]))
    events = [e async for e in langgraph_loop(model, ListingTools(), "hi")]

    assert events[-1]["type"] == "error"
    assert "429" in events[-1]["message"]


# ── #2 the policy gate ───────────────────────────────────────────────────────


async def deny_rm(call: Any) -> dict[str, Any] | None:
    return {"type": "error", "message": "not allowed"} if call.name == "rm" else None


async def test_plain_the_gate_denies_without_running_the_tool() -> None:
    tools = ListingTools(["rm"])
    events = await run_plain(
        [[*call("c1", "rm"), done()], [done("recovered")]], tools, before_tool_call=deny_rm
    )

    assert tools.ran == []
    denied = next(e for e in events if e["type"] == "tool_result")
    assert denied["is_error"] is True


async def test_langgraph_the_gate_denies_without_running_the_tool() -> None:
    tools = ListingTools(["rm"])
    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "rm", "args": {}}])
    events = await run_langgraph(
        [asked, AIMessage(content="recovered")], tools, before_tool_call=deny_rm
    )

    assert tools.ran == []
    denied = next(e for e in events if e["type"] == "tool_result")
    assert denied["is_error"] is True


# ── #4 event emission ────────────────────────────────────────────────────────


class Recorder:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.seen.append(event_type)


async def test_plain_a_tool_round_emits_the_hook_sequence() -> None:
    sink = Recorder()
    await run_plain(
        [[*call("c1", "read"), done()], [done("fin")]], ListingTools(["read"]), emit=sink
    )

    assert sink.seen == ["pre_tool_use", "post_tool_use", "post_tool_batch", "stop"]


async def test_langgraph_a_tool_round_emits_the_hook_sequence() -> None:
    sink = Recorder()
    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "read", "args": {}}])
    await run_langgraph([asked, AIMessage(content="fin")], ListingTools(["read"]), emit=sink)

    assert sink.seen == ["pre_tool_use", "post_tool_use", "post_tool_batch", "stop"]


# ── #12 tool result fidelity ─────────────────────────────────────────────────

_MULTIMODAL: dict[str, Any] = {
    "type": "content",
    "blocks": [
        {"type": "text", "text": "page 1"},
        {"type": "image", "data": "xxx", "mime_type": "image/png"},
    ],
}


async def test_plain_a_tool_result_reaches_the_caller_unchanged() -> None:
    tools = ListingTools(["render"], results={"render": _MULTIMODAL})
    events = await run_plain([[*call("c1", "render"), done()], [done("fin")]], tools)

    result = next(e for e in events if e["type"] == "tool_result")["result"]
    assert result == _MULTIMODAL


async def test_langgraph_a_tool_result_reaches_the_caller_unchanged() -> None:
    tools = ListingTools(["render"], results={"render": _MULTIMODAL})
    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "render", "args": {}}])
    events = await run_langgraph([asked, AIMessage(content="fin")], tools)

    result = next(e for e in events if e["type"] == "tool_result")["result"]
    assert result == _MULTIMODAL


# ── #10 abort part-way through a run ─────────────────────────────────────────


class AbortAfterFirstRound:
    """Cancels once a tool has run, the way an operator would mid-run."""

    def __init__(self, tools: ListingTools) -> None:
        self._tools = tools

    def __call__(self) -> bool:
        return bool(self._tools.ran)


async def test_plain_an_abort_part_way_through_stops_the_run() -> None:
    tools = ListingTools(["read"])
    events = await run_plain(
        [[*call("c1", "read"), done()], [done("should not be reached")]],
        tools,
        aborted=AbortAfterFirstRound(tools),
    )

    assert events[-1]["stop_reason"] == "aborted"


async def test_langgraph_an_abort_part_way_through_stops_the_run() -> None:
    tools = ListingTools(["read"])
    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "read", "args": {}}])
    events = await run_langgraph(
        [asked, AIMessage(content="should not be reached")],
        tools,
        aborted=AbortAfterFirstRound(tools),
    )

    assert events[-1]["stop_reason"] == "aborted"


# ── #6 a steer arriving as the turn finishes ─────────────────────────────────


def late_steer() -> Any:
    """Returns a steer only on the second drain — i.e. while the first turn is ending."""
    drains = 0

    def drain() -> list[Any]:
        nonlocal drains
        drains += 1
        return [{"role": "user", "content": "wait"}] if drains == 2 else []

    return drain


async def test_plain_a_late_steer_cancels_the_stop() -> None:
    events = await run_plain(
        [[done("almost")], [done("really done")]],
        ListingTools(),
        drain_steers=late_steer(),
    )

    assert events[-1]["content"] == "really done"


async def test_langgraph_a_late_steer_cancels_the_stop() -> None:
    events = await run_langgraph(
        [AIMessage(content="almost"), AIMessage(content="really done")],
        ListingTools(),
        drain_steers=late_steer(),
    )

    assert events[-1]["content"] == "really done"
