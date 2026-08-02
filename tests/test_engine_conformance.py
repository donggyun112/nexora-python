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

from nexora.loop import react_loop
from tests.test_loop import Llm, Tools, call, done

pytest.importorskip("langchain", reason="the langgraph extra is not installed")

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from nexora_langgraph.engine import langgraph_loop


class ScriptedModel(GenericFakeChatModel):
    """A fake that accepts `bind_tools`, which `create_agent` always calls."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


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


def types_of(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


# ── Both engines ─────────────────────────────────────────────────────────────


async def test_plain_a_reply_without_tool_calls_completes() -> None:
    events = await run_plain([[done("all done")]], ListingTools())

    assert types_of(events) == ["done"]
    assert events[-1]["content"] == "all done"
    assert events[-1]["stop_reason"] == "completed"


async def test_langgraph_a_reply_without_tool_calls_completes() -> None:
    events = await run_langgraph([AIMessage(content="all done")], ListingTools())

    assert types_of(events) == ["text", "done"]
    assert events[-1]["content"] == "all done"
    assert events[-1]["stop_reason"] == "completed"


async def test_plain_a_tool_round_reports_call_then_result() -> None:
    events = await run_plain(
        [[*call("c1", "read"), done()], [done("finished")]],
        ListingTools(["read"]),
    )

    assert types_of(events) == ["tool_call", "tool_result", "done"]


async def test_langgraph_a_tool_round_reports_call_then_result() -> None:
    asked = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "read", "args": {}}],
    )
    events = await run_langgraph(
        [asked, AIMessage(content="finished")],
        ListingTools(["read"]),
    )

    assert "tool_call" in types_of(events)
    assert "tool_result" in types_of(events)
    assert events[-1]["type"] == "done"


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
        def _generate(self, messages: Any, *a: Any, **k: Any) -> Any:
            seen.append([(m.type, m.content) for m in messages])
            return super()._generate(messages, *a, **k)

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
    def _generate(self, messages: Any, *a: Any, **k: Any) -> Any:
        raise RuntimeError("429 rate limited")


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
