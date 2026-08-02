"""One suite, both engines, identical inputs.

"Swappable engine" only means something if the two produce the same events from the same
inputs. Both now take a LangChain chat model and our `Tools`, so every test here is
parametrized over the pair rather than written twice — a divergence shows up as a failure
instead of as two tests quietly asserting different things.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from nexora.engines.plain import react_loop
from tests.test_loop import Llm, Tools, a_call, says, scripted

pytest.importorskip("langchain", reason="the langgraph extra is not installed")

from nexora.engines.langgraph import langgraph_loop

ENGINES = [pytest.param(react_loop, id="plain"), pytest.param(langgraph_loop, id="langgraph")]


async def run(engine: Any, replies: list[AIMessage], tools: Tools, **kw: Any) -> list[Any]:
    return [e async for e in engine(scripted(*replies), tools, "hi", **kw)]


def shape_of(events: list[dict[str, Any]]) -> list[str]:
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


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_reply_without_tool_calls_completes(engine: Any) -> None:
    events = await run(engine, [says("all done")], Tools())

    assert structure_of(events) == ["done"]
    assert text_of(events) == "all done"
    assert events[-1]["content"] == "all done"
    assert events[-1]["stop_reason"] == "completed"


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_tool_round_reports_call_then_result(engine: Any) -> None:
    tools = Tools()
    events = await run(engine, [says("", a_call("c1", "read")), says("finished")], tools)

    assert structure_of(events) == ["tool_call", "tool_result", "done"]
    assert tools.ran == ["read"]


@pytest.mark.parametrize("engine", ENGINES)
async def test_the_system_prompt_reaches_the_provider(engine: Any) -> None:
    llm = scripted(says("ok"))

    async for _ in engine(llm, Tools(), "hi", system_prompt="너는 도우미다"):
        pass

    assert llm.seen[0][0].content == "너는 도우미다"


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_terminating_tool_ends_the_run(engine: Any) -> None:
    tools = Tools(defs={"submit": {"terminates_loop": True}}, names=["submit"])
    events = await run(engine, [says("bye", a_call("c1", "submit")), says("kept going")], tools)

    assert events[-1]["stop_reason"] == "tool"


@pytest.mark.parametrize("engine", ENGINES)
async def test_the_policy_hook_can_end_the_run(engine: Any) -> None:
    async def stop_now(turn: int, content: str, calls: list[Any]) -> bool:
        return True

    events = await run(
        engine,
        [says("", a_call("c1", "read")), says("never")],
        Tools(),
        should_stop_after_turn=stop_now,
    )

    assert events[-1]["stop_reason"] == "policy"


@pytest.mark.parametrize("engine", ENGINES)
async def test_the_gate_denies_without_running_the_tool(engine: Any) -> None:
    tools = Tools(names=["rm"])

    async def deny(call: dict[str, Any]) -> dict[str, Any] | None:
        return {"type": "error", "message": "not allowed"} if call["name"] == "rm" else None

    events = await run(
        engine, [says("", a_call("c1", "rm")), says("recovered")], tools, before_tool_call=deny
    )

    assert tools.ran == []
    assert next(e for e in events if e["type"] == "tool_result")["is_error"] is True


@pytest.mark.parametrize("engine", ENGINES)
async def test_an_abort_part_way_through_stops_the_run(engine: Any) -> None:
    tools = Tools()

    events = await run(
        engine,
        [says("", a_call("c1", "read")), says("should not be reached")],
        tools,
        aborted=lambda: bool(tools.ran),
    )

    assert events[-1]["stop_reason"] == "aborted"


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_late_steer_cancels_the_stop(engine: Any) -> None:
    drains = 0

    def drain() -> list[BaseMessage]:
        nonlocal drains
        drains += 1
        return [HumanMessage("wait")] if drains == 2 else []

    events = await run(
        engine, [says("almost"), says("really done")], Tools(), drain_steers=drain
    )

    assert events[-1]["content"] == "really done"


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_tool_result_reaches_the_caller_unchanged(engine: Any) -> None:
    """A multimodal result must survive; flattening it to a string loses the images."""
    multimodal: dict[str, Any] = {
        "type": "content",
        "blocks": [
            {"type": "text", "text": "page 1"},
            {"type": "image", "data": "xxx", "mime_type": "image/png"},
        ],
    }
    tools = Tools(results={"render": multimodal}, names=["render"])

    events = await run(engine, [says("", a_call("c1", "render")), says("fin")], tools)

    assert next(e for e in events if e["type"] == "tool_result")["result"] == multimodal


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_tool_round_emits_the_hook_sequence(engine: Any) -> None:
    seen: list[str] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        seen.append(event_type)

    await run(engine, [says("", a_call("c1", "read")), says("fin")], Tools(), emit=emit)

    assert seen == ["pre_tool_use", "post_tool_use", "post_tool_batch", "stop"]


@pytest.mark.parametrize("engine", ENGINES)
async def test_a_provider_failure_is_reported(engine: Any) -> None:
    class Broken(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("429 rate limited")
            yield  # pragma: no cover — makes this a generator

    events = [e async for e in engine(Broken(messages=iter([])), Tools(), "hi")]

    assert events[-1]["type"] == "error"
    assert "429" in events[-1]["message"]


@pytest.mark.parametrize("engine", ENGINES)
async def test_gate_events_carry_the_same_payload(engine: Any) -> None:
    """Types alone are not enough. One engine used to omit `turn` from its payloads and this
    suite could not see it, because the gate was implemented twice — once per engine."""
    seen: list[tuple[str, dict[str, Any]]] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        seen.append((event_type, payload))

    async def deny(call: dict[str, Any]) -> dict[str, Any] | None:
        return {"type": "error", "message": "not allowed"}

    await run(
        engine,
        [says("", a_call("c1", "rm")), says("fin")],
        Tools(names=["rm"]),
        before_tool_call=deny,
        emit=emit,
    )

    gate_events = {t: p for t, p in seen if t.startswith(("pre_tool", "permission"))}
    assert gate_events["pre_tool_use"] == {
        "turn": 0,
        "call_id": "c1",
        "name": "rm",
        "input": {},
    }
    assert gate_events["permission_denied"]["reason"] == {
        "type": "error",
        "message": "not allowed",
    }


@pytest.mark.parametrize("engine", ENGINES)
async def test_the_tool_call_id_reaches_the_executor(engine: Any) -> None:
    """The call id is the idempotency key (ADR-002), so it has to survive to `execute`.

    The LangGraph engine passed the tool *name* instead. Every call of a tool then shared one
    key: a genuine retry looked like a new call, and two different calls looked like the same
    one. Only a heartbeat-driven re-execution would have surfaced it in production.
    """
    seen: list[str] = []

    class Recording(Tools):
        async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
            seen.append(call_id)
            return {"type": "text", "text": "ok"}

    await run(
        engine,
        [says("", a_call("c1", "read"), a_call("c2", "read")), says("fin")],
        Recording(names=["read"]),
    )

    assert seen == ["c1", "c2"]
