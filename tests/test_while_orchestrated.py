"""While-loop recovery when the orchestrator owns tool execution."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nexora.controls import Ctx
from nexora.engines.plain import react_loop
from nexora.orchestrator import Indeterminate, MemorySteps, Orchestrator

from .test_loop import Tools, a_call, says, scripted


async def collect(events: Any) -> list[dict[str, Any]]:
    return [event async for event in events]


async def test_live_while_round_is_owned_by_orchestrator() -> None:
    log = MemorySteps()
    orchestrator = Orchestrator("run-1", log)
    tools = Tools()
    model = scripted(says("", a_call("c1", "read")), says("finished"))

    events = await collect(react_loop(model, tools, execute_round=orchestrator.execute_round))

    assert tools.ran == ["read"]
    assert (await log.read("run-1", "c1")).status == "done"
    assert events[-1]["content"] == "finished"


async def test_pending_round_recovers_without_replaying_model() -> None:
    log = MemorySteps()
    tools = Tools(
        results={
            "first": {"type": "text", "text": "one"},
            "second": {"type": "text", "text": "two"},
        },
        names=["first", "second"],
    )
    calls = [a_call("c1", "first"), a_call("c2", "second")]
    history = [HumanMessage("go"), AIMessage(content="", tool_calls=calls)]
    stopping = {"now": False}

    original_execute = tools.execute

    async def stop_after_first(name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        result = await original_execute(name, call_id, arguments)
        if call_id == "c1":
            stopping["now"] = True
        return result

    tools.execute = stop_after_first  # type: ignore[assignment]

    # The server committed c1's tool step, then died before appending either ToolMessage.
    crashed = Orchestrator("run-2", log)
    await crashed.execute_round(
        tools,
        calls,
        lambda: stopping["now"],
        ctx=Ctx(turn=0, messages=history),
    )
    assert [call["id"] for call in await crashed.pending_calls()] == ["c1", "c2"]
    tools.ran.clear()
    stopping["now"] = False

    resumed = Orchestrator("run-2", log)
    recovered = await resumed.recover_pending(history, tools)

    assert tools.ran == ["second"]
    answers = [message for message in recovered.history if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in answers] == ["c1", "c2"]
    assert [message.content for message in answers] == ["one", "two"]
    assert [call["id"] for call in await resumed.pending_calls()] == ["c1", "c2"]

    # Only the post-tool model call runs. The original tool-calling model turn is not replayed.
    model = scripted(says("finished"))
    events = await collect(
        react_loop(
            model,
            tools,
            history=recovered.history,
            execute_round=resumed.execute_round,
        )
    )

    assert len(model.seen) == 1
    assert events[-1]["content"] == "finished"


async def test_running_step_stops_recovery_before_any_new_tool() -> None:
    log = MemorySteps()
    tools = Tools(names=["first", "second"])
    calls = [a_call("c1", "first"), a_call("c2", "second")]
    history = [HumanMessage("go"), AIMessage(content="", tool_calls=calls)]
    await log.start("run-3", "c2")

    with pytest.raises(Indeterminate, match="c2"):
        await Orchestrator("run-3", log).recover_pending(history, tools, retry_running=False)

    assert tools.ran == []


async def test_running_step_retries_with_the_same_idempotency_key_by_default() -> None:
    log = MemorySteps()
    tools = Tools()
    call = a_call("c1", "read")
    history = [HumanMessage("go"), AIMessage(content="", tool_calls=[call])]
    await log.finish("run-4", "agent:pending-round", {"calls": [call], "turn": 0})
    await log.start("run-4", "c1")

    recovered = await Orchestrator("run-4", log).recover_pending(history, tools)

    assert tools.ran == ["read"]
    assert recovered.resolved[0].call["id"] == "c1"
    assert (await log.read("run-4", "c1")).status == "done"
