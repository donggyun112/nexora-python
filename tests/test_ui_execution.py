"""The local console shows the model-visible tool exchange, not only final text."""

import json
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from nexora.contracts.types import ToolCall, Tools
from nexora.controls import Ctx
from nexora.orchestrator import AgentSuspended, MemorySteps, Orchestrator
from nexora.runtime import AgentRuntime
from nexora_ui.execution import AgentEvent, stream_attempt
from nexora_ui.policy import permission_controls
from nexora_ui.state import FaultInjectingMemorySteps, RuntimeState, SimulatedWorkerCrash
from nexora_ui.tools import DemoTools


class TrackingTools(DemoTools):
    def __init__(self) -> None:
        super().__init__()
        self.executed: list[str] = []

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        self.executed.append(name)
        return await super().execute(name, call_id, arguments)


async def test_agent_stream_forwards_tool_call_and_result() -> None:
    async def attempt(
        _runtime: AgentRuntime, _tools: Tools, on_event: AgentEvent
    ) -> dict[str, Any]:
        await on_event(
            {"type": "tool_call", "id": "call-1", "name": "echo", "input": {"text": "hi"}}
        )
        await on_event(
            {
                "type": "tool_result",
                "id": "call-1",
                "name": "echo",
                "result": {"type": "text", "text": "hi"},
                "is_error": False,
            }
        )
        await on_event({"type": "text", "text": "done"})
        await on_event(
            {
                "type": "tool_result",
                "id": "blocked-1",
                "name": "write",
                "result": {"type": "suspend"},
                "is_error": False,
                "executed": False,
            }
        )
        await on_event({"type": "done", "stop_reason": "completed"})
        return {"type": "done"}

    frames = [
        json.loads(line)
        async for line in stream_attempt("visible-tools", attempt, RuntimeState())
    ]
    agent_events = [frame["event"] for frame in frames if frame["kind"] == "agent"]

    assert [event["type"] for event in agent_events] == ["tool_call", "tool_result", "text"]
    assert agent_events[0]["input"] == {"text": "hi"}
    assert agent_events[1]["result"] == {"type": "text", "text": "hi"}


async def test_agent_stream_forwards_thinking_beside_the_answer() -> None:
    """Reading the answer without what preceded it is reading half the turn."""

    async def attempt(
        _runtime: AgentRuntime, _tools: Tools, on_event: AgentEvent
    ) -> dict[str, Any]:
        await on_event({"type": "thinking", "content": "먼저 도구를 쓰자"})
        await on_event({"type": "text", "text": "확인했습니다"})
        await on_event({"type": "done", "stop_reason": "completed"})
        return {"type": "done"}

    frames = [
        json.loads(line) async for line in stream_attempt("thinker", attempt, RuntimeState())
    ]
    agent_events = [frame["event"] for frame in frames if frame["kind"] == "agent"]

    assert [event["type"] for event in agent_events] == ["thinking", "text"]
    assert agent_events[0]["content"] == "먼저 도구를 쓰자"


@pytest.fixture
def logged() -> Any:
    """Every line loguru emitted during a test, so a swallowed failure can be caught."""
    written: list[str] = []
    sink = logger.add(written.append, level="WARNING", format="{level}|{message}")
    yield written
    logger.remove(sink)


async def test_no_failure_branch_reaches_the_browser_without_reaching_the_log(
    logged: list[str],
) -> None:
    """Every browser-visible failure is also written to the server log."""

    async def crashed(
        _runtime: AgentRuntime, _tools: Tools, _on_event: AgentEvent
    ) -> dict[str, Any]:
        raise SimulatedWorkerCrash("logged-crash", "call-9")

    async def broke(
        _runtime: AgentRuntime, _tools: Tools, _on_event: AgentEvent
    ) -> dict[str, Any]:
        raise RuntimeError("something nobody planned for")

    async for _ in stream_attempt("logged-crash", crashed, RuntimeState()):
        pass
    async for _ in stream_attempt("logged-break", broke, RuntimeState()):
        pass

    assert any("logged-crash" in line and "committed step" in line for line in logged), logged
    assert any("logged-break" in line for line in logged), logged
    assert any("Traceback" in line for line in logged), "an unexpected failure must keep its stack"


async def test_agent_stream_marks_a_post_commit_worker_crash_as_recoverable() -> None:
    async def attempt(
        _runtime: AgentRuntime, _tools: Tools, _on_event: AgentEvent
    ) -> dict[str, Any]:
        raise SimulatedWorkerCrash("recoverable", "call-1")

    frames = [
        json.loads(line)
        async for line in stream_attempt("recoverable", attempt, RuntimeState())
    ]

    assert frames[-1] == {
        "kind": "recoverable",
        "message": "simulated worker crash after committed step 'call-1'",
        "tool_call_id": "call-1",
    }


async def test_pre_tool_permission_suspends_before_effect_then_runs_on_approval() -> None:
    tools = TrackingTools()
    store = MemorySteps()
    events: list[dict[str, Any]] = []
    call = cast(
        ToolCall,
        {
            "id": "write-1",
            "name": "remember_note",
            "args": {"key": "deploy", "value": "ready"},
            "type": "tool_call",
        },
    )
    controls = permission_controls()

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    async with Orchestrator("gate-waits", store, on_agent_event=collect) as owner:
        with pytest.raises(AgentSuspended):
            await owner.execute_round(tools, [call], lambda: False, controls=controls)

    assert tools.executed == []
    assert tools.notes == {}
    assert next(event for event in events if event["type"] == "tool_result")["executed"] is False

    async with Orchestrator("gate-waits", store) as owner:
        result = await owner.resume_effect(
            tools,
            call,
            {"type": "text", "text": "approved"},
            {"type": "suspend", "pending_id": "write-1"},
            "",
            controls=controls,
        )

    assert result == {
        "type": "text",
        "text": "remembered deploy=ready",
        "execution_count": 1,
    }
    assert tools.executed == ["remember_note"]
    assert tools.notes == {"deploy": "ready"}


async def test_demo_tool_api_failure_is_structured_and_non_retryable() -> None:
    result = await DemoTools().execute("simulate_api_failure", "failure-1", {})

    assert result == {
        "type": "error",
        "message": "simulated upstream API failure: 503 Service Unavailable",
        "code": "upstream_unavailable",
        "retryable": False,
    }


async def test_step_recovery_reuses_committed_result_without_rerunning_tool() -> None:
    store = FaultInjectingMemorySteps()
    tools = TrackingTools()
    call = cast(
        ToolCall,
        {
            "id": "recover-1",
            "name": "remember_note",
            "args": {"key": "crash-demo", "value": "committed"},
            "type": "tool_call",
        },
    )
    history = [
        HumanMessage("remember it", id="prompt-1"),
        AIMessage(content="", tool_calls=[call]),
    ]
    store.arm("step-crash")

    with pytest.raises(SimulatedWorkerCrash, match="recover-1"):
        async with Orchestrator("step-crash", store) as owner:
            await owner.execute_round(
                tools,
                [call],
                lambda: False,
                ctx=Ctx(turn=0, messages=history),
            )

    assert (await store.read("step-crash", "recover-1")).status == "done"
    assert tools.executed == ["remember_note"]
    assert tools.notes == {"crash-demo": "committed"}

    async with Orchestrator("step-crash", store) as owner:
        recovered = await owner.recover_pending(
            history,
            tools,
            retry_running=False,
        )

    assert tools.executed == ["remember_note"]
    assert recovered.resolved[0].result == {
        "type": "text",
        "text": "remembered crash-demo=committed",
        "execution_count": 1,
    }
    assert tools.execution_counts == {"recover-1": 1}
