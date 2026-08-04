"""The local console shows the model-visible tool exchange, not only final text."""

import json
from typing import Any, cast

import pytest

from nexora.contracts.types import ToolCall
from nexora.orchestrator import AgentSuspended, MemorySteps, Orchestrator
from nexora.runtime import AgentRuntime
from nexora.ui.execution import AgentEvent, stream_attempt
from nexora.ui.policy import permission_controls
from nexora.ui.state import RuntimeState
from nexora.ui.tools import DemoTools


class TrackingTools(DemoTools):
    def __init__(self) -> None:
        super().__init__()
        self.executed: list[str] = []

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        self.executed.append(name)
        return await super().execute(name, call_id, arguments)


async def test_agent_stream_forwards_tool_call_and_result() -> None:
    async def attempt(
        _runtime: AgentRuntime, _tools: DemoTools, on_event: AgentEvent
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


async def test_tool_originated_suspension_runs_the_tool_before_waiting() -> None:
    tools = TrackingTools()
    events: list[dict[str, Any]] = []
    call = cast(
        ToolCall,
        {
            "id": "ask-1",
            "name": "request_approval",
            "args": {"reason": "deploy?"},
            "type": "tool_call",
        },
    )

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    async with Orchestrator("tool-waits", MemorySteps(), on_agent_event=collect) as owner:
        with pytest.raises(AgentSuspended):
            await owner.execute_round(tools, [call], lambda: False)

    assert tools.executed == ["request_approval"]
    assert next(event for event in events if event["type"] == "tool_result")["executed"] is True


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

    assert result == {"type": "text", "text": "remembered deploy=ready"}
    assert tools.executed == ["remember_note"]
    assert tools.notes == {"deploy": "ready"}
