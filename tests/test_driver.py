"""The driver: one attempt, fresh or continued, as a value."""

from typing import Any

import pytest
from nexora_contracts import ToolCall
from nexora_contracts.controls import ControlPlane, Permissions, gate
from nexora_engines.plain import react_loop
from nexora_orchestrator import AgentSuspended, MemorySteps

from nexora import AgentRuntime
from nexora.driver import drive
from tests.test_loop import Tools, a_call, says, scripted

DEPLOY = "deploy"


async def test_a_fresh_attempt_returns_the_outcome() -> None:
    outcome = await drive(
        react_loop,
        scripted(says("", a_call("c1", "read")), says("drafted")),
        Tools(),
    )

    assert outcome["content"] == "drafted"
    assert outcome["stop_reason"] == "completed"


async def test_the_full_cycle_through_the_driver() -> None:
    """Suspend, park, answer, continue — and the driver is told which, never asked to find out."""
    log = MemorySteps()
    runtime = AgentRuntime(store=log)
    tools = Tools(names=["read", DEPLOY])

    async def hold(call: ToolCall) -> dict[str, Any] | None:
        return {"type": "suspend", "pending_id": call["id"]} if call["name"] == DEPLOY else None

    with pytest.raises(AgentSuspended) as stopped:
        await runtime.run(
            "run-1",
            scripted(says("", a_call("c1", "read"), a_call("c2", DEPLOY))),
            tools,
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(hold))),
        )

    assert stopped.value.tool_call_id == "c2"
    assert tools.ran == ["read"]

    # ── a person approved; the orchestrator hands the continuation over ──
    resumed = Tools(names=["read", DEPLOY])
    outcome = await runtime.resume(
        "run-1",
        "c2",
        {"type": "text", "text": "approved"},
        scripted(says("deployed")),
        resumed,
    )

    assert resumed.ran == [DEPLOY]  # read is not replayed; the approved effect runs now
    assert outcome["content"] == "deployed"


async def test_events_can_be_watched_while_the_value_is_collected() -> None:
    seen: list[str] = []

    async def watch(event: dict[str, Any]) -> None:
        seen.append(event["type"])

    outcome = await drive(
        react_loop,
        scripted(says("", a_call("c1", "read")), says("done")),
        Tools(),
        on_event=watch,
    )

    assert outcome["content"] == "done"
    assert seen[:2] == ["tool_call", "tool_result"]
    assert seen[-1] == "done"
