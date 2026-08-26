"""Continuing an agent after a suspension — the supervisor's half of `on_suspend`."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from nexora import Agent, AgentRuntime
from nexora.contracts import ToolCall
from nexora.controls import ControlPlane, Permissions, gate
from nexora.engines.plain import react_loop
from nexora.history import resume_after_suspend
from nexora.orchestrator import AgentFailed, AgentSuspended, MemorySteps

from tests.test_loop import Tools, a_call, says, scripted


async def test_a_suspended_run_continues_where_it_stopped() -> None:
    """The approval arrives, and the agent goes on without redoing anything.

    Two properties: the tool that already ran is not run again, and the suspended call is
    answered rather than re-issued. Without this there is no point approving anything — the
    supervisor could stop an agent but never resume it.
    """
    handed_over: list[tuple[Any, dict[str, Any], list[BaseMessage], list[Any]]] = []

    async def on_suspend(
        call: Any, result: dict[str, Any], snapshot: list[BaseMessage], done: list[Any]
    ) -> None:
        handed_over.append((call, result, snapshot, done))

    async def hold(call: ToolCall) -> dict[str, Any] | None:
        if call["name"] == "deploy":
            return {"type": "suspend", "pending_id": "approval-1"}
        return None

    first = Tools(names=["read", "deploy"])
    runtime = AgentRuntime(store=MemorySteps())
    with pytest.raises(AgentSuspended):
        await runtime.run(
            "resume-1",
            scripted(says("", a_call("c1", "read"), a_call("c2", "deploy"))),
            first,
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(hold))),
            on_suspend=on_suspend,
        )

    assert first.ran == ["read"]
    _call, pending, _snapshot, done = handed_over[0]
    assert [c["id"] for c in done] == ["c1"]

    # ── the human approves, hours later, in a new process ──
    resumed = Tools(
        results={"deploy": {"type": "text", "text": "deployment-id-7"}},
        names=["read", "deploy"],
    )
    model = scripted(says("deployed"))
    outcome = await runtime.resume(
        "resume-1",
        pending["pending_id"],
        {"type": "text", "text": "approved"},
        model,
        resumed,
    )

    assert resumed.ran == ["deploy"]  # c1 is not replayed; the approved effect runs now
    answer = model.seen[0][-1]
    assert isinstance(answer, ToolMessage)
    assert answer.content == "deployment-id-7"
    assert outcome["content"] == "deployed"
    assert outcome["stop_reason"] == "completed"


async def test_repeating_resume_reuses_the_committed_effect_result() -> None:
    """A failure after the effect commit must not execute the approved call twice."""
    store = MemorySteps()
    runtime = AgentRuntime(store=store)

    async def hold(_call: ToolCall) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approval-1"}

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "resume-retry",
            scripted(says("", a_call("c1", "deploy"))),
            Tools(names=["deploy"]),
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(hold))),
        )

    first_attempt = Tools(
        results={"deploy": {"type": "text", "text": "deployment-id-7"}},
        names=["deploy"],
    )
    with pytest.raises(AgentFailed):
        await runtime.resume(
            "resume-retry",
            "approval-1",
            {"type": "text", "text": "approved"},
            scripted(),
            first_attempt,
        )

    second_attempt = Tools(names=["deploy"])
    model = scripted(says("recovered"))
    outcome = await runtime.resume(
        "resume-retry",
        "approval-1",
        {"type": "text", "text": "approved"},
        model,
        second_attempt,
    )

    assert first_attempt.ran == ["deploy"]
    assert second_attempt.ran == []
    answer = model.seen[0][-1]
    assert isinstance(answer, ToolMessage)
    assert answer.content == "deployment-id-7"
    assert outcome["content"] == "recovered"


async def test_resume_accepts_the_agent_that_started_the_run() -> None:
    """resume() expands an Agent like run() does, and the snapshot keeps one system message.

    Without the expansion a host that ran an Agent must unpack model/tools by hand; without the
    adoption, re-supplying the agent's system_prompt stacked a second SystemMessage above the
    one restored from the suspension snapshot.
    """

    async def hold(call: ToolCall) -> dict[str, Any] | None:
        if call["name"] == "deploy":
            return {"type": "suspend", "pending_id": "approval-1"}
        return None

    runtime = AgentRuntime(store=MemorySteps())
    first = Agent(
        name="deployer",
        description="Deploys",
        model=scripted(says("", a_call("c1", "deploy"))),
        tools=Tools(names=["deploy"]),
        system_prompt="Deploy carefully.",
    )
    with pytest.raises(AgentSuspended):
        await runtime.run(
            "resume-agent",
            first,
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(hold))),
        )

    resumed_tools = Tools(
        results={"deploy": {"type": "text", "text": "deployment-id-7"}}, names=["deploy"]
    )
    model = scripted(says("deployed"))
    resumed = Agent(
        name="deployer",
        description="Deploys",
        model=model,
        tools=resumed_tools,
        system_prompt="Deploy carefully.",
    )
    outcome = await runtime.resume(
        "resume-agent", "approval-1", {"type": "text", "text": "approved"}, resumed
    )

    assert resumed_tools.ran == ["deploy"]
    systems = [m for m in model.seen[0] if isinstance(m, SystemMessage)]
    assert [m.content for m in systems] == ["Deploy carefully."]
    assert outcome["content"] == "deployed"


async def test_the_resumed_history_answers_the_call_that_stopped() -> None:
    """Resumed history places the suspended call's answer directly after its request."""
    snapshot: list[BaseMessage] = [
        AIMessage(content="", tool_calls=[a_call("c1", "deploy")]),
    ]

    history = resume_after_suspend(snapshot, "c1", {"type": "text", "text": "ok"}, name="deploy")

    answer = history[-1]
    assert isinstance(answer, ToolMessage)
    assert (answer.tool_call_id, answer.content, answer.status) == ("c1", "ok", "success")


async def test_a_refused_answer_resumes_as_a_failed_call() -> None:
    """Denial is an answer too — the model sees a failed call and can react."""
    snapshot: list[BaseMessage] = [
        AIMessage(content="", tool_calls=[a_call("c1", "deploy")]),
    ]

    history = resume_after_suspend(snapshot, "c1", {"type": "error", "message": "no"})

    answer = history[-1]
    assert isinstance(answer, ToolMessage)
    assert answer.status == "error"
    assert answer.content == "[ERROR] no"


async def test_an_empty_prompt_adds_no_user_turn() -> None:
    llm = scripted(says("continuing"))
    history: list[BaseMessage] = [AIMessage(content="earlier")]

    async for _ in react_loop(llm, Tools(), history=history):
        pass

    assert [type(m).__name__ for m in llm.seen[0]] == ["AIMessage"]
