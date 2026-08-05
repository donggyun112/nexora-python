"""Continuing an agent after a suspension — the supervisor's half of `on_suspend`."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from nexora import AgentRuntime
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
    call, _pending, _snapshot, done = handed_over[0]
    assert [c["id"] for c in done] == ["c1"]

    # ── the human approves, hours later, in a new process ──
    resumed = Tools(
        results={"deploy": {"type": "text", "text": "deployment-id-7"}},
        names=["read", "deploy"],
    )
    model = scripted(says("deployed"))
    outcome = await runtime.resume(
        "resume-1",
        call["id"],
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


async def test_a_suspension_keeps_the_work_the_rest_of_the_batch_already_did() -> None:
    """react.ts:196-245 — the round is absorbed whole, then it stops.

    A call that finished is an external fact. Dropping it because a *different* call suspended
    means the resumed run re-issues it, and a `write` runs twice. This engine used to return at
    the suspended call, so anything after it in call order vanished.
    """

    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "p1"}},
        defs={
            "read": {"is_concurrency_safe": True},
            "ask": {"is_concurrency_safe": True},
            "write": {"is_concurrency_safe": True},
        },
        names=["read", "ask", "write"],
    )

    handed: list[tuple[str, list[str]]] = []

    async def on_suspend(
        call: Any, result: dict[str, Any], snapshot: list[BaseMessage], done: list[Any]
    ) -> None:
        handed.append((call["id"], [d["id"] for d in done]))

    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    with pytest.raises(AgentSuspended):
        await AgentRuntime().run(
            "resume-batch",
            scripted(
                says("", a_call("c1", "read"), a_call("c2", "ask"), a_call("c3", "write")),
                says("after"),
            ),
            tools,
            "go",
            on_suspend=on_suspend,
            on_event=collect,
        )

    assert [e["id"] for e in events if e["type"] == "tool_result"] == ["c1", "c2", "c3"]
    assert events[-1]["type"] == "suspended"
    assert handed == [("c2", ["c1", "c3"])]  # c3 finished after the suspension and is kept


async def test_an_elicitation_answer_is_injected_without_reexecuting_the_tool() -> None:
    """A tool that asks the human has already run; its answer is the eventual tool result."""
    runtime = AgentRuntime(store=MemorySteps())
    first = Tools(
        results={
            "request_approval": {
                "type": "suspend",
                "pending_id": "question-1",
                "prompt": "deploy?",
            }
        },
        names=["request_approval"],
    )

    with pytest.raises(AgentSuspended) as stopped:
        await runtime.run(
            "resume-elicitation",
            scripted(says("", a_call("c1", "request_approval"))),
            first,
            "ship it",
        )

    resumed = Tools(names=["request_approval"])
    model = scripted(says("approved"))
    outcome = await runtime.resume(
        "resume-elicitation",
        stopped.value.tool_call_id,
        {"type": "text", "text": "yes"},
        model,
        resumed,
    )

    assert first.ran == ["request_approval"]
    assert resumed.ran == []
    answer = model.seen[0][-1]
    assert isinstance(answer, ToolMessage)
    assert answer.content == "yes"
    assert outcome["content"] == "approved"


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
            "c1",
            {"type": "text", "text": "approved"},
            scripted(),
            first_attempt,
        )

    second_attempt = Tools(names=["deploy"])
    model = scripted(says("recovered"))
    outcome = await runtime.resume(
        "resume-retry",
        "c1",
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


async def test_the_resumed_history_answers_the_call_that_stopped() -> None:
    """A provider rejects an assistant turn whose tool calls have no results, so the shape has
    to be exactly: the asking turn, then its answer."""
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
