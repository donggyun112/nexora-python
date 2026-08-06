"""The public Agent Runtime facade over the single plain planner."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nexora import AgentRuntime, ToolCall, run
from nexora.contracts import EventType, PendingInput
from nexora.controls import ControlPlane, Permissions, gate
from nexora.orchestrator import AgentFailed, AgentSuspended, MemorySteps, Orchestrator
from nexora.tools import InvalidToolResult

from .test_loop import Tools, a_call, says, scripted


async def test_top_level_run_is_the_default_agent_entry_point() -> None:
    outcome = await run(scripted(says("done")), Tools(), "hello")

    assert outcome["content"] == "done"


async def test_runtime_hides_orchestrator_wiring_but_records_tool_effects() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store=store)

    outcome = await runtime.run(
        "run-1",
        scripted(says("", a_call("c1", "read")), says("done")),
        Tools(),
        "hello",
    )

    assert outcome["content"] == "done"
    assert (await store.read("run-1", "c1")).status == "done"


async def test_invalid_tool_suspension_is_not_committed_as_a_completed_effect() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store=store)
    tools = Tools(results={"ask": {"type": "suspend", "pending_id": "p1"}}, names=["ask"])

    with pytest.raises(InvalidToolResult, match="pre_tool_use"):
        await runtime.run(
            "invalid-tool-suspend",
            scripted(says("", a_call("c1", "ask"))),
            tools,
            "ask",
        )

    assert tools.ran == ["ask"]
    assert (await store.read("invalid-tool-suspend", "c1")).status == "absent"


async def test_submitted_background_result_uses_the_same_input_queue() -> None:
    store = MemorySteps()
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        emitted.append((event_type, payload))

    runtime = AgentRuntime(store=store, emit=emit)
    call = a_call("bg-call", "background_read")
    history = [AIMessage(content="", tool_calls=[call])]
    result = ToolMessage(content="background complete", tool_call_id="bg-call")
    await runtime.submit(
        "run-background",
        PendingInput("background_tool_result", result, "background-task-1"),
    )

    model = scripted(says("continued"))
    outcome = await runtime.run(
        "run-background",
        model,
        Tools(names=["background_read"]),
        history=history,
    )

    assert model.seen[0][-1].content == "background complete"
    assert outcome["content"] == "continued"
    assert [event for event, _ in emitted] == ["context_injected", "stop"]
    assert (await store.list_inputs("run-background"))[0].status == "admitted"


async def test_an_admitted_input_is_replayed_when_the_attempt_has_no_transcript() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store=store)
    await runtime.submit(
        "run-input-retry",
        PendingInput("user_prompt", HumanMessage("do it"), "prompt-stable"),
    )

    with pytest.raises(AgentFailed):
        await runtime.run("run-input-retry", scripted(), Tools(), "")

    model = scripted(says("done"))
    outcome = await runtime.run("run-input-retry", model, Tools(), "")

    assert model.seen[0][-1].content == "do it"
    assert outcome["content"] == "done"


async def test_retrying_the_same_submission_id_does_not_duplicate_input_or_event() -> None:
    store = MemorySteps()
    emitted: list[str] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        emitted.append(event_type)

    runtime = AgentRuntime(store=store, emit=emit)
    item = PendingInput("user_prompt", HumanMessage("hello"), "prompt-stable")
    await runtime.submit("run-submit-retry", item)
    await runtime.submit("run-submit-retry", item)

    assert len(await store.list_inputs("run-submit-retry")) == 1
    assert emitted == ["user_prompt_submit"]


async def test_runtime_parks_and_resumes_without_exposing_the_ledger() -> None:
    runtime = AgentRuntime(store=MemorySteps())

    async def ask(_call: Any) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approval-1"}

    with pytest.raises(AgentSuspended) as stopped:
        await runtime.run(
            "run-2",
            scripted(says("", a_call("c1", "deploy"))),
            Tools(names=["deploy"]),
            "ship",
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask))),
        )

    resumed = Tools(names=["deploy"])
    outcome = await runtime.resume(
        "run-2",
        stopped.value.pending_id,
        {"type": "text", "text": "approved"},
        scripted(says("deployed")),
        resumed,
    )

    assert resumed.ran == ["deploy"]
    assert outcome["content"] == "deployed"


async def test_new_interactive_input_cancels_pending_effect_before_switching() -> None:
    store = MemorySteps()
    emitted: list[str] = []

    async def emit(event_type: str, _payload: dict[str, Any]) -> None:
        emitted.append(event_type)

    async def ask(_call: Any) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approval-1"}

    runtime = AgentRuntime(store=store, emit=emit)
    with pytest.raises(AgentSuspended):
        await runtime.run(
            "cancel-switch",
            scripted(says("", a_call("c1", "deploy"))),
            Tools(names=["deploy"]),
            "deploy it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask))),
        )

    tools = Tools(names=["deploy"])
    model = scripted(says("switched"))
    outcome = await runtime.run(
        "cancel-switch",
        model,
        tools,
        "do something else",
        prompt_id="replacement-1",
    )

    seen = model.seen[0]
    cancelled = next(message for message in seen if isinstance(message, ToolMessage))
    replacement = next(
        message
        for message in seen
        if isinstance(message, HumanMessage) and message.content == "do something else"
    )
    assert seen.index(cancelled) < seen.index(replacement)
    assert "cancelled" in str(cancelled.content)
    assert tools.ran == []
    assert (await store.read("cancel-switch", "c1")).value["code"] == "cancelled"
    assert EventType.TOOL_REQUEST_CANCELLED in emitted
    assert outcome["content"] == "switched"

    with pytest.raises(LookupError, match="no active suspension"):
        await runtime.resume(
            "cancel-switch",
            "approval-1",
            {"type": "text", "text": "approved too late"},
            scripted(says("never")),
            tools,
        )


async def test_headless_input_waits_then_follows_the_eventual_tool_result() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store=store)

    async def ask(_call: Any) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approval-1"}

    controls = ControlPlane(pre_tool_use=Permissions(gate(ask)))
    with pytest.raises(AgentSuspended):
        await runtime.run(
            "headless-wait",
            scripted(says("", a_call("c1", "deploy"))),
            Tools(names=["deploy"]),
            "deploy",
            controls=controls,
        )

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "headless-wait",
            scripted(says("unused")),
            Tools(names=["deploy"]),
            "also summarize",
            input_mode="headless",
        )

    model = scripted(says("deployed and summarized"))
    tools = Tools(results={"deploy": {"type": "text", "text": "ok"}}, names=["deploy"])
    await runtime.resume(
        "headless-wait",
        "approval-1",
        {"type": "text", "text": "approved"},
        model,
        tools,
        controls=controls,
    )

    tool_index = next(
        i for i, message in enumerate(model.seen[0]) if isinstance(message, ToolMessage)
    )
    user_index = next(
        i
        for i, message in enumerate(model.seen[0])
        if isinstance(message, HumanMessage) and message.content == "also summarize"
    )
    assert tool_index < user_index
    assert tools.ran == ["deploy"]


async def test_suspension_commits_then_releases_the_worker_instead_of_waiting() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store=store, owner="worker-1")
    announced_after_commit = False

    async def ask(_call: Any) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approval-1"}

    async def on_event(event: dict[str, Any]) -> None:
        nonlocal announced_after_commit
        if event["type"] == "suspended":
            announced_after_commit = (await store.read("run-dead", "suspend:c1")).status == "done"

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-dead",
            scripted(says("", a_call("c1", "deploy"))),
            Tools(names=["deploy"]),
            "ship",
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask))),
            on_event=on_event,
        )

    assert announced_after_commit
    assert await store.acquire("run-dead", "worker-2", 60.0) != 0


async def test_runtime_recovers_a_tool_round_without_replaying_the_model() -> None:
    store = MemorySteps()
    calls = [a_call("c1", "first"), a_call("c2", "second")]
    history = [HumanMessage("go"), AIMessage(content="", tool_calls=calls)]
    tools = Tools(
        results={
            "first": {"type": "text", "text": "one"},
            "second": {"type": "text", "text": "two"},
        },
        names=["first", "second"],
    )

    # The transcript already contains the model response; c1 committed before the process died.
    crashed = Orchestrator("run-3", store)
    await crashed.execute_round(tools, calls, lambda: tools.ran == ["first"])
    tools.ran.clear()

    model = scripted(says("finished"))
    outcome = await AgentRuntime(store=store).recover("run-3", history, model, tools)

    assert tools.ran == ["second"]
    assert len(model.seen) == 1
    assert outcome["content"] == "finished"


async def test_recovery_does_not_re_decide_a_refusal_the_transcript_already_answered() -> None:
    """The other half of the rule the test below states, and the reason a `Deny` needs no ledger
    entry of its own: a refusal is an answer, and an answered call is not part of the pending
    round. The gate re-runs only for calls with nothing in the transcript — record the refusal in
    the ledger instead and `recover_pending` would replay it as a completed step, journalling an
    effect that never happened.
    """
    store = MemorySteps()
    requested = [a_call("c1", "rm"), a_call("c2", "read")]
    tools = Tools(names=["rm", "read"])
    asked: list[str] = []

    async def deny_rm(call: ToolCall) -> dict[str, Any] | None:
        asked.append(call["id"] or "")
        return {"type": "error", "message": "not allowed"} if call["name"] == "rm" else None

    history = [
        HumanMessage("go"),
        AIMessage(content="", tool_calls=requested),
        ToolMessage(content="not allowed", tool_call_id="c1", status="error"),
    ]

    outcome = await AgentRuntime(store=store).recover(
        "run-denied",
        history,
        scripted(says("finished")),
        tools,
        controls=ControlPlane(pre_tool_use=Permissions(gate(deny_rm))),
    )

    assert asked == ["c2"]  # c1 was answered; nothing re-decides it
    assert tools.ran == ["read"]
    assert outcome["content"] == "finished"


async def test_public_recover_resumes_the_durable_round_without_model_replay() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store=store)
    requested = [a_call("c1", "first"), a_call("c2", "second")]
    tools = Tools(names=["first", "second"])
    seen: list[str] = []
    fail = True

    async def policy(call: ToolCall) -> None:
        nonlocal fail
        seen.append(call["id"] or "")
        if call["id"] == "c2" and fail:
            fail = False
            raise RuntimeError("policy unavailable")

    controls = ControlPlane(pre_tool_use=Permissions(gate(policy)))
    first_model = scripted(says("", *requested))
    with pytest.raises(RuntimeError, match="policy unavailable"):
        await runtime.run("run-5", first_model, tools, "hello", controls=controls)

    history = [HumanMessage("hello"), AIMessage(content="", tool_calls=requested)]
    final_model = scripted(says("finished"))
    outcome = await runtime.recover("run-5", history, final_model, tools, controls=controls)

    assert len(first_model.seen) == 1
    assert len(final_model.seen) == 1
    # Direct recovery deliberately re-evaluates gates for calls that never reached an effect.
    assert seen == ["c1", "c2", "c1", "c2"]
    assert tools.ran == ["first", "second"]
    assert outcome["content"] == "finished"
