"""What a durable step has to guarantee, and that `react_loop` composes as one."""

from typing import Any

import pytest
from nexora.engines.plain import react_loop
from nexora.orchestrator import (
    AgentFailed,
    AgentSuspended,
    MemorySteps,
    Orchestrator,
    Suspended,
    run_agent,
)

from tests.test_loop import Llm, Tools, a_call, says, scripted


async def test_a_step_runs_once_across_replays() -> None:
    log = MemorySteps()
    ran: list[str] = []

    async def workflow() -> str:
        o = Orchestrator("run-1", log)
        first = await o.run("send", lambda: _effect(ran, "send"))
        second = await o.run("bill", lambda: _effect(ran, "bill"))
        return f"{first}+{second}"

    assert await workflow() == "did:send+did:bill"
    assert await workflow() == "did:send+did:bill"  # replay
    assert ran == ["send", "bill"]  # each effect happened exactly once


async def test_a_signal_ends_the_attempt_until_it_is_answered() -> None:
    """Approval takes days, so waiting must cost nothing — the attempt stops instead of blocking."""
    log = MemorySteps()
    ran: list[str] = []

    async def workflow() -> str:
        o = Orchestrator("run-2", log)
        await o.run("draft", lambda: _effect(ran, "draft"))
        decision = await o.signal("signoff")
        if not decision["approved"]:
            return "rejected"
        await o.run("meds", lambda: _effect(ran, "meds"))
        return "sent"

    with pytest.raises(Suspended, match="signoff"):
        await workflow()
    assert ran == ["draft"]

    await Orchestrator("run-2", log).resolve("signoff", {"approved": True})

    assert await workflow() == "sent"
    # The replay did not re-draft. That is the property the whole design rests on.
    assert ran == ["draft", "meds"]


async def test_a_rejected_signal_skips_the_effect() -> None:
    log = MemorySteps()
    ran: list[str] = []
    await Orchestrator("run-3", log).resolve("signoff", {"approved": False})

    o = Orchestrator("run-3", log)
    decision = await o.signal("signoff")
    assert decision == {"approved": False}
    assert ran == []


async def test_a_duplicate_step_name_is_refused() -> None:
    """Silent otherwise: the second call would return the first one's value and never run."""
    o = Orchestrator("run-4")
    await o.run("once", lambda: "a")

    with pytest.raises(ValueError, match="duplicate step name"):
        await o.run("once", lambda: "b")


async def test_nondeterminism_is_captured_by_a_step() -> None:
    log = MemorySteps()
    clock = iter([100, 200])

    async def workflow() -> int:
        o = Orchestrator("run-5", log)
        return int(await o.run("now", lambda: next(clock)))

    assert await workflow() == 100
    assert await workflow() == 100  # the replay reads the record, not the clock


async def test_the_workflow_is_the_control_layer() -> None:
    """The agent drafts read-only work while the workflow owns mutating steps."""
    log = MemorySteps()
    effects: list[str] = []
    read_only = Tools(names=["fetch_labs", "review_history"])

    async def discharge() -> dict[str, Any]:
        o = Orchestrator("patient-7", log)
        plan = await o.run(
            "draft",
            lambda: run_agent(
                    react_loop(
                        scripted(says("", a_call("c1", "fetch_labs")), says("plan: rest")),
                        read_only,
                    )
            ),
        )
        ok = await o.signal("signoff")
        if not ok["approved"]:
            return {"status": "rejected", "plan": plan["content"]}
        await o.run("meds", lambda: _effect(effects, "pharmacy"))
        await o.run("bill", lambda: _effect(effects, "invoice"))
        return {"status": "sent", "plan": plan["content"]}

    with pytest.raises(Suspended, match="signoff"):
        await discharge()
    assert effects == []  # nothing left the building before a human said so
    assert read_only.ran == ["fetch_labs"]

    await Orchestrator("patient-7", log).resolve("signoff", {"approved": True})
    assert await discharge() == {"status": "sent", "plan": "plan: rest"}

    assert effects == ["pharmacy", "invoice"]
    assert read_only.ran == ["fetch_labs"]  # the replay did not re-run the agent


async def test_a_suspended_run_is_not_recorded_as_an_outcome() -> None:
    """A suspension is never memoized as a completed agent outcome."""
    log = MemorySteps()

    async def suspended_events() -> Any:
        yield {"type": "suspended", "pending_id": "approval-1", "tool_call_id": "c1"}

    async def workflow() -> Any:
        o = Orchestrator("run-8", log)
        return await o.run("draft", lambda: run_agent(suspended_events()))

    with pytest.raises(AgentSuspended) as raised:
        await workflow()
    assert (raised.value.pending_id, raised.value.tool_call_id) == ("approval-1", "c1")
    assert isinstance(raised.value, Suspended)  # a workflow can catch both the same way

    assert (await log.read("run-8", "draft")).status == "absent"


async def test_a_failed_run_is_not_recorded_as_an_outcome() -> None:
    """Otherwise the step memoises the failure and the retry never happens."""

    class Broken(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    log = MemorySteps()

    async def workflow() -> Any:
        o = Orchestrator("run-7", log)
        return await o.run(
            "draft", lambda: run_agent(react_loop(Broken(messages=iter([])), Tools()))
        )

    with pytest.raises(AgentFailed, match="provider down"):
        await workflow()

    assert (await log.read("run-7", "draft")).status == "absent"



async def test_the_agent_loop_composes_as_a_step() -> None:
    """The loop is a leaf: durability wraps it, and it knows nothing about any of this."""
    log = MemorySteps()
    models: list[str] = []

    async def one_run() -> Any:
        tools = Tools()
        llm = scripted(says("", a_call("c1", "read")), says("drafted"))
        models.append("called")
        outcome = await run_agent(react_loop(llm, tools))
        return outcome["content"]

    async def workflow() -> Any:
        o = Orchestrator("run-6", log)
        return await o.run("draft", one_run)

    assert await workflow() == "drafted"
    assert await workflow() == "drafted"
    assert models == ["called"]  # the second attempt never reached the provider


async def _effect(ran: list[str], name: str) -> str:
    ran.append(name)
    return f"did:{name}"
