"""One state-aware entry point — dispatch routes commands by what the ledger allows."""

from typing import Any, ClassVar

import pytest
from langchain_core.messages import HumanMessage
from semora import Agent, AgentRuntime
from semora.contracts import ToolCall
from semora.controls import ControlPlane, Permissions, gate
from semora.dispatch import (
    Answer,
    CommandRouter,
    InvalidTransition,
    Prompt,
    QueueSteer,
    Recover,
    RecoverInterrupted,
    ReplayJournal,
    ResumeApproval,
    StartRun,
)
from semora.orchestrator import AgentSuspended, MemorySteps, Orchestrator
from semora_store import Contended, MemoryTranscript

from tests.test_loop import Tools, a_call, says, scripted


def _runtime() -> AgentRuntime:
    return AgentRuntime(store=MemorySteps(), transcript=MemoryTranscript())


def _agent(model: Any, tools: Tools | None = None) -> Agent:
    return Agent("worker", "Works", model, tools if tools is not None else Tools())


async def _hold(call: ToolCall) -> dict[str, Any] | None:
    if call["name"] == "deploy":
        return {"type": "suspend", "pending_id": "approval-1"}
    return None


_HOLDING = ControlPlane(pre_tool_use=Permissions(gate(_hold)))


async def test_dispatch_requires_the_durable_collaborators() -> None:
    """State comes from the ledger and history from the transcript.

    Without both there is nothing to route by.
    """
    command = Prompt("hi")
    with pytest.raises(TypeError, match="dispatch requires"):
        await AgentRuntime(store=MemorySteps()).dispatch(
            "d-no-transcript", _agent(scripted(says("hi"))), command
        )
    with pytest.raises(TypeError, match="dispatch requires"):
        await AgentRuntime(transcript=MemoryTranscript()).dispatch(
            "d-no-store", _agent(scripted(says("hi"))), command
        )


async def test_a_prompt_starts_an_idle_run() -> None:
    outcome = await _runtime().dispatch(
        "d-fresh", _agent(scripted(says("done"))), Prompt("hello")
    )

    assert outcome["content"] == "done"
    assert outcome["stop_reason"] == "completed"


async def test_an_answer_resumes_the_parked_call() -> None:
    """The adapter sends an Answer without knowing it must call resume().

    That routing is the whole point of the single entry point.
    """
    runtime = _runtime()
    with pytest.raises(AgentSuspended):
        await runtime.dispatch(
            "d-answer",
            _agent(scripted(says("", a_call("c1", "deploy"))), Tools(names=["deploy"])),
            Prompt("ship it"),
            controls=_HOLDING,
        )

    tools = Tools(results={"deploy": {"type": "text", "text": "id-7"}}, names=["deploy"])
    outcome = await runtime.dispatch(
        "d-answer",
        _agent(scripted(says("deployed")), tools),
        Answer("approval-1", {"type": "text", "text": "approved"}),
        controls=_HOLDING,
    )

    assert tools.ran == ["deploy"]
    assert outcome["content"] == "deployed"


async def test_an_answer_with_no_park_is_an_invalid_transition() -> None:
    """The standard error carries the precise state so an adapter maps it without string-matching.

    ``completed``, not a bare ``idle``: the refusal names what the run record actually says, the
    same vocabulary ``Recover`` refusals already used.
    """
    runtime = _runtime()
    await runtime.dispatch("d-idle", _agent(scripted(says("done"))), Prompt("hello"))

    with pytest.raises(InvalidTransition) as caught:
        await runtime.dispatch(
            "d-idle",
            _agent(scripted(says("again"))),
            Answer("approval-1", {"type": "text", "text": "approved"}),
        )

    assert caught.value.state == "completed"
    assert isinstance(caught.value.command, Answer)


async def test_an_interactive_prompt_on_a_parked_run_cancels_the_park() -> None:
    """run()'s cancel-and-switch contract survives the dispatch route unchanged."""
    runtime = _runtime()
    with pytest.raises(AgentSuspended):
        await runtime.dispatch(
            "d-switch",
            _agent(scripted(says("", a_call("c1", "deploy"))), Tools(names=["deploy"])),
            Prompt("ship it"),
            controls=_HOLDING,
        )

    tools = Tools(names=["deploy"])
    outcome = await runtime.dispatch(
        "d-switch",
        _agent(scripted(says("dropped")), tools),
        Prompt("never mind", input_mode="interactive"),
        controls=_HOLDING,
    )

    assert tools.ran == []  # the parked deploy was cancelled, not approved
    assert outcome["content"] == "dropped"


async def test_a_headless_prompt_on_a_parked_run_queues_and_re_announces() -> None:
    """Headless input must not cancel a pending approval on its own."""
    runtime = _runtime()
    with pytest.raises(AgentSuspended):
        await runtime.dispatch(
            "d-queue",
            _agent(scripted(says("", a_call("c1", "deploy"))), Tools(names=["deploy"])),
            Prompt("ship it"),
            controls=_HOLDING,
        )

    with pytest.raises(AgentSuspended) as still_parked:
        await runtime.dispatch(
            "d-queue",
            _agent(scripted(says("later")), Tools(names=["deploy"])),
            Prompt("also do this", input_mode="headless"),
            controls=_HOLDING,
        )

    assert still_parked.value.pending_id == "approval-1"


async def test_recover_finishes_an_interrupted_round() -> None:
    """A crashed round is finished from the transcript without replaying its model turn."""
    runtime = _runtime()
    fail = True

    async def flaky(call: ToolCall) -> None:
        nonlocal fail
        if call["id"] == "c2" and fail:
            fail = False
            raise RuntimeError("policy unavailable")

    controls = ControlPlane(pre_tool_use=Permissions(gate(flaky)))
    first_model = scripted(says("", a_call("c1", "first"), a_call("c2", "second")))
    tools = Tools(names=["first", "second"])
    with pytest.raises(RuntimeError, match="policy unavailable"):
        await runtime.dispatch(
            "d-crash", _agent(first_model, tools), Prompt("go"), controls=controls
        )

    final_model = scripted(says("finished"))
    outcome = await runtime.dispatch(
        "d-crash", _agent(final_model, tools), Recover(), controls=controls
    )

    assert tools.ran == ["first", "second"]  # c1 not re-run, c2 finally executed
    assert len(final_model.seen) == 1  # the interrupted model turn came from the journal
    assert outcome["content"] == "finished"


async def test_recover_with_nothing_to_recover_is_an_invalid_transition() -> None:
    """Recover on a run that never started or already completed must not spend a model turn."""
    runtime = _runtime()
    with pytest.raises(InvalidTransition) as fresh:
        await runtime.dispatch("d-nothing", _agent(scripted()), Recover())
    assert fresh.value.state == "fresh"

    completed_model = scripted(says("done"))
    await runtime.dispatch("d-done", _agent(completed_model), Prompt("hello"))
    with pytest.raises(InvalidTransition) as completed:
        await runtime.dispatch("d-done", _agent(scripted(says("again"))), Recover())
    assert completed.value.state == "completed"


async def test_a_prompt_behind_a_live_worker_is_enqueued() -> None:
    """A busy run keeps its lease; the prompt lands durably for the live loop to drain."""
    store = MemorySteps()
    runtime = AgentRuntime(store=store, transcript=MemoryTranscript())
    model = scripted(says("never"))

    async with Orchestrator("d-busy", store, owner="other-worker"):
        result = await runtime.dispatch(
            "d-busy", _agent(model), Prompt("hi", prompt_id="p1")
        )

    assert result == {"type": "enqueued", "input_id": "p1"}
    assert model.seen == []  # no model turn was spent fighting for the lease


async def test_a_repeated_prompt_id_is_delivered_once() -> None:
    """Command idempotency: replaying the same Prompt must not deliver the input twice."""
    runtime = _runtime()
    await runtime.dispatch(
        "d-idem", _agent(scripted(says("first"))), Prompt("hello", prompt_id="p1")
    )

    replay_model = scripted(says("second"))
    await runtime.dispatch(
        "d-idem", _agent(replay_model), Prompt("hello", prompt_id="p1")
    )

    delivered = [
        m for m in replay_model.seen[0] if isinstance(m, HumanMessage) and m.content == "hello"
    ]
    assert len(delivered) == 1


async def test_a_prompt_routes_without_observing_the_run_state() -> None:
    """A framework must not charge every host for a read only some rows need.

    ``read_run`` is called from exactly one place — ``AgentRuntime.state`` — so counting it is
    counting observations. A Prompt matches on the command alone; if this ever reads, the
    router went back to observing eagerly and every dispatch pays for the Recover rows' needs.
    """

    class CountingTranscript(MemoryTranscript):
        def __init__(self) -> None:
            super().__init__()
            self.run_reads = 0

        async def read_run(self, run_id: str) -> dict[str, Any] | None:
            self.run_reads += 1
            return await super().read_run(run_id)

    transcript = CountingTranscript()
    runtime = AgentRuntime(store=MemorySteps(), transcript=transcript)

    outcome = await runtime.dispatch(
        "d-blind", _agent(scripted(says("done"))), Prompt("hello")
    )

    assert outcome["stop_reason"] == "completed"
    assert transcript.run_reads == 0  # routed by command alone; no state was observed


async def test_removing_queue_steer_makes_contention_the_hosts_problem() -> None:
    """The enqueue fallback is one removable row, not dispatch policy.

    Without ``QueueSteer`` a busy run surfaces ``Contended`` and the host decides — retry,
    queue elsewhere, or fail loudly. If this passes with the row present, the hand-off is
    being decided somewhere other than the table.
    """
    store = MemorySteps()
    runtime = AgentRuntime(store=store, transcript=MemoryTranscript())
    bare = CommandRouter(ResumeApproval(), RecoverInterrupted(), ReplayJournal(), StartRun())

    async with Orchestrator("d-bare", store, owner="other-worker"):
        with pytest.raises(Contended):
            await bare.dispatch(runtime, "d-bare", _agent(scripted(says("never"))), Prompt("hi"))


async def test_removing_the_recover_rows_refuses_the_command() -> None:
    """Journal replay is a row too: drop it and Recover on an interrupted round is refused.

    The refusal carries the observed state, so the host that removed the row still learns what
    it declined to handle.
    """
    runtime = _runtime()
    fail = True

    async def flaky(call: ToolCall) -> None:
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await runtime.dispatch(
            "d-norec",
            _agent(scripted(says("", a_call("c1", "first"))), Tools(names=["first"])),
            Prompt("go"),
            controls=ControlPlane(pre_tool_use=Permissions(gate(flaky))),
        )

    without_recover = CommandRouter(ResumeApproval(), StartRun(), QueueSteer())
    with pytest.raises(InvalidTransition) as refused:
        await without_recover.dispatch(runtime, "d-norec", _agent(scripted()), Recover())

    assert refused.value.state == "interrupted"


async def test_a_host_transition_slots_into_the_table() -> None:
    """A custom row composes with the presets by order alone — no subclassing, no registry."""

    class Reject:
        """Refuse prompts while anything is parked, instead of run()'s cancel-and-switch."""

        states: ClassVar[frozenset[str] | None] = frozenset({"waiting"})

        def applies(self, command: Any) -> bool:
            return isinstance(command, Prompt)

        async def apply(
            self,
            runtime: Any,
            run_id: Any,
            agent: Any,
            command: Any,
            state: str | None,
            *,
            controls: Any = None,
            **options: Any,
        ) -> dict[str, Any]:
            return {"type": "rejected", "reason": "answer the pending request first"}

    runtime = _runtime()
    with pytest.raises(AgentSuspended):
        await runtime.dispatch(
            "d-custom",
            _agent(scripted(says("", a_call("c1", "deploy"))), Tools(names=["deploy"])),
            Prompt("ship it"),
            controls=_HOLDING,
        )

    guarded = CommandRouter(Reject(), ResumeApproval(), StartRun(), QueueSteer())
    outcome = await guarded.dispatch(
        runtime, "d-custom", _agent(scripted(says("never"))), Prompt("do more")
    )

    assert outcome == {"type": "rejected", "reason": "answer the pending request first"}
