"""Event emission from the loop, and envelope identity."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from nexora import AgentRuntime
from nexora.contracts import (
    BLOCKING,
    EventEnvelope,
    EventStream,
    EventType,
    PendingInput,
    RuntimeEvents,
)
from nexora.contracts.types import ToolCall
from nexora.controls import (
    Continue,
    ControlPlane,
    Controls,
    Halt,
    Ingress,
    Permissions,
    Proceed,
    Suspend,
    ToolDecision,
    gate,
)
from nexora.engines.plain import react_loop
from nexora.orchestrator import AgentSuspended, MemorySteps
from nexora_store import ExecutionContext

from tests.test_loop import Tools, a_call, says, scripted


class Recorder:
    """A sink that keeps every envelope, wired the way a real caller would."""

    def __init__(self) -> None:
        self.seen: list[EventEnvelope] = []

    async def __call__(self, envelope: EventEnvelope) -> None:
        self.seen.append(envelope)

    def types(self) -> list[str]:
        return [e.event_type for e in self.seen]


async def test_runtime_isolates_observation_sink_failures() -> None:
    """A broken UI or monitoring destination must not change the agent outcome."""

    async def broken(_event: EventEnvelope) -> None:
        raise RuntimeError("monitor unavailable")

    outcome = await AgentRuntime(event_sink=broken).run(
        "run-1", scripted(says("finished")), Tools(), "go"
    )

    assert outcome["content"] == "finished"


async def test_runtime_envelopes_events_with_trusted_execution_scope() -> None:
    """Observation coordinates come from the host context rather than model-visible data."""
    recorder = Recorder()
    context = ExecutionContext(
        "run-1", session_id="session-1", namespace="company-thread", subject="user-7"
    )

    await AgentRuntime(event_sink=recorder).run(
        context, scripted(says("finished")), Tools(), "go"
    )

    envelope = recorder.seen[0]
    assert (envelope.run_id, envelope.session_id, envelope.thread_id) == (
        "run-1",
        "session-1",
        "company-thread",
    )


def a_stream(sink: Recorder) -> EventStream:
    return EventStream(sink, session_id="s1", thread_id="t1", run_id="r1")


async def test_a_tool_round_emits_the_full_hook_sequence() -> None:
    llm = scripted(says("", a_call("c1", "read")), says("finished"))
    sink = Recorder()

    await AgentRuntime(store=MemorySteps(), emit=a_stream(sink)).run(
        "event-round", llm, Tools(), "hi"
    )

    assert sink.types() == [
        EventType.USER_PROMPT_SUBMIT,
        EventType.CONTEXT_INJECTED,
        EventType.PRE_TOOL_USE,
        EventType.POST_TOOL_USE,
        EventType.CONTEXT_INJECTED,
        EventType.STOP,
    ]


async def test_a_pruned_continue_after_suspension_emits_no_orphan_gate_event() -> None:
    """A call removed from the round must leave no PRE_TOOL_USE without a terminal event."""
    observed: list[tuple[str, str]] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        if event_type in {EventType.PRE_TOOL_USE, EventType.PERMISSION_REQUEST}:
            observed.append((event_type, str(payload["call_id"])))

    async def suspend_first(ctx: Any, call: ToolCall) -> ToolDecision:
        if call["id"] == "c1":
            return Suspend({"type": "suspend", "pending_id": "approve-c1"})
        return Continue()

    controls = ControlPlane(pre_tool_use=Permissions(suspend_first))
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=MemorySteps(), emit=emit).run(
            "pruned-gate-event",
            scripted(says("", a_call("c1", "read"), a_call("c2", "read"))),
            Tools(),
            "go",
            controls=controls,
        )

    assert observed == [
        (EventType.PRE_TOOL_USE, "c1"),
        (EventType.PERMISSION_REQUEST, "c1"),
    ]


async def test_a_failing_tool_emits_the_failure_variant() -> None:
    llm = scripted(says("", a_call("c1", "read")), says("x"))
    tools = Tools(results={"read": {"type": "error", "message": "nope"}})
    sink = Recorder()

    async for _ in react_loop(llm, tools, emit=a_stream(sink)):
        pass

    assert EventType.POST_TOOL_USE_FAILURE in sink.types()
    assert EventType.POST_TOOL_USE not in sink.types()


async def test_a_submission_is_announced_once_and_carries_no_prompt_text() -> None:
    """The payload identifies a submission and nothing more.

    Text here would be the pre-mask original — this event fires at the inbox, before `on_inputs`
    can screen anything. The admitted version is `CONTEXT_INJECTED`'s job.
    """
    sink = Recorder()

    await AgentRuntime(store=MemorySteps(), emit=a_stream(sink)).run(
        "event-prompt", scripted(says("done")), Tools(), "hello"
    )

    assert sink.types().count(EventType.USER_PROMPT_SUBMIT) == 1
    submitted = next(e for e in sink.seen if e.event_type == EventType.USER_PROMPT_SUBMIT)
    assert set(submitted.payload) == {"input_id", "source"}
    assert submitted.payload["source"] == "user_prompt"
    assert submitted.payload["input_id"]


async def test_a_run_with_no_prompt_announces_no_submission() -> None:
    """Continuing a run without a prompt emits no submission event."""
    resumed = Recorder()

    await AgentRuntime(store=MemorySteps(), emit=a_stream(resumed)).run(
        "event-resume", scripted(says("done")), Tools()
    )

    assert resumed.types() == [EventType.STOP]


async def test_a_masked_prompt_leaves_no_original_in_the_event_stream() -> None:
    """The submission path end to end, which is where the leak was.

    `test_loop.py` masks an input that arrives by `drain_inputs`, so `USER_PROMPT_SUBMIT` never
    fires there and the original could ride out on it unnoticed. Going through `run()` puts the
    prompt through `enqueue_input` first, the way a real caller does.
    """
    sink = Recorder()

    async def mask(ctx: Any, inputs: list[PendingInput]) -> list[PendingInput]:
        return [
            PendingInput(
                item.kind,
                HumanMessage(str(item.message.content).replace("123-45", "***")),
                item.origin_id,
            )
            for item in inputs
        ]

    await AgentRuntime(store=MemorySteps(), emit=a_stream(sink)).run(
        "masked-prompt",
        scripted(says("done")),
        Tools(),
        "ssn is 123-45",
        controls=ControlPlane(on_inputs=Ingress(mask)),
    )

    assert EventType.USER_PROMPT_SUBMIT in sink.types()  # the announcing path really ran
    assert "123-45" not in str([e.payload for e in sink.seen])
    injected = [e.payload for e in sink.seen if e.event_type == EventType.CONTEXT_INJECTED]
    assert injected and "***" in str(injected)


async def test_inputs_are_recorded_when_they_enter_the_model_context() -> None:
    sink = Recorder()
    queued = [PendingInput("user_steer", HumanMessage("wait"), "prompt-2")]

    async def drain() -> list[PendingInput]:
        return [queued.pop()] if queued else []

    async def control(_ctx: Any) -> Proceed:
        return Proceed([HumanMessage("policy context")])

    model = scripted(says("done"))
    async for _ in react_loop(
        model,
        Tools(),
        drain_inputs=drain,
        controls=ControlPlane(before_model=control),
        emit=a_stream(sink),
    ):
        pass

    injected = [e for e in sink.seen if e.event_type == EventType.CONTEXT_INJECTED]
    assert [(e.payload["turn"], e.payload["kind"], e.payload["origin_id"]) for e in injected] == [
        (0, "user_steer", "prompt-2"),
        (0, "control", None),
    ]
    assert [e.payload["message"]["type"] for e in injected] == ["human", "human"]
    assert [message.content for message in model.seen[0][-2:]] == ["wait", "policy context"]


async def test_a_halt_does_not_claim_that_drained_input_reached_the_model() -> None:
    sink = Recorder()
    queued = [PendingInput("user_steer", HumanMessage("wait"), "prompt-2")]

    async def drain() -> list[PendingInput]:
        return [queued.pop()] if queued else []

    async def halt(_ctx: Any) -> Halt:
        return Halt("policy")

    model = scripted(says("never"))
    async for _ in react_loop(
        model,
        Tools(),
        drain_inputs=drain,
        controls=ControlPlane(before_model=halt),
        emit=a_stream(sink),
    ):
        pass

    assert EventType.CONTEXT_INJECTED not in sink.types()
    assert model.seen == []


async def test_a_late_input_is_injected_into_the_next_model_turn() -> None:
    sink = Recorder()
    drains = 0

    async def drain() -> list[PendingInput]:
        nonlocal drains
        drains += 1
        if drains == 2:
            return [PendingInput("user_steer", HumanMessage("wait"), "prompt-late")]
        if drains == 3:
            return [PendingInput("user_steer", HumanMessage("also wait"), "prompt-new")]
        return []

    model = scripted(says("almost"), says("really done"))
    async for _ in react_loop(
        model,
        Tools(),
        drain_inputs=drain,
        emit=a_stream(sink),
    ):
        pass

    injected = [e for e in sink.seen if e.event_type == EventType.CONTEXT_INJECTED]
    assert [(e.payload["turn"], e.payload["origin_id"]) for e in injected] == [
        (1, "prompt-late"),
        (1, "prompt-new"),
    ]
    assert [message.content for message in model.seen[1][-2:]] == ["wait", "also wait"]


async def test_the_gate_decision_is_announced() -> None:
    llm = scripted(says("", a_call("c1", "rm")), says("x"))
    sink = Recorder()

    async def deny(c: Any) -> dict[str, Any]:
        return {"type": "error", "message": "not allowed"}

    stream = a_stream(sink)
    async for _ in react_loop(
        llm,
        Tools(),
        controls=ControlPlane(pre_tool_use=Permissions(gate(deny))),
        emit=stream,
    ):
        pass

    assert EventType.PERMISSION_DENIED in sink.types()


async def test_an_ask_decision_announces_a_permission_request() -> None:
    llm = scripted(says("", a_call("c1", "deploy")))
    sink = Recorder()

    async def ask(c: Any) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "p1"}

    stream = a_stream(sink)
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=MemorySteps(), emit=stream).run(
            "ask-event",
            llm,
            Tools(),
            "hi",
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask))),
        )

    assert EventType.PERMISSION_REQUEST in sink.types()
    assert EventType.POST_TOOL_USE not in sink.types()
    assert EventType.POST_TOOL_USE_FAILURE not in sink.types()
    assert EventType.STOP not in sink.types()  # suspended, not stopped


async def test_stop_carries_the_reason() -> None:
    sink = Recorder()

    async for _ in react_loop(scripted(says("bye")), Tools(), emit=a_stream(sink)):
        pass

    stop = sink.seen[-1]
    assert stop.event_type == EventType.STOP
    assert stop.payload["reason"] == "completed"


async def test_runtime_boundaries_emit_the_external_lifecycle_vocabulary() -> None:
    sink = Recorder()
    events = RuntimeEvents(a_stream(sink))

    await events.session_start("startup")
    await events.setup(version="1")
    await events.config_change({"model": "new"})
    await events.cwd_changed("/old", "/new")
    await events.instructions_loaded(["AGENTS.md"])
    await events.pre_compact("budget")
    await events.post_compact("budget")
    await events.subagent_start("agent-1", "search")
    await events.subagent_stop("agent-1", "completed")
    await events.task_created("task-1", "index")
    await events.task_completed("task-1", {"ok": True})
    await events.teammate_idle("agent-2")
    await events.elicitation("request-1", {"question": "continue?"})
    await events.elicitation_result("request-1", {"answer": "yes"})
    await events.notification("finished")
    await events.session_end("clear")

    assert sink.types() == [
        EventType.SESSION_START,
        EventType.SETUP,
        EventType.CONFIG_CHANGE,
        EventType.CWD_CHANGED,
        EventType.INSTRUCTIONS_LOADED,
        EventType.PRE_COMPACT,
        EventType.POST_COMPACT,
        EventType.SUBAGENT_START,
        EventType.SUBAGENT_STOP,
        EventType.TASK_CREATED,
        EventType.TASK_COMPLETED,
        EventType.TEAMMATE_IDLE,
        EventType.ELICITATION,
        EventType.ELICITATION_RESULT,
        EventType.NOTIFICATION,
        EventType.SESSION_END,
    ]
    assert sink.seen[0].payload == {"source": "startup"}


async def test_orchestrator_owns_one_publisher_for_lifecycle_and_agent_events() -> None:
    sink = Recorder()
    runtime = AgentRuntime(store=MemorySteps(), emit=a_stream(sink))

    await runtime.events.session_start("startup")
    await runtime.run(
        "r1",
        scripted(says("", a_call("c1", "read")), says("done")),
        Tools(),
        "hello",
    )
    await runtime.events.session_end("completed")

    assert sink.types() == [
        EventType.SESSION_START,
        EventType.USER_PROMPT_SUBMIT,
        EventType.CONTEXT_INJECTED,
        EventType.PRE_TOOL_USE,
        EventType.POST_TOOL_USE,
        EventType.CONTEXT_INJECTED,
        EventType.STOP,
        EventType.SESSION_END,
    ]


# ── Envelope ─────────────────────────────────────────────────────────────────


async def test_event_id_is_derived_so_a_replayed_event_dedupes() -> None:
    a, b = Recorder(), Recorder()
    payload = {"turn": 0, "call_id": "c1"}

    await a_stream(a)(EventType.POST_TOOL_USE, payload)
    await a_stream(b)(EventType.POST_TOOL_USE, payload)

    assert a.seen[0].event_id == b.seen[0].event_id


async def test_event_id_differs_by_position_in_the_run() -> None:
    sink = Recorder()
    stream = a_stream(sink)

    await stream(EventType.POST_TOOL_USE, {"turn": 0, "call_id": "c1"})
    await stream(EventType.POST_TOOL_USE, {"turn": 0, "call_id": "c1"})

    assert [e.sequence for e in sink.seen] == [0, 1]
    assert sink.seen[0].event_id != sink.seen[1].event_id


def test_every_blocking_event_names_a_control_point_that_exists() -> None:
    """A `BLOCKING` entry promises a gate decides that event, so the gate has to be real.

    `PRE_COMPACT` and `ELICITATION` sat in this table while `Controls` had no method for either —
    a contract advertising a decision point that nothing could decide. This fails on the next one
    instead of documenting it.
    """
    for event_type, control_point in BLOCKING.items():
        assert hasattr(Controls, control_point), (
            f"{event_type} names control point {control_point!r}, absent from `Controls`"
        )


async def test_announcing_a_tool_call_and_asking_the_gate_are_one_step() -> None:
    """Per call: the event, then the decision it announces. Nothing between them, nothing missing.

    The pairing is code placement in `decide_tool_call`, not a type — an engine that emitted
    `PRE_TOOL_USE` and skipped the control point would run the call unguarded, and every default
    in `ControlPlane` is fail-open. Two calls in the round, because with one an interleaving bug
    and correct behaviour look the same.
    """
    # ponytail: asserted for `engines/plain`, the only engine there is. A second engine needs this
    # test parametrized over engines, not a conformance harness built before there is a second one.
    trail: list[str] = []

    class Watching(Recorder):
        async def __call__(self, envelope: EventEnvelope) -> None:
            if envelope.event_type == EventType.PRE_TOOL_USE:
                trail.append(f"event:{envelope.payload['call_id']}")
            await super().__call__(envelope)

    async def watch(ctx: Any, call: ToolCall) -> ToolDecision:
        trail.append(f"control:{call['id']}")
        return Continue()

    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "read")), says("done"))
    await AgentRuntime(store=MemorySteps(), emit=a_stream(Watching())).run(
        "gate-pairing",
        llm,
        Tools(),
        "hi",
        controls=ControlPlane(pre_tool_use=Permissions(watch)),
    )

    assert trail == ["event:c1", "control:c1", "event:c2", "control:c2"]


async def test_no_answer_on_this_channel_can_decide_anything() -> None:
    """A subscriber that tries to decide is ignored, and one that dies is logged and skipped.

    Both together are the point: this channel is observation, so nothing load-bearing may ride
    on it. `BLOCKING` only says *which* events describe a decision point and *which* control point
    decides it — the decision itself is that call, whose answer comes back by `return`.
    """
    assert EventType.PRE_TOOL_USE in BLOCKING  # names a decision point
    assert EventType.POST_TOOL_USE not in BLOCKING

    async def opinionated(envelope: EventEnvelope) -> Any:
        if envelope.payload.get("boom"):
            raise RuntimeError("socket died")
        return {"type": "error", "message": "denied"}

    stream = EventStream(opinionated, session_id="s", thread_id="t", run_id="r")

    await stream(EventType.PRE_TOOL_USE, {"turn": 0})  # answer discarded
    await stream(EventType.PRE_TOOL_USE, {"turn": 0, "boom": True})  # swallowed
