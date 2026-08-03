"""Event emission from the loop, and envelope identity."""

from typing import Any

from nexora.contracts import BLOCKING, EventEnvelope, EventStream, EventType
from nexora.engines.plain import react_loop
from tests.test_loop import Tools, a_call, says, scripted


class Recorder:
    """A sink that keeps every envelope, wired the way a real caller would."""

    def __init__(self) -> None:
        self.seen: list[EventEnvelope] = []

    async def __call__(self, envelope: EventEnvelope) -> None:
        self.seen.append(envelope)

    def types(self) -> list[str]:
        return [e.event_type for e in self.seen]


def a_stream(sink: Recorder) -> EventStream:
    return EventStream(sink, session_id="s1", thread_id="t1", run_id="r1")


async def test_a_tool_round_emits_the_full_hook_sequence() -> None:
    llm = scripted(says("", a_call("c1", "read")), says("finished"))
    sink = Recorder()

    async for _ in react_loop(llm, Tools(), "hi", emit=a_stream(sink)):
        pass

    assert sink.types() == [
        EventType.PRE_TOOL_USE,
        EventType.POST_TOOL_USE,
        EventType.POST_TOOL_BATCH,
        EventType.STOP,
    ]


async def test_a_failing_tool_emits_the_failure_variant() -> None:
    llm = scripted(says("", a_call("c1", "read")), says("x"))
    tools = Tools(results={"read": {"type": "error", "message": "nope"}})
    sink = Recorder()

    async for _ in react_loop(llm, tools, "hi", emit=a_stream(sink)):
        pass

    assert EventType.POST_TOOL_USE_FAILURE in sink.types()
    assert EventType.POST_TOOL_USE not in sink.types()


async def test_the_gate_decision_is_announced() -> None:
    llm = scripted(says("", a_call("c1", "rm")), says("x"))
    sink = Recorder()

    async def deny(c: Any) -> dict[str, Any]:
        return {"type": "error", "message": "not allowed"}

    async for _ in react_loop(
        llm, Tools(), "hi", before_tool_call=deny, emit=a_stream(sink)
    ):
        pass

    assert EventType.PERMISSION_DENIED in sink.types()


async def test_an_ask_decision_announces_a_permission_request() -> None:
    llm = scripted(says("", a_call("c1", "deploy")))
    sink = Recorder()

    async def ask(c: Any) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "p1"}

    async for _ in react_loop(llm, Tools(), "hi", before_tool_call=ask, emit=a_stream(sink)):
        pass

    assert EventType.PERMISSION_REQUEST in sink.types()
    assert EventType.STOP not in sink.types()  # suspended, not stopped


async def test_stop_carries_the_reason() -> None:
    sink = Recorder()

    async for _ in react_loop(scripted(says("bye")), Tools(), "hi", emit=a_stream(sink)):
        pass

    stop = sink.seen[-1]
    assert stop.event_type == EventType.STOP
    assert stop.payload["reason"] == "completed"


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


def test_blocking_events_are_hooks_not_subscriptions() -> None:
    """A blocking event's answer must reach the loop, so it is never published via `emit`."""
    assert EventType.PRE_TOOL_USE in BLOCKING
    assert EventType.POST_TOOL_USE not in BLOCKING
