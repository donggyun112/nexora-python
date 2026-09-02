"""`semora_fork.fork_run` — re-running from before one injected input, under new controls."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from semora import AgentRuntime, PendingInput
from semora.controls import ControlPlane, Ingress
from semora_fork import (
    RERUNS,
    EventCheckpoint,
    ForkCoordinate,
    fork_event,
    fork_run,
    record_event_checkpoint,
    resume_point,
)
from semora_store import MemorySteps, MemoryTranscript

from tests.test_loop import Tools, says, scripted


async def mask(_ctx: Any, inputs: list[PendingInput]) -> list[PendingInput]:
    return [
        PendingInput(
            item.kind,
            HumanMessage(str(item.message.content).replace("123-45", "***")),
            item.origin_id,
        )
        for item in inputs
    ]


async def a_masked_conversation(runtime: AgentRuntime) -> None:
    """Two turns on one conversation; the second prompt enters context masked."""
    await runtime.run("run-a", scripted(says("hello")), Tools(), "intro", conversation_id="conv")
    await runtime.run(
        "run-b",
        scripted(says("acted on masked text")),
        Tools(),
        "ssn is 123-45",
        prompt_id="p2",
        controls=ControlPlane(on_inputs=Ingress(mask)),
        conversation_id="conv",
    )


async def test_the_fork_feeds_the_ledger_original_after_the_untouched_prefix() -> None:
    """The forked model must see the pre-mask original from the ledger, not the transcript copy.

    Breaks if the cut point misses the injected message, or if the fork replays the screened
    transcript copy instead of reading the ledger.
    """
    steps = MemorySteps()
    runtime = AgentRuntime(store=steps, transcript=MemoryTranscript())
    await a_masked_conversation(runtime)

    forked = scripted(says("acted on original"))
    outcome = await fork_run(
        runtime,
        steps,
        from_run_id="run-b",
        origin_id="p2",
        run_id="run-c",
        model=forked,
        tools=Tools(),
        conversation_id="conv",
    )

    assert outcome["content"] == "acted on original"
    assert [m.content for m in forked.seen[0]] == ["intro", "hello", "ssn is 123-45"]


async def test_the_fork_moves_the_conversation_head_and_keeps_the_source_ledger() -> None:
    """The active branch shows the original lineage; the source ledger record is untouched.

    Breaks if the fork rewrites the source ledger, or lands on a fresh conversation instead of
    rewinding this one.
    """
    steps = MemorySteps()
    runtime = AgentRuntime(store=steps, transcript=MemoryTranscript())
    await a_masked_conversation(runtime)

    await fork_run(
        runtime,
        steps,
        from_run_id="run-b",
        origin_id="p2",
        run_id="run-c",
        model=scripted(says("acted on original")),
        tools=Tools(),
        conversation_id="conv",
    )

    head = await runtime.committed_history("run-c", "conv")
    assert [m.content for m in head] == [
        "intro",
        "hello",
        "ssn is 123-45",
        "acted on original",
    ]
    source = next(r for r in await steps.list_inputs("run-b") if r.input_id == "p2")
    assert "123-45" in str(source.value)
    assert [r.input_id for r in await steps.list_inputs("run-c")] == ["p2"]


async def test_the_fork_demands_a_ledger_record_that_reached_model_context() -> None:
    """Without a ledger original or an injection point there is nothing to fork from."""
    steps = MemorySteps()
    runtime = AgentRuntime(store=steps, transcript=MemoryTranscript())
    await a_masked_conversation(runtime)

    with pytest.raises(ValueError, match="no ledger record"):
        await fork_run(
            runtime,
            steps,
            from_run_id="run-b",
            origin_id="ghost",
            run_id="run-c",
            model=scripted(says("never")),
            tools=Tools(),
            conversation_id="conv",
        )

    await runtime.submit("run-x", PendingInput("user_prompt", HumanMessage("queued"), "px"))
    with pytest.raises(ValueError, match="never entered the model context"):
        await fork_run(
            runtime,
            steps,
            from_run_id="run-x",
            origin_id="px",
            run_id="run-c",
            model=scripted(says("never")),
            tools=Tools(),
        )


def _message_leaf(entries: list[dict[str, Any]], message_id: str) -> str:
    for entry in entries:
        message = entry.get("message")
        data = message.get("data") if isinstance(message, dict) else None
        if isinstance(data, dict) and data.get("id") == message_id:
            return str(entry["uuid"])
    raise AssertionError(f"no transcript message {message_id!r}")


async def test_event_fork_before_replays_the_checkpoint_input_original() -> None:
    """A before-edge event fork must reuse the source ledger original, not its masked leaf."""
    steps = MemorySteps()
    transcript = MemoryTranscript()
    runtime = AgentRuntime(store=steps, transcript=transcript)
    await a_masked_conversation(runtime)
    entries = await transcript.read("conv")
    masked_leaf = _message_leaf(entries, "p2")
    await record_event_checkpoint(
        transcript,
        EventCheckpoint(
            event_id="event-before-model",
            conversation_id="conv",
            before=ForkCoordinate("run-b", "p2", masked_leaf),
            after=ForkCoordinate("run-b", "p2", masked_leaf),
        ),
    )

    forked = scripted(says("acted on original"))
    await fork_event(
        runtime,
        steps,
        transcript,
        event_id="event-before-model",
        edge="before",
        run_id="run-c",
        model=forked,
        tools=Tools(),
        conversation_id="conv",
    )

    assert [message.content for message in forked.seen[0]] == [
        "intro",
        "hello",
        "ssn is 123-45",
    ]


async def test_event_fork_after_continues_from_the_recorded_leaf() -> None:
    """An after-edge event fork must continue at its leaf without reviving the input queue."""
    steps = MemorySteps()
    transcript = MemoryTranscript()
    runtime = AgentRuntime(store=steps, transcript=transcript)
    await a_masked_conversation(runtime)
    entries = await transcript.read("conv")
    masked_leaf = _message_leaf(entries, "p2")
    await record_event_checkpoint(
        transcript,
        EventCheckpoint(
            event_id="event-after-input",
            conversation_id="conv",
            before=ForkCoordinate("run-b", "p2", masked_leaf),
            after=ForkCoordinate("run-b", "p2", masked_leaf),
        ),
    )

    forked = scripted(says("continued after event"))
    await fork_event(
        runtime,
        steps,
        transcript,
        event_id="event-after-input",
        edge="after",
        run_id="run-c",
        model=forked,
        tools=Tools(),
        conversation_id="conv",
    )

    assert [message.content for message in forked.seen[0]] == [
        "intro",
        "hello",
        "ssn is ***",
    ]
    assert await steps.list_inputs("run-c") == []


async def test_event_fork_rejects_unknown_event_without_queuing_input() -> None:
    """An absent checkpoint must fail before the destination run receives any input."""
    steps = MemorySteps()
    transcript = MemoryTranscript()
    runtime = AgentRuntime(store=steps, transcript=transcript)

    with pytest.raises(ValueError, match="no fork checkpoint"):
        await fork_event(
            runtime,
            steps,
            transcript,
            event_id="missing",
            edge="before",
            run_id="run-c",
            model=scripted(says("never")),
            tools=Tools(),
            conversation_id="conv",
        )

    assert await steps.list_inputs("run-c") == []


async def test_event_fork_rejects_an_unknown_edge() -> None:
    """The edge name must select one of the checkpoint's two explicit coordinates."""
    steps = MemorySteps()
    transcript = MemoryTranscript()
    runtime = AgentRuntime(store=steps, transcript=transcript)

    with pytest.raises(ValueError, match="edge must be"):
        await fork_event(
            runtime,
            steps,
            transcript,
            event_id="anything",
            edge="middle",
            run_id="run-c",
            model=scripted(says("never")),
            tools=Tools(),
            conversation_id="conv",
        )


def test_a_coordinate_says_which_control_point_it_resumes_at() -> None:
    """Where a fork picks the conversation up decides which policies run — so say it.

    An operator choosing a branch point for a policy needs this before branching: a
    result coordinate never re-runs the gate or the journal for the recorded call, and
    the console had been guessing that from row labels.
    """
    asked = AIMessage(content="", tool_calls=[{"id": "c1", "name": "read", "args": {}}])
    answered = ToolMessage(content="ok", tool_call_id="c1")
    leaf = ForkCoordinate("run-1", None, "leaf")

    assert resume_point([HumanMessage("hi")], ForkCoordinate("run-1", "p1", None)) == "on_inputs"
    assert resume_point([HumanMessage("hi"), asked], leaf) == "pre_tool_use"
    assert resume_point([HumanMessage("hi"), asked, answered], leaf) == "before_model"
    assert resume_point([HumanMessage("hi"), asked, answered, AIMessage("done")], leaf) == (
        "before_model"
    )


def test_what_reruns_from_a_resume_point_is_the_loop_order() -> None:
    assert RERUNS["pre_tool_use"] == ("pre_tool_use", "post_tool_use", "before_finish")
    assert RERUNS["before_model"] == ("before_model", "before_finish")
    assert RERUNS["on_inputs"][0] == "on_inputs" and "post_tool_use" in RERUNS["on_inputs"]
    assert "pre_tool_use" not in RERUNS["before_model"], "a recorded call is not re-gated"
