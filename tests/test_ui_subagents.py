"""The console's subagent surface: what it shows, what it can pull, what it can reach."""

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from nexora import Compiled
from nexora.runtime import AgentRuntime
from nexora_store import MemoryTranscript
from nexora_ui.execution import AgentEvent, stream_attempt
from nexora_ui.recording import Recorder, history_of
from nexora_ui.state import RuntimeState

fastapi = pytest.importorskip("fastapi", reason="the console is the `ui` extra")
from fastapi.testclient import TestClient  # noqa: E402
from nexora_ui.app import app  # noqa: E402


def answering(text: str, *, delay: float = 0.0) -> Any:
    """A child that answers its parent on purpose, the way a handed-off one must."""

    async def run(_prompt: str, reply: Any, _run_id: str) -> Any:
        if delay:
            await asyncio.sleep(delay)
        await reply(text, False)
        yield {"type": "done", "content": text}

    return run


async def frames_of(run_id: str, attempt: Any, state: RuntimeState) -> list[dict[str, Any]]:
    return [json.loads(line) async for line in stream_attempt(run_id, attempt, state)]


async def test_a_delegated_child_streams_into_the_console_while_it_works() -> None:
    """An operator watching a delegation should see the child, not only the parent's summary."""
    state = RuntimeState()

    async def attempt(_runtime: AgentRuntime, tools: Any, _on_event: AgentEvent) -> dict[str, Any]:
        tools._subagents["scout"] = Compiled("scout", "scouts", answering("found it"))
        await tools.execute("delegate", "c1", {"agent": "scout", "input": "look"})
        return {"type": "done"}

    frames = await frames_of("child-visible", attempt, state)
    children = [frame for frame in frames if frame["kind"] == "child"]

    assert [frame["agent"] for frame in children] == ["scout"]
    assert children[0]["event"]["type"] == "done"


async def test_the_console_announces_a_child_starting_and_stopping() -> None:
    """`subagent_start`/`subagent_stop` are what the rail draws a child from."""
    state = RuntimeState()

    async def attempt(_runtime: AgentRuntime, tools: Any, _on_event: AgentEvent) -> dict[str, Any]:
        tools._subagents["scout"] = Compiled("scout", "scouts", answering("found it"))
        await tools.execute("delegate", "c1", {"agent": "scout", "input": "look"})
        return {"type": "done"}

    frames = await frames_of("child-events", attempt, state)
    lifecycle = [f for f in frames if f["kind"] == "lifecycle" and f["type"].startswith("subagent")]

    assert [f["type"] for f in lifecycle] == ["subagent_start", "subagent_stop"]
    assert lifecycle[0]["payload"]["run_id"] == "child-events:c1"
    assert lifecycle[1]["payload"]["reason"] == "completed"


async def test_an_independent_agent_is_listed_by_address_and_not_as_a_task() -> None:
    """The console has to keep the two apart: one is cancellable, the other is only reachable."""
    state = RuntimeState()

    async def attempt(_runtime: AgentRuntime, tools: Any, _on_event: AgentEvent) -> dict[str, Any]:
        tools._subagents["scout"] = Compiled("scout", "scouts", answering("mine"))
        await tools.execute("delegate", "c1", {"agent": "scout", "input": "x", "wait": "none"})
        return {"type": "done"}

    await frames_of("opened", attempt, state)
    session = state.sessions["opened"]

    assert [item["run_id"] for item in session.opened] == ["opened:c1"]
    assert session.tasks.list() == [], "an independent agent is not on the parent's leash"


async def test_a_handed_off_child_is_listed_as_a_task_the_console_can_cancel() -> None:
    """The other half: what `wait="async"` launches stays reachable by `cancel_task`."""
    state = RuntimeState()

    async def attempt(_runtime: AgentRuntime, tools: Any, _on_event: AgentEvent) -> dict[str, Any]:
        tools._subagents["slow"] = Compiled("slow", "slow", answering("late", delay=1.0))
        await tools.execute("delegate", "c1", {"agent": "slow", "input": "x", "wait": "async"})
        return {"type": "done"}

    await frames_of("handed-off", attempt, state)
    session = state.sessions["handed-off"]

    assert [(t["label"], t["status"]) for t in session.tasks.list()] == [("slow", "running")]
    assert session.tasks.cancel(session.tasks.list()[0]["task_id"]) is True


def test_the_tasks_route_answers_for_a_run_that_never_launched_anything() -> None:
    """An empty rail is a normal state, not a 404 the console has to special-case."""
    with TestClient(app) as client:
        body = client.get("/api/tasks/never-ran")

    assert body.status_code == 200
    assert body.json() == {"run_id": "never-ran", "background": [], "independent": []}


def test_cancelling_a_task_on_an_unknown_run_is_refused_rather_than_silently_accepted() -> None:
    """A cancel that reports success while cancelling nothing is worse than an error."""
    with TestClient(app) as client:
        answer = client.post("/api/tasks/cancel", json={"run_id": "nope", "task_id": "t1"})

    assert answer.status_code == 404


def test_the_console_page_offers_every_delegation_mode() -> None:
    """A mode with no button is a mode nobody will test from here."""
    with TestClient(app) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text

    for mode in ("wait=sync", "wait=async", "wait=none"):
        assert mode in page, f"the console offers no way to try {mode}"
    assert "tasks" in page and "check_tasks" in page
    assert 'id="agents"' in page, "the subagent rail needs somewhere to render"
    assert "/api/tasks/" in script and "/api/attach" in script


# ── Continuity ──────────────────────────────────────────────────────────────


async def test_a_run_remembers_its_earlier_turns_when_it_is_reached_again() -> None:
    """The gap the console found: reachable by run id, and answering as a stranger.

    `AgentRuntime` persists no transcript, so a second turn on the same run id had nothing to read
    back. `Recorder` is what the console hands to `history` instead — without it the agent is asked
    twice and remembers neither.
    """
    store = MemoryTranscript()
    first = await Recorder.open(store, "conv", "remember the number 41")
    await first.observe({"type": "text", "text": "41 it is."})
    await first.observe({"type": "done", "stop_reason": "completed"})
    await first.closed()

    assert [(type(m).__name__, m.content) for m in await history_of(store, "conv")] == [
        ("HumanMessage", "remember the number 41"),
        ("AIMessage", "41 it is."),
    ]


async def test_a_tool_round_is_recorded_as_the_model_saw_it() -> None:
    """A `ToolMessage` may only answer a call the assistant turn before it actually made."""
    store = MemoryTranscript()
    recorder = await Recorder.open(store, "conv", "store it")
    await recorder.observe({"type": "text", "text": "storing"})
    await recorder.observe({"type": "tool_call", "id": "c1", "name": "echo", "input": {"t": "x"}})
    await recorder.observe(
        {"type": "tool_result", "id": "c1", "result": {"type": "text", "text": "x"}}
    )
    await recorder.observe({"type": "text", "text": "stored"})
    await recorder.closed()

    restored = await history_of(store, "conv")
    shape = [type(message).__name__ for message in restored]

    answered = restored[2]
    assert shape == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert [call["id"] for call in restored[1].tool_calls] == ["c1"]  # type: ignore[attr-defined]
    assert isinstance(answered, ToolMessage) and answered.tool_call_id == "c1"
    assert restored[3].content == "stored"


async def test_reaching_a_run_again_continues_its_branch_instead_of_forking_one() -> None:
    """The bug the live console found: a recorder per turn forked the conversation every turn.

    Each new writer started at `parent_uuid=None`, so `active_branch` returned only the newest
    fork and everything said before sat one link away as a sibling. The agent was reachable and
    remembered nothing.
    """
    store = MemoryTranscript()
    first = await Recorder.open(store, "conv", "the number is 41")
    await first.observe({"type": "text", "text": "noted"})
    await first.closed()

    carried = await history_of(store, "conv")
    second = await Recorder.open(store, "conv", "what was the number?")
    await second.observe({"type": "text", "text": "41"})
    await second.closed()

    assert [m.content for m in carried] == ["the number is 41", "noted"]
    assert [m.content for m in await history_of(store, "conv")] == [
        "the number is 41",
        "noted",
        "what was the number?",
        "41",
    ]
