"""A tool call is a step. What that does to a run that resumes."""

from typing import Any, cast

import pytest
from semora.engines.plain import react_loop
from semora.orchestrator import MemorySteps, Orchestrator, Step, Suspended
from semora.tools import Concurrent, InvalidToolCall, Stepped

from tests.test_loop import Tools, a_call, says, scripted

SAFE = {"is_concurrency_safe": True}


class Counting(Tools):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.executed: list[str] = []

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        self.executed.append(call_id)
        return {"type": "text", "text": name}


async def test_a_replayed_step_returns_the_recorded_result() -> None:
    log = MemorySteps()
    tools = Counting()
    o = Orchestrator("run-2", log)

    first = await Stepped(tools, o).execute("read", "c1", {})
    again = await Stepped(tools, Orchestrator("run-2", log)).execute("read", "c1", {})

    assert first == again
    assert tools.executed == ["c1"]  # once


async def test_concurrent_outside_stepped_keeps_both_properties() -> None:
    """The documented nesting: `Concurrent` decides who runs together, each of those is a step."""
    log = MemorySteps()
    tools = Counting(defs={"read": SAFE}, names=["read"])
    executor = Concurrent(Stepped(tools, Orchestrator("run-3", log)))

    llm = scripted(says("", a_call("a", "read"), a_call("b", "read")), says("fin"))
    async for _ in react_loop(llm, executor):
        pass

    assert sorted(tools.executed) == ["a", "b"]
    assert (await log.read("run-3", "a")).status == "done"
    assert (await log.read("run-3", "b")).status == "done"


async def test_a_tool_that_raises_commits_the_failure_it_reported() -> None:
    """We watched it fail, so the step is `done`. `Indeterminate` is for what nobody watched."""
    log = MemorySteps()

    class Breaking(Counting):
        async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
            self.executed.append(call_id)
            raise RuntimeError("disk on fire")

    tools = Breaking(defs={"read": {}}, names=["read"])
    stepped = Stepped(tools, Orchestrator("run-4", log))

    assert await stepped.execute("read", "c1", {}) == {
        "type": "error",
        "message": "RuntimeError: disk on fire",
    }
    assert (await log.read("run-4", "c1")).status == "done"
    # Replayed from the record rather than retried: the same id is the same attempt.
    assert await Stepped(tools, Orchestrator("run-4", log)).execute("read", "c1", {}) == {
        "type": "error",
        "message": "RuntimeError: disk on fire",
    }
    assert tools.executed == ["c1"]


async def test_a_suspension_raised_by_a_tool_is_not_reported_as_a_tool_failure() -> None:
    """The one thing the tool boundary must not swallow: a pause nobody would come back to."""
    log = MemorySteps()

    class Asking(Counting):
        async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
            self.executed.append(call_id)
            return cast(dict[str, Any], await Orchestrator("run-5", log).signal("signoff"))

    tools = Asking(defs={"ask": SAFE}, names=["ask"])
    executor = Concurrent(Stepped(tools, Orchestrator("run-5", log)))

    with pytest.raises(Suspended, match="signoff"):
        await executor.execute_batch([{"call_id": "a", "name": "ask", "input": {}}])


async def test_an_unkeyable_round_executes_no_tool_and_records_no_step() -> None:
    """The durable boundary is public, so it checks the ids itself — before `record_pending`.

    An invalid round leaves nothing behind: no effect, and no pending-round entry claiming a round
    that could never be keyed.
    """
    log = MemorySteps()
    tools = Counting()
    calls = [a_call(cast(str, None), "read"), a_call(cast(str, None), "read")]

    with pytest.raises(InvalidToolCall):
        await Orchestrator("unkeyable", log).execute_round(tools, calls, lambda: False)

    assert tools.executed == []
    assert await log.read("unkeyable", "agent:pending-round") == Step("absent")
