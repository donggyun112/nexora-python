"""A tool call is a step. What that does to a run that resumes."""

from typing import Any

from nexora.engines.plain import react_loop
from nexora.orchestrator import MemorySteps, Orchestrator
from nexora.tools import Concurrent, Stepped
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
