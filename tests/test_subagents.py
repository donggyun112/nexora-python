"""Delegation composes: harness `SubAgents` runs a child `Agent` inside the boundary.

The parent records the delegation as one effect; the child runs in a run of its own, over the
same ledger, with its own model and tool steps. A parent recovered after the delegation
committed does not run the child again.
"""

import asyncio
import contextlib

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import SubAgent, SubAgents
from semora import Agent, AgentRuntime, MemorySteps, MemoryTranscript, tool
from semora.dispatch import Recover


def child_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart("read", {"path": "a"}, tool_call_id="child-c1")])
    return ModelResponse(parts=[TextPart("child done")])


class Explorer(Agent):
    """Explores the codebase without modifying anything."""

    llm = FunctionModel(child_model)

    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        self.reads: list[str] = []
        super().__init__(runtime=runtime)

    @tool
    async def read(self, path: str) -> str:
        """Read a file."""
        self.reads.append(path)
        return f"<{path}>"


def lead_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task",
                    {"agent_name": "Explorer", "task": "look around"},
                    tool_call_id="lead-c1",
                )
            ]
        )
    return ModelResponse(parts=[TextPart("lead done")])


async def test_a_delegation_is_one_effect_and_the_child_runs_in_its_own_run() -> None:
    ledger = MemorySteps()
    explorer = Explorer(runtime=AgentRuntime(ledger))  # the child shares the parent's ledger

    class Lead(Agent):
        """Coordinates the explorer."""

        llm = FunctionModel(lead_model)
        store = ledger
        uses = (SubAgents(agents=[SubAgent(explorer)], agent_folders=None),)

    outcome = await Lead().run("go", branch_id="lead-1")

    assert outcome.output == "lead done"
    assert explorer.reads == ["a"]
    delegation = await ledger.read("lead-1", "tool:lead-c1")
    assert delegation.status == "done" and delegation.value == {"ok": True, "value": "child done"}
    child_runs = {run for run, key in ledger._entries if key == "tool:child-c1"}
    assert len(child_runs) == 1 and "lead-1" not in child_runs, "the child has a run of its own"


async def test_a_recovered_parent_does_not_run_the_child_again() -> None:
    ledger, transcript = MemorySteps(), MemoryTranscript()
    explorer = Explorer()
    block = asyncio.Event()

    def slow_lead(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"agent_name": "Explorer", "task": "look around"},
                        tool_call_id="lead-c1",
                    ),
                    ToolCallPart("wait", {}, tool_call_id="lead-c2"),
                ]
            )
        return ModelResponse(parts=[TextPart("lead done")])

    class Lead(Agent):
        """Coordinates the explorer."""

        llm = FunctionModel(slow_lead)
        uses = (SubAgents(agents=[SubAgent(explorer)], agent_folders=None),)

        @tool
        async def wait(self) -> str:
            """Block until the test lets go."""
            await block.wait()
            return "waited"

    lead = Lead(runtime=AgentRuntime(ledger, transcript=transcript))
    worker = asyncio.create_task(lead.run("go", branch_id="lead-2"))
    while (await ledger.read("lead-2", "tool:lead-c2")).status != "running":
        await asyncio.sleep(0)
    worker.cancel()  # the worker dies after the delegation committed, mid second call
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    block.set()

    later = Lead(runtime=AgentRuntime(ledger, transcript=transcript, retry_running=True))
    outcome = await later.dispatch(Recover(), branch_id="lead-2")

    assert outcome.output == "lead done"
    assert explorer.reads == ["a"], "the delegation replayed from the record; the child ran once"
