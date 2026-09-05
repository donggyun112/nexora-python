"""A worker dies mid-round. Recovery finishes the round without asking the model again.

The ledger keys every call by its `tool_call_id`, so recovery tells three cases apart: `done`
results are restored, `absent` calls run, and a call that started and never finished is
`Indeterminate` unless the caller says repeating that effect is safe.
"""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from semora import AgentRuntime
from semora_store import Indeterminate, MemorySteps


class Files:
    def __init__(self) -> None:
        self.ran: list[str] = []
        self.contents: dict[str, str] = {}
        self.block: asyncio.Event | None = None

    async def write(self, path: str, text: str) -> str:
        self.ran.append(path)
        if self.block is not None and path == "b.md":
            await self.block.wait()
        self.contents[path] = text
        return f"wrote {path}"


def never_asked_twice() -> tuple[Agent[None, str], list[int]]:
    """A model that scripts one tool round, then answers. Counts how often it is consulted."""
    consulted: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        consulted.append(len(messages))
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("write", {"path": "a.md", "text": "one"}, tool_call_id="c1"),
                    ToolCallPart("write", {"path": "b.md", "text": "two"}, tool_call_id="c2"),
                ]
            )
        return ModelResponse(parts=[TextPart("both written")])

    return Agent(FunctionModel(model)), consulted


def dead_workers_transcript() -> list[ModelMessage]:
    """What the dead worker had committed: the prompt and the model's two calls."""
    return [
        ModelRequest(parts=[UserPromptPart("write both files")]),
        ModelResponse(
            parts=[
                ToolCallPart("write", {"path": "a.md", "text": "one"}, tool_call_id="c1"),
                ToolCallPart("write", {"path": "b.md", "text": "two"}, tool_call_id="c2"),
            ]
        ),
    ]


async def test_committed_call_replays_and_absent_call_runs() -> None:
    store = MemorySteps()
    files = Files()
    agent, consulted = never_asked_twice()
    agent.tool_plain(files.write)
    # c1 committed before the crash; c2 never started.
    await store.start("run-4", "tool:c1")
    await store.finish_effect("run-4", "tool:c1", {"ok": True, "value": "wrote a.md"})

    outcome = await AgentRuntime(execution_store=store).recover(
        "run-4", agent, dead_workers_transcript()
    )

    assert files.ran == ["b.md"], "the committed write must not run a second time"
    assert outcome.output == "both written"
    assert consulted == [3], "the model was consulted once, for the answer, with both results"


async def test_started_but_unreported_call_is_indeterminate() -> None:
    store = MemorySteps()
    files = Files()
    agent, _ = never_asked_twice()
    agent.tool_plain(files.write)
    await store.start("run-5", "tool:c1")
    await store.finish_effect("run-5", "tool:c1", {"ok": True, "value": "wrote a.md"})
    await store.start("run-5", "tool:c2")  # intent recorded, effect never reported

    with pytest.raises(Indeterminate):
        await AgentRuntime(execution_store=store).recover("run-5", agent, dead_workers_transcript())
    assert files.ran == [], "an indeterminate effect is not guessed at"

    outcome = await AgentRuntime(execution_store=store, retry_running=True).recover(
        "run-5", agent, dead_workers_transcript()
    )
    assert files.ran == ["b.md"], "only the caller may say the effect is safe to repeat"
    assert outcome.output == "both written"


async def test_a_cancelled_worker_leaves_its_running_step_as_intent() -> None:
    store = MemorySteps()
    files = Files()
    files.block = asyncio.Event()
    agent, _ = never_asked_twice()
    agent.tool_plain(files.write)
    runtime = AgentRuntime(execution_store=store)

    task = asyncio.create_task(runtime.run("run-6", agent, "write both files"))
    while files.ran != ["a.md", "b.md"]:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await store.read("run-6", "tool:c1")).status == "done"
    assert (await store.read("run-6", "tool:c2")).status == "running"


async def test_a_tool_that_raises_failed_but_the_round_did_not() -> None:
    store = MemorySteps()
    agent, consulted = never_asked_twice()

    @agent.tool_plain
    async def write(path: str, text: str) -> str:
        if path == "b.md":
            raise OSError("disk full")
        return f"wrote {path}"

    outcome = await AgentRuntime(execution_store=store).run("run-7", agent, "write both files")

    assert outcome.output == "both written"
    assert consulted == [1, 3], "the model saw both results and answered"
    step = await store.read("run-7", "tool:c2")
    assert step.status == "done" and step.value == {"ok": False, "error": "disk full"}


async def test_writes_run_in_call_order_unless_declared_concurrency_safe() -> None:
    order: list[str] = []
    started = asyncio.Event()

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("slow", {}, tool_call_id="c1"),
                    ToolCallPart("fast", {}, tool_call_id="c2"),
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    @agent.tool_plain
    async def slow() -> str:
        started.set()
        await asyncio.sleep(0.01)
        order.append("slow")
        return "slow"

    @agent.tool_plain
    async def fast() -> str:
        order.append("fast")
        return "fast"

    await AgentRuntime(execution_store=MemorySteps()).run("run-8", agent, "go")
    assert order == ["slow", "fast"], "two writes to one file differ by order; the batch is serial"
