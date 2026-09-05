"""One state-aware entry point: given this run's durable state, may this command arrive now?"""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from semora import AgentRuntime, AgentSuspended, ControlPlane
from semora.controls import Continue, Ctx, Permissions, Suspend, ToolDecision
from semora.dispatch import (
    Answer,
    CommandRouter,
    InvalidTransition,
    Prompt,
    Recover,
    ResumeApproval,
    StartRun,
)
from semora_store import Contended, MemorySteps, MemoryTranscript


class Files:
    def __init__(self) -> None:
        self.ran: list[str] = []
        self.block: asyncio.Event | None = None

    async def write(self, path: str) -> str:
        self.ran.append(path)
        if self.block is not None and path == "b.md":
            await self.block.wait()
        return f"wrote {path}"


def scripted(*rounds: list[str]) -> tuple[Agent[None, str], list[int]]:
    """One write per path per round, then "done". Counts live model calls."""
    consulted: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        consulted.append(len(messages))
        index = sum(isinstance(m, ModelResponse) for m in messages)
        if index < len(rounds):
            return ModelResponse(
                parts=[
                    ToolCallPart("write", {"path": path}, tool_call_id=f"c{index}{n}")
                    for n, path in enumerate(rounds[index])
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    return Agent(FunctionModel(model)), consulted


def ask_for(*paths: str) -> ControlPlane:
    async def gate(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        if call.args_as_dict()["path"] in paths:
            return Suspend({"pending_id": f"approve-{call.tool_call_id}"})
        return Continue()

    return ControlPlane(pre_tool_use=Permissions(gate))


def durable() -> AgentRuntime:
    return AgentRuntime(MemorySteps(), transcript=MemoryTranscript())


async def test_dispatch_requires_the_durable_collaborators() -> None:
    agent, _ = scripted()
    with pytest.raises(TypeError):
        await AgentRuntime(MemorySteps()).dispatch("run-1", agent, Prompt("hi"))


async def test_a_prompt_starts_an_idle_run() -> None:
    runtime, files = durable(), Files()
    agent, _ = scripted(["a.md"])
    agent.tool_plain(files.write)

    outcome = await runtime.dispatch("run-2", agent, Prompt("go"))

    assert outcome.output == "done" and files.ran == ["a.md"]
    assert await runtime.state("run-2") == "completed"


async def test_an_answer_resumes_the_parked_call() -> None:
    runtime, files = durable(), Files()
    agent, _ = scripted(["stale.md"])
    agent.tool_plain(files.write)
    controls = ask_for("stale.md")
    with pytest.raises(AgentSuspended) as parked:
        await runtime.dispatch("run-3", agent, Prompt("delete"), controls=controls)
    assert await runtime.state("run-3") == "waiting"

    outcome = await runtime.dispatch(
        "run-3", agent, Answer(parked.value.pending_id, {"type": "approve"}), controls=controls
    )

    assert files.ran == ["stale.md"] and outcome.output == "done"
    assert await runtime.state("run-3") == "completed"


async def test_an_answer_with_no_park_is_an_invalid_transition() -> None:
    runtime = durable()
    agent, _ = scripted()
    with pytest.raises(InvalidTransition) as refused:
        await runtime.dispatch("run-4", agent, Answer("approve-c00", {"type": "approve"}))
    assert refused.value.state == "fresh"


async def test_recover_finishes_an_interrupted_round_without_paying_for_the_turn_again() -> None:
    store, transcript, files = MemorySteps(), MemoryTranscript(), Files()
    files.block = asyncio.Event()
    agent, consulted = scripted(["a.md", "b.md"])
    agent.tool_plain(files.write)
    worker = asyncio.create_task(
        AgentRuntime(store, transcript=transcript).dispatch("run-5", agent, Prompt("go"))
    )
    while files.ran != ["a.md", "b.md"]:
        await asyncio.sleep(0)
    worker.cancel()  # the worker dies with b.md mid-flight
    with pytest.raises(asyncio.CancelledError):
        await worker
    files.block.set()
    files.ran.clear()
    consulted.clear()

    runtime = AgentRuntime(store, transcript=transcript, retry_running=True)
    assert await runtime.state("run-5") == "interrupted"
    outcome = await runtime.dispatch("run-5", agent, Recover())

    assert files.ran == ["b.md"], "a.md committed before the crash and replays from the record"
    assert consulted == [3], (
        "the interrupted model turn replays from the journal; only the answer is live"
    )
    assert outcome.output == "done"
    assert await runtime.state("run-5") == "completed"


async def test_recover_with_nothing_to_recover_is_an_invalid_transition() -> None:
    runtime = durable()
    agent, _ = scripted()
    with pytest.raises(InvalidTransition) as refused:
        await runtime.dispatch("run-6", agent, Recover())
    assert refused.value.state == "fresh"


async def test_a_prompt_behind_a_live_worker_is_enqueued() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store, transcript=MemoryTranscript())
    agent, _ = scripted()
    assert await store.acquire("run-7", "another-worker", 60)

    result = await runtime.dispatch("run-7", agent, Prompt("also check the tests", "p1"))

    assert result == {"type": "enqueued", "input_id": "p1"}
    assert [r.input_id for r in await store.list_inputs("run-7")] == ["p1"]


async def test_a_repeated_prompt_id_is_delivered_once() -> None:
    runtime = durable()
    agent, consulted = scripted()

    await runtime.dispatch("run-8", agent, Prompt("hello", "p1"))
    await runtime.dispatch("run-8", agent, Prompt("hello", "p1"))

    assert consulted == [1], "the second dispatch carried no new input"


async def test_removing_queue_steer_makes_contention_the_hosts_problem() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store, transcript=MemoryTranscript())
    agent, _ = scripted()
    assert await store.acquire("run-9", "another-worker", 60)

    with pytest.raises(Contended):
        await CommandRouter(ResumeApproval(), StartRun()).dispatch(
            runtime, "run-9", agent, Prompt("go")
        )
