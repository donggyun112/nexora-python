"""Answer updates and continuation completion share the attempt's lease."""

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from semora import AgentRuntime, AgentSuspended
from semora.controls import Continue, ControlPlane, Ctx, Permissions, ResumeInput, Suspend
from semora.runtime import ACTIVE_SUSPENSION
from semora_store import Contended, MemorySteps


class PausedWrites(MemorySteps):
    def __init__(self) -> None:
        super().__init__()
        self.pause_state: str | None = None
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()

    async def write_control(
        self, branch_id: str, key: str, value: dict[str, Any], token: int = 0
    ) -> None:
        if key == ACTIVE_SUSPENSION and value.get("state") == self.pause_state:
            self.pause_state = None
            self.entered.set()
            await self.proceed.wait()
        await super().write_control(branch_id, key, value, token)


def setup_agent() -> tuple[Agent[None, str], ControlPlane, list[str]]:
    ran: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(message, ModelResponse) for message in messages):
            return ModelResponse(
                parts=[ToolCallPart("write", {"path": path}, tool_call_id=path) for path in "ab"]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model))

    @agent.tool_plain
    async def write(path: str) -> str:
        ran.append(path)
        return path

    async def gate(ctx: Ctx, call: ToolCallPart) -> Suspend:
        return Suspend({"pending_id": call.tool_call_id})

    return agent, ControlPlane(pre_tool_use=Permissions(gate)), ran


async def test_concurrent_answers_contend_and_retry_without_losing_an_answer() -> None:
    store = PausedWrites()
    runtime = AgentRuntime(store)
    agent, controls, ran = setup_agent()
    with pytest.raises(AgentSuspended):
        await runtime.run("run", agent, "go", controls=controls)
    store.pause_state = "waiting"
    first = asyncio.create_task(
        runtime.resume("run", "a", {"type": "approve"}, agent, controls=controls)
    )
    await asyncio.wait_for(store.entered.wait(), 2)
    try:
        with pytest.raises(Contended):
            await AgentRuntime(store).resume(
                "run", "b", {"type": "approve"}, agent, controls=controls
            )
    finally:
        store.proceed.set()
        results = await asyncio.gather(first, return_exceptions=True)
    assert isinstance(results[0], AgentSuspended)
    active = await store.read("run", ACTIVE_SUSPENSION)
    assert active.value["answers"] == {"a": {"type": "approve"}}
    outcome = await runtime.resume("run", "b", {"type": "approve"}, agent, controls=controls)
    assert outcome.output == "done"
    assert ran == ["a", "b"]
    assert await runtime.pending("run") == []


async def test_resuspension_replaces_the_old_continuation_and_releases_its_lease() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store)
    agent, controls, ran = setup_agent()
    with pytest.raises(AgentSuspended):
        await runtime.run("run", agent, "go", controls=controls)
    with pytest.raises(AgentSuspended):
        await runtime.resume("run", "a", {"type": "approve"}, agent, controls=controls)

    async def revalidate(ctx: Ctx, call: ToolCallPart, answer: ResumeInput) -> Suspend | Continue:
        return Suspend({"pending_id": "a-again"}) if call.tool_call_id == "a" else Continue()

    with pytest.raises(AgentSuspended) as parked:
        await runtime.resume(
            "run", "b", {"type": "approve"}, agent, controls=ControlPlane(on_resume=revalidate)
        )
    assert parked.value.pending == [("a-again", "a")]
    assert await runtime.pending("run") == [("a-again", "a")]
    outcome = await AgentRuntime(store).resume(
        "run", "a-again", {"type": "approve"}, agent, controls=ControlPlane()
    )
    assert outcome.output == "done"
    assert sorted(ran) == ["a", "b"]
    assert await runtime.pending("run") == []


@pytest.mark.parametrize("recover", [False, True])
async def test_finalization_keeps_lease_until_completion_is_persisted(recover: bool) -> None:
    store = PausedWrites()
    runtime = AgentRuntime(store)
    agent, controls, ran = setup_agent()
    with pytest.raises(AgentSuspended):
        await runtime.run("run", agent, "go", controls=controls)
    with pytest.raises(AgentSuspended):
        await runtime.resume("run", "a", {"type": "approve"}, agent, controls=controls)
    if recover:
        active = await store.read("run", ACTIVE_SUSPENSION)
        active.value["answers"]["b"] = {"type": "approve"}
        await store.write_control("run", ACTIVE_SUSPENSION, active.value)
    store.pause_state = "completed"
    attempt = asyncio.create_task(
        runtime.recover("run", agent, [], controls=controls)
        if recover
        else runtime.resume("run", "b", {"type": "approve"}, agent, controls=controls)
    )
    await asyncio.wait_for(store.entered.wait(), 2)
    try:
        with pytest.raises(Contended):
            await AgentRuntime(store).resume(
                "run", "b", {"type": "approve"}, agent, controls=controls
            )
        with pytest.raises(Contended):
            await AgentRuntime(store).run(
                "run", agent, "next prompt", prompt_id="next", controls=controls
            )
    finally:
        store.proceed.set()
        outcomes = await asyncio.gather(attempt, return_exceptions=True)
    assert not isinstance(outcomes[0], BaseException)
    assert not await store.enqueue_input("run", "next", {})
    assert ran == ["a", "b"]
    assert await runtime.pending("run") == []
