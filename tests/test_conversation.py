"""One conversation holds many runs.

A run is the execution coordinate; the conversation is what the model remembers.
"""

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
from semora import (
    AgentRuntime,
    AgentSuspended,
    ControlPlane,
    MemorySteps,
    MemoryTranscript,
)
from semora.controls import Ctx, Permissions, Suspend, ToolDecision


def prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        str(p.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart)
    ]


async def test_conversation_history_is_independent_from_run_identity() -> None:
    transcript = MemoryTranscript()
    runtime = AgentRuntime(MemorySteps(), transcript=transcript)
    seen: list[list[str]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(prompts(messages))
        return ModelResponse(parts=[TextPart(f"answer {len(seen)}")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    await runtime.run("run-1", agent, "first", conversation_id="chat-9")
    await runtime.run("run-2", agent, "second", conversation_id="chat-9")

    assert seen[1] == ["first", "second"], "the second run continues the first run's conversation"
    assert await runtime.state("run-1") == "completed"
    assert await runtime.state("run-2") == "completed"
    assert prompts(await runtime.committed_history("run-2", "chat-9")) == ["first", "second"]
    assert await transcript.read("run-2") == [], "nothing is keyed by the run id alone"


async def test_a_resumed_run_stays_in_its_conversation() -> None:
    transcript = MemoryTranscript()
    runtime = AgentRuntime(MemorySteps(), transcript=transcript)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("ping", {}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart("done")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    @agent.tool_plain
    async def ping() -> str:
        return "pong"

    async def ask(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        return Suspend({"pending_id": "p1"})

    controls = ControlPlane(pre_tool_use=Permissions(ask))
    with pytest.raises(AgentSuspended):
        await runtime.run("run-3", agent, "go", conversation_id="chat-9", controls=controls)

    outcome = await runtime.resume("run-3", "p1", {"type": "approve"}, agent, controls=controls)

    assert outcome.output == "done"
    kinds = [
        type(p).__name__
        for m in await runtime.committed_history("run-3", "chat-9")
        for p in m.parts
    ]
    assert kinds[-1] == "TextPart" and "ToolReturnPart" in kinds


async def test_a_parked_round_may_not_reuse_a_pending_id() -> None:
    runtime = AgentRuntime(MemorySteps())

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("ping", {}, tool_call_id="c1"),
                    ToolCallPart("ping", {}, tool_call_id="c2"),
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    @agent.tool_plain
    async def ping() -> str:
        return "pong"

    async def same_id(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        return Suspend({"pending_id": "approve"})

    with pytest.raises(ValueError, match="reuses a pending_id"):
        await runtime.run("run-4", agent, "go", controls=ControlPlane(pre_tool_use=same_id))
    assert (await runtime.store.read("run-4", "agent:active-suspension")).status == "absent"  # type: ignore[union-attr]
