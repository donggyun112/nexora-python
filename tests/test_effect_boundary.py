"""Recovery must retain runtime signals and the model-visible policy projection."""

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from semora import AgentRuntime, ControlPlane, ControlSignal, Ctx, MemorySteps


def history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart("read")]),
        ModelResponse(parts=[ToolCallPart("read", {}, tool_call_id="c1")]),
    ]


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("done")])


async def test_runtime_signal_leaves_an_unreported_effect() -> None:
    store = MemorySteps()
    signal = ControlSignal("worker stopped")

    async def read() -> str:
        raise signal

    agent = Agent(FunctionModel(answer), tools=[read])
    with pytest.raises(ControlSignal) as raised:
        await AgentRuntime(store).recover("signal", agent, history())

    assert raised.value is signal
    assert (await store.read("signal", "tool:c1")).status == "running"


@pytest.mark.parametrize("keep_controls", [True, False])
async def test_recovery_replays_redacted_result_without_repeating_journal(
    keep_controls: bool,
) -> None:
    store = MemorySteps()
    called: list[str] = []
    journaled: list[str] = []

    async def read() -> dict[str, str]:
        called.append("read")
        return {"text": "123-45-6789"}

    async def redact(ctx: Ctx, call: ToolCallPart, result: Any) -> None:
        journaled.append(call.tool_call_id)
        result["text"] = "***"

    controls = ControlPlane(post_tool_use=redact)
    agent = Agent(FunctionModel(answer), tools=[read])
    runtime = AgentRuntime(store)
    first = await runtime.recover("redaction", agent, history(), controls=controls)
    resumed = await runtime.recover(
        "redaction", agent, history(), controls=controls if keep_controls else None
    )

    for outcome in (first, resumed):
        results = [
            part.content
            for message in outcome.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert results == [{"text": "***"}]
    assert called == ["read"]
    assert journaled == ["c1"]
    assert (await store.read("redaction", "tool:c1")).value == {
        "ok": True,
        "value": {"text": "123-45-6789"},
    }
