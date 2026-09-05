"""The transcript is the conversation that ran, chained, rewindable, never deleted."""

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
from semora import AgentRuntime, AgentSuspended, ControlPlane
from semora.controls import Ctx, Permissions, Suspend, ToolDecision
from semora.transcript import (
    TranscriptWriter,
    active_branch,
    encode_message,
    messages_of,
    stripped,
)
from semora_store import MemorySteps, MemoryTranscript


def tool_round() -> Agent[None, str]:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("ping", {}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart("done")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    @agent.tool_plain
    async def ping() -> str:
        return "pong"

    return agent


def same(a: list[ModelMessage], b: list[ModelMessage]) -> bool:
    return [stripped(encode_message(m)) for m in a] == [stripped(encode_message(m)) for m in b]


async def test_a_stored_conversation_reads_back_as_the_conversation_that_ran() -> None:
    transcript = MemoryTranscript()
    runtime = AgentRuntime(MemorySteps(), transcript=transcript)

    outcome = await runtime.run("run-1", tool_round(), "go")

    committed = await runtime.committed_history("run-1")
    assert same(committed, outcome.all_messages())
    kinds = [type(p).__name__ for m in committed for p in m.parts]
    assert kinds == ["UserPromptPart", "ToolCallPart", "ToolReturnPart", "TextPart"]


async def test_entries_chain_through_parent_uuid() -> None:
    transcript = MemoryTranscript()
    await AgentRuntime(MemorySteps(), transcript=transcript).run("run-2", tool_round(), "go")

    branch = active_branch(await transcript.read("run-2"))
    assert branch[0]["parent_uuid"] is None
    assert [e["parent_uuid"] for e in branch[1:]] == [e["uuid"] for e in branch[:-1]]


async def test_a_rewind_shortens_the_branch_without_deleting_anything() -> None:
    transcript = MemoryTranscript()
    writer = TranscriptWriter(transcript, conversation_id="c", branch_id="r")
    first = ModelRequest(parts=[UserPromptPart("one")])
    await writer.record(first)
    await writer.record(ModelResponse(parts=[TextPart("two")]))
    tip_after_one = active_branch(await transcript.read("c"))[0]["uuid"]

    await writer.rewind(tip_after_one)

    entries = await transcript.read("c")
    assert same(messages_of(entries), [first])
    assert sum("parent_uuid" in e for e in entries) == 2, "nothing was deleted"


async def test_a_run_that_never_ends_is_a_row_with_no_ending() -> None:
    transcript = MemoryTranscript()
    runtime = AgentRuntime(MemorySteps(), transcript=transcript)

    async def ask(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        return Suspend({"pending_id": "p1"})

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-4", tool_round(), "go", controls=ControlPlane(pre_tool_use=Permissions(ask))
        )

    record = await transcript.read_branch("run-4")
    assert record is not None and "started_at" in record and "ended_at" not in record
    committed = await runtime.committed_history("run-4")
    assert isinstance(committed[-1].parts[0], ToolCallPart), "the parked round is transcript fact"


async def test_a_crashed_round_leaves_the_request_but_not_the_turn() -> None:
    """Recording the assistant turn before its round ran would make a failed round a fact."""
    transcript = MemoryTranscript()
    runtime = AgentRuntime(MemorySteps(), transcript=transcript)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("boom", {}, tool_call_id="c1")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    class Crash(BaseException):
        """The process goes away; not a tool failure."""

    @agent.tool_plain
    async def boom() -> str:
        raise Crash

    with pytest.raises(Crash):
        await runtime.run("run-5", agent, "go")

    committed = await runtime.committed_history("run-5")
    assert len(committed) == 1 and isinstance(committed[0], ModelRequest)
    assert not any(isinstance(p, ToolReturnPart) for m in committed for p in m.parts)
