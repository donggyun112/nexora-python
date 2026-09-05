"""Input arrives at the model boundary, screened first, committed beside the model call."""

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
from semora import AgentRuntime, ControlPlane, Ingress
from semora.contracts import PendingInput
from semora.controls import Ctx, Halt
from semora_store import MemorySteps, MemoryTranscript


def prompts_seen(messages: list[ModelMessage]) -> list[str]:
    return [
        str(p.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart)
    ]


async def test_a_steer_submitted_mid_run_enters_the_next_model_request() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store, transcript=MemoryTranscript())
    seen: list[list[ModelMessage]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("ping", {}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart("done")])

    agent: Agent[None, str] = Agent(FunctionModel(model))

    @agent.tool_plain
    async def ping() -> str:
        # Another process steers the run while its tool is executing.
        await runtime.submit("run-1", PendingInput("user_steer", UserPromptPart("also b"), "s1"))
        return "pong"

    await runtime.run("run-1", agent, "go")

    assert prompts_seen(seen[-1]) == ["go", "also b"]
    assert [r.status for r in await store.list_inputs("run-1")] == ["admitted"]


async def test_a_steer_arriving_at_the_end_resumes_instead_of_completing() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store, transcript=MemoryTranscript())
    answers: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answers.append(prompts_seen(messages)[-1])
        if len(answers) == 1:
            # Lands while the turn is finishing: the stop is cancelled, not the steer.
            await store.enqueue_input(
                "run-2",
                "late",
                {
                    "kind": "user_steer",
                    "part": {"content": "one more", "part_kind": "user-prompt"},
                    "origin_id": "late",
                },
            )
        return ModelResponse(parts=[TextPart(f"answered {len(answers)}")])

    outcome = await runtime.run("run-2", Agent(FunctionModel(model)), "go")

    assert answers == ["go", "one more"]
    assert outcome.output == "answered 2"


async def test_a_screened_out_input_is_discarded_durably() -> None:
    store = MemorySteps()
    runtime = AgentRuntime(store, transcript=MemoryTranscript())
    await runtime.submit("run-3", PendingInput("user_steer", UserPromptPart("drop me"), "bad"))
    seen: list[list[ModelMessage]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart("done")])

    async def screen(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput] | Halt:
        return [i for i in inputs if i.origin_id != "bad"]

    await runtime.run(
        "run-3", Agent(FunctionModel(model)), "go", controls=ControlPlane(on_inputs=Ingress(screen))
    )

    assert prompts_seen(seen[-1]) == ["go"]
    assert [r.status for r in await store.list_inputs("run-3")] == ["discarded"]
