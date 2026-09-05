"""Policy lands at a seam, a suspension parks the worker, resume re-decides under current rules."""

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
from semora import AgentRuntime, AgentSuspended
from semora.contracts import PendingInput, StopReason
from semora.controls import (
    Continue,
    ControlPlane,
    Ctx,
    Deny,
    FinishPolicy,
    Halt,
    Ingress,
    Journal,
    Permissions,
    Proceed,
    ResumeInput,
    Suspend,
    ToolDecision,
)
from semora_store import MemorySteps


class Files:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def write(self, path: str) -> str:
        self.ran.append(path)
        return f"wrote {path}"


def scripted(*rounds: list[str]) -> tuple[Agent[None, str], list[list[ModelMessage]]]:
    """A model that issues one write per path per round, then answers. Records what it saw."""
    seen: list[list[ModelMessage]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        index = sum(isinstance(m, ModelResponse) for m in messages)
        if index < len(rounds):
            return ModelResponse(
                parts=[
                    ToolCallPart("write", {"path": path}, tool_call_id=f"c{index}{n}")
                    for n, path in enumerate(rounds[index])
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    return Agent(FunctionModel(model)), seen


def ask_for(*paths: str) -> Permissions:
    async def gate(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        if call.args_as_dict()["path"] in paths:
            return Suspend({"pending_id": f"approve-{call.tool_call_id}", "path": paths})
        return Continue()

    return Permissions(gate)


def tool_returns(messages: list[ModelMessage]) -> list[tuple[str, object]]:
    return [
        (p.tool_call_id, p.content)
        for m in messages
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]


async def test_the_gate_suspends_instead_of_blocking_on_an_answer() -> None:
    store, files = MemorySteps(), Files()
    agent, _ = scripted(["stale.md"])
    agent.tool_plain(files.write)

    with pytest.raises(AgentSuspended) as parked:
        await AgentRuntime(store).run(
            "run-1", agent, "delete", controls=ControlPlane(pre_tool_use=ask_for("stale.md"))
        )

    assert parked.value.pending_id == "approve-c00"
    assert parked.value.pending == [("approve-c00", "c00")]
    assert files.ran == [], "the worker did not wait on a person; it parked and left"
    assert (await store.read("run-1", "tool:c00")).status == "absent"


async def test_a_suspension_survives_the_process_and_the_run_continues() -> None:
    store, files = MemorySteps(), Files()
    agent, seen = scripted(["stale.md"])
    agent.tool_plain(files.write)
    controls = ControlPlane(pre_tool_use=ask_for("stale.md"))
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store).run("run-2", agent, "delete", controls=controls)

    # Hours later, another process, a fresh runtime over the same ledger.
    outcome = await AgentRuntime(store).resume(
        "run-2", "approve-c00", {"type": "approve"}, agent, controls=controls
    )

    assert files.ran == ["stale.md"]
    assert outcome.output == "done"
    assert tool_returns(seen[-1]) == [("c00", "wrote stale.md")]


async def test_resume_revalidates_the_latest_policy_before_the_effect() -> None:
    store, files = MemorySteps(), Files()
    agent, seen = scripted(["stale.md"])
    agent.tool_plain(files.write)
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store).run(
            "run-3",
            agent,
            "delete",
            controls=ControlPlane(pre_tool_use=ask_for("stale.md")),
            rules_version="v1",
        )

    versions: list[tuple[str, str]] = []

    async def revoked(ctx: Ctx, call: ToolCallPart, resume: ResumeInput) -> ToolDecision:
        versions.append((resume.suspended_rules_version, resume.current_rules_version))
        return Deny({"type": "error", "message": "deleting was revoked while you decided"})

    outcome = await AgentRuntime(store).resume(
        "run-3",
        "approve-c00",
        {"type": "approve"},
        agent,
        controls=ControlPlane(on_resume=revoked),
        rules_version="v2",
    )

    assert versions == [("v1", "v2")]
    assert files.ran == [], "the approval is an input to policy, never the decision"
    assert outcome.output == "done"
    assert "revoked" in str(tool_returns(seen[-1])[0][1])


async def test_a_denied_human_answer_cannot_be_lifted_by_the_resume_handler() -> None:
    store, files = MemorySteps(), Files()
    agent, seen = scripted(["stale.md"])
    agent.tool_plain(files.write)
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store).run(
            "run-4", agent, "delete", controls=ControlPlane(pre_tool_use=ask_for("stale.md"))
        )
    asked: list[str] = []

    async def lift(ctx: Ctx, call: ToolCallPart, resume: ResumeInput) -> ToolDecision:
        asked.append(call.tool_call_id)
        return Continue()

    outcome = await AgentRuntime(store).resume(
        "run-4",
        "approve-c00",
        {"type": "error", "message": "no"},
        agent,
        controls=ControlPlane(on_resume=lift),
    )

    assert files.ran == [] and asked == []
    assert outcome.output == "done"
    assert tool_returns(seen[-1]) == [("c00", "no")]


async def test_repeating_resume_reuses_the_committed_effect_result() -> None:
    store, files = MemorySteps(), Files()
    agent, _ = scripted(["stale.md"])
    agent.tool_plain(files.write)
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store).run(
            "run-5", agent, "delete", controls=ControlPlane(pre_tool_use=ask_for("stale.md"))
        )
    # The previous resume executed the effect and died before finishing the round.
    await store.start("run-5", "tool:c00")
    await store.finish_effect("run-5", "tool:c00", {"ok": True, "value": "wrote stale.md"})
    asked: list[str] = []

    async def gate(ctx: Ctx, call: ToolCallPart, resume: ResumeInput) -> ToolDecision:
        asked.append(call.tool_call_id)
        return Continue()

    outcome = await AgentRuntime(store).resume(
        "run-5", "approve-c00", {"type": "approve"}, agent, controls=ControlPlane(on_resume=gate)
    )

    assert files.ran == [], "a committed effect is replayed, not repeated"
    assert asked == [], "and policy is not re-asked about an effect that already happened"
    assert outcome.output == "done"


async def test_a_mid_batch_suspension_does_not_run_the_calls_behind_it() -> None:
    store, files = MemorySteps(), Files()
    agent, _ = scripted(["a.md", "stale.md", "c.md"])
    agent.tool_plain(files.write)

    with pytest.raises(AgentSuspended) as parked:
        await AgentRuntime(store).run(
            "run-6", agent, "go", controls=ControlPlane(pre_tool_use=ask_for("stale.md"))
        )

    assert files.ran == ["a.md"], "c.md waits: two writes to one tree differ by order"
    assert parked.value.pending == [("approve-c01", "c01")]
    assert (await store.read("run-6", "tool:c02")).status == "absent"


async def test_every_approval_in_one_round_parks_together_and_the_last_answer_runs_all() -> None:
    store, files = MemorySteps(), Files()
    agent, _ = scripted(["a.md", "b.md"])
    agent.tool_plain(files.write)
    controls = ControlPlane(pre_tool_use=ask_for("a.md", "b.md"))
    runtime = AgentRuntime(store)

    with pytest.raises(AgentSuspended) as parked:
        await runtime.run("run-7", agent, "go", controls=controls)
    assert parked.value.pending == [("approve-c00", "c00"), ("approve-c01", "c01")]

    with pytest.raises(AgentSuspended) as still:
        await runtime.resume("run-7", "approve-c01", {"type": "approve"}, agent, controls=controls)
    assert still.value.pending == [("approve-c00", "c00")]
    assert files.ran == [], "a partial batch answer is recorded but executes nothing"

    outcome = await runtime.resume(
        "run-7", "approve-c00", {"type": "approve"}, agent, controls=controls
    )
    assert files.ran == ["a.md", "b.md"], "the whole batch, in model order"
    assert outcome.output == "done"


async def test_the_gate_can_deny_a_call_without_stopping_the_round() -> None:
    store, files = MemorySteps(), Files()
    agent, seen = scripted(["a.md", "stale.md"])
    agent.tool_plain(files.write)
    journaled: list[str] = []

    async def no_stale(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        if call.args_as_dict()["path"] == "stale.md":
            return Deny("policy: stale.md is protected")
        return Continue()

    async def journal(ctx: Ctx, call: ToolCallPart, result: object) -> None:
        journaled.append(call.tool_call_id)

    outcome = await AgentRuntime(store).run(
        "run-8",
        agent,
        "go",
        controls=ControlPlane(pre_tool_use=Permissions(no_stale), post_tool_use=Journal(journal)),
    )

    assert files.ran == ["a.md"]
    assert tool_returns(seen[-1]) == [
        ("c00", "wrote a.md"),
        ("c01", "policy: stale.md is protected"),
    ]
    assert journaled == ["c00"], "a policy result stands in for an effect; it is not journaled"
    assert outcome.output == "done"


async def test_post_tool_use_crosses_its_durable_boundary_once_per_call() -> None:
    store, files = MemorySteps(), Files()
    journaled: list[str] = []

    async def journal(ctx: Ctx, call: ToolCallPart, result: object) -> None:
        journaled.append(call.tool_call_id)

    controls = ControlPlane(post_tool_use=Journal(journal))
    agent, _ = scripted(["a.md", "b.md"])
    agent.tool_plain(files.write)
    # The dead worker committed c00 and journaled it; c01 never started.
    await store.start("run-9", "tool:c00")
    await store.finish_effect("run-9", "tool:c00", {"ok": True, "value": "wrote a.md"})
    await store.write_control(
        "run-9", "after:c00", {"hooked": True, "record": {"ok": True, "value": "wrote a.md"}}
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(
            parts=[
                ToolCallPart("write", {"path": "a.md"}, tool_call_id="c00"),
                ToolCallPart("write", {"path": "b.md"}, tool_call_id="c01"),
            ]
        ),
    ]

    await AgentRuntime(store).recover("run-9", agent, history, controls=controls)

    assert files.ran == ["b.md"]
    assert journaled == ["c01"], "the replayed result already crossed the journal"


async def test_a_halting_screen_ends_the_run_before_the_model_is_called() -> None:
    agent, seen = scripted(["a.md"])

    async def refuse(ctx: Ctx, inputs: list[PendingInput]) -> Halt:
        return Halt("policy")

    outcome = await AgentRuntime(MemorySteps()).run(
        "run-10", agent, "go", controls=ControlPlane(on_inputs=Ingress(refuse))
    )

    assert seen == []
    assert outcome.stop_reason == "policy"


async def test_an_ingress_screen_masks_an_input_before_the_model_sees_it() -> None:
    agent, seen = scripted()

    async def mask(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput]:
        return [PendingInput(i.kind, UserPromptPart("[redacted]")) for i in inputs]

    await AgentRuntime(MemorySteps()).run(
        "run-11", agent, "my card is 4111", controls=ControlPlane(on_inputs=Ingress(mask))
    )

    first = seen[0][0].parts[0]
    assert isinstance(first, UserPromptPart) and first.content == "[redacted]"


async def test_a_verifier_vetoes_the_finish_and_the_run_goes_around_again() -> None:
    agent, seen = scripted()
    vetoed: list[str] = []

    async def cite(ctx: Ctx, reason: StopReason) -> Proceed | Halt:
        if vetoed:
            return Halt(reason)
        vetoed.append(ctx.text)
        return Proceed([UserPromptPart("not done: cite a source")])

    outcome = await AgentRuntime(MemorySteps()).run(
        "run-12", agent, "go", controls=ControlPlane(before_finish=FinishPolicy(cite))
    )

    assert vetoed == ["done"]
    assert len(seen) == 2 and seen[1][-1].parts[-1].content == "not done: cite a source"  # type: ignore[union-attr]
    assert outcome.stop_reason == "completed"
