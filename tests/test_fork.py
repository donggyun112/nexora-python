"""A run branches from one entry of another run's transcript.

What the source finished before that entry replays in the branch, ungated; what it had not yet
done runs again under the branch's own policy. The source run is never written.
"""

from typing import Any

import pytest
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from semora import AgentRuntime, ControlPlane, ExecutionContext, MemorySteps, MemoryTranscript
from semora.controls import Continue, Ctx, Deny, Suspend, ToolDecision
from semora_store import Indeterminate
from test_recovery import Files, dead_workers_transcript, never_asked_twice


async def ask_everything(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
    return Suspend({"pending_id": f"approve-{call.tool_call_id}"})


async def source_run(
    store: MemorySteps, transcript: MemoryTranscript, files: Files
) -> list[dict[str, Any]]:
    """One completed run and its transcript entries, in order."""
    agent, _ = never_asked_twice()
    agent.tool_plain(files.write)
    outcome = await AgentRuntime(store, transcript=transcript).run("run-a", agent, "write both")
    assert outcome.output == "both written" and files.ran == ["a.md", "b.md"]
    return await transcript.for_execution(ExecutionContext(run_id="run-a")).read("run-a")


def entry_of(entries: list[dict[str, Any]], kind: str) -> str:
    """The uuid of the first message entry of one kind."""
    return next(str(e["uuid"]) for e in entries if e.get("message", {}).get("kind") == kind)


async def test_a_fork_after_the_round_replays_its_effects_without_gating_them() -> None:
    store, transcript, files = MemorySteps(), MemoryTranscript(), Files()
    entries = await source_run(store, transcript, files)
    agent, consulted = never_asked_twice()
    agent.tool_plain(files.write)

    outcome = await AgentRuntime(store, transcript=transcript).fork(
        "run-a",
        entry_of(entries, "response"),
        "run-b",
        agent,
        controls=ControlPlane(pre_tool_use=ask_everything),
    )

    assert outcome.stop_reason == "completed", "nothing parked: the effects had already happened"
    assert files.ran == ["a.md", "b.md"], "neither write ran again"
    assert consulted == [3], "the branch paid for one model call, the answer"
    assert (await store.read("run-b", "tool:c1")).value == {"ok": True, "value": "wrote a.md"}
    assert (await store.read("run-b", "after:c1")).status == "done", "journaled under the branch"
    assert (await store.read("run-a", "after:c1")).status == "absent", "not on the source"


async def test_regate_asks_the_branch_policy_and_replays_only_what_it_allows() -> None:
    """A changed policy gets its say on copied effects; a Continue replays, a Deny does not."""
    store, transcript, files = MemorySteps(), MemoryTranscript(), Files()
    entries = await source_run(store, transcript, files)
    agent, _ = never_asked_twice()
    agent.tool_plain(files.write)
    asked: list[str] = []

    async def freeze_b(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        asked.append(call.tool_call_id)
        if call.args_as_dict()["path"] == "b.md":
            return Deny("policy: b.md is frozen")
        return Continue()

    outcome = await AgentRuntime(store, transcript=transcript).fork(
        "run-a",
        entry_of(entries, "response"),
        "run-d",
        agent,
        regate=True,
        controls=ControlPlane(pre_tool_use=freeze_b),
    )

    assert asked == ["c1", "c2"], "both copied records went through the branch's gate"
    assert files.ran == ["a.md", "b.md"], "nothing ran again either way"
    returns = {
        part.tool_call_id: part.content
        for message in outcome.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert returns == {"c1": "wrote a.md", "c2": "policy: b.md is frozen"}


async def test_a_fork_inherits_the_sources_doubt_about_an_unreported_effect() -> None:
    """A call the source started and never finished is not guessed at by the branch either."""
    store, files = MemorySteps(), Files()
    await store.start("run-a", "tool:c1")
    await store.finish_effect("run-a", "tool:c1", {"ok": True, "value": "wrote a.md"})
    await store.start("run-a", "tool:c2")  # the source's worker died here
    agent, _ = never_asked_twice()
    agent.tool_plain(files.write)

    with pytest.raises(Indeterminate):
        await AgentRuntime(store).fork(
            "run-a", None, "run-e", agent, history=dead_workers_transcript()
        )
    assert files.ran == []

    outcome = await AgentRuntime(store, retry_running=True).fork(
        "run-a", None, "run-f", agent, history=dead_workers_transcript()
    )
    assert files.ran == ["b.md"] and outcome.output == "both written"


async def test_a_fork_before_the_round_runs_it_again_as_the_branch() -> None:
    store, transcript, files = MemorySteps(), MemoryTranscript(), Files()
    entries = await source_run(store, transcript, files)
    agent, consulted = never_asked_twice()
    agent.tool_plain(files.write)

    outcome = await AgentRuntime(store, transcript=transcript).fork(
        "run-a", entry_of(entries, "request"), "run-c", agent
    )

    assert outcome.output == "both written"
    assert consulted == [1, 3], "the branch asked the model for the round, then for the answer"
    assert files.ran == ["a.md", "b.md", "a.md", "b.md"], "the branch's own effects, run once"
    assert len(await transcript.for_execution(ExecutionContext(run_id="run-c")).read("run-c")) > 0
