"""Human-in-the-loop, end to end: stop for a person, survive the process, continue.

The window here is days. Everything in this file exists because of that: the snapshot has to
outlive the worker, and the decision has to be re-derived rather than replayed.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from semora import AgentRuntime
from semora.contracts import BaseMessage, EventType, ToolCall
from semora.controls import (
    Continue,
    ControlPlane,
    Ctx,
    Deny,
    Permissions,
    ResumeInput,
    Suspend,
    gate,
)
from semora.history import decode_continuation, encode_continuation
from semora.orchestrator import AgentSuspended, InvalidSuspension, MemorySteps, Orchestrator
from semora_permissions import PolicyContext, Rule, resolve_rules

from tests.test_loop import Tools, a_call, says, scripted

DEPLOY = "deploy"


async def test_a_mid_batch_suspension_ends_the_round_and_drops_the_calls_behind_it() -> None:
    """Pins the current contract: a suspension ends the round, it does not pause it.

    `execute_calls` promises no synthesized results for unexecuted calls, and
    `suspend_history_snapshot` prunes them from the assistant turn so every surviving call has an
    answer. The cost is here in one place: `notify` was requested, never gated, never executed, and
    is absent from the resumed history — the model has to ask again. Change this test only when
    that policy changes.
    """
    log = MemorySteps()
    runtime = AgentRuntime(store=log)
    tools = Tools(names=["read", DEPLOY, "notify"])

    async def ask_about_deploy(call: ToolCall) -> dict[str, Any] | None:
        if call["name"] == DEPLOY:
            return {"type": "suspend", "pending_id": "approval-mid"}
        return None

    controls = ControlPlane(pre_tool_use=Permissions(gate(ask_about_deploy)))
    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-mid-batch",
            scripted(says("", a_call("c1", "read"), a_call("c2", DEPLOY), a_call("c3", "notify"))),
            tools,
            "ship it and tell them",
            controls=controls,
            rules_version="v1",
        )

    assert tools.ran == ["read"]  # the round stopped at the gate, it did not run past it

    resumed = Tools(names=["read", DEPLOY, "notify"])
    model = scripted(says("done"))
    await runtime.resume(
        "run-mid-batch",
        "approval-mid",
        {"type": "text", "text": "approved"},
        model,
        resumed,
        controls=controls,
    )

    assert resumed.ran == [DEPLOY]  # only the parked call finishes; nothing continues behind it
    assistant = next(m for m in model.seen[0] if isinstance(m, AIMessage))
    assert [c["id"] for c in assistant.tool_calls] == ["c1", "c2"]  # c3 pruned, so nothing dangles
    assert [m.tool_call_id for m in model.seen[0] if isinstance(m, ToolMessage)] == ["c1", "c2"]


async def _ask_deploy(call: ToolCall) -> dict[str, Any] | None:
    """Suspend every deploy with a pending id derived from the call, allow everything else."""
    if call["name"] == DEPLOY:
        return {"type": "suspend", "pending_id": f"approve-{call['id']}"}
    return None


async def _parked_batch(log: MemorySteps, run_id: str) -> ControlPlane:
    """Park c2 and c3 (both deploys) out of a four-call round; returns the shared controls."""
    controls = ControlPlane(pre_tool_use=Permissions(gate(_ask_deploy)))
    tools = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=log).run(
            run_id,
            scripted(
                says(
                    "",
                    a_call("c1", "read"),
                    a_call("c2", DEPLOY),
                    a_call("c3", DEPLOY),
                    a_call("c4", "notify"),
                )
            ),
            tools,
            "ship both and tell them",
            controls=controls,
            rules_version="v1",
        )
    assert tools.ran == ["read"]  # setup invariant: nothing past the first gate executed
    return controls


async def test_every_approval_in_one_round_parks_together_and_surfaces_at_once() -> None:
    """One human round-trip for N gated calls.

    The gate keeps collecting suspends past the first one, and `AgentSuspended.pending`
    carries every request from the single park.
    """
    log = MemorySteps()
    tools = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(AgentSuspended) as parked:
        await AgentRuntime(store=log).run(
            "run-batch",
            scripted(
                says(
                    "",
                    a_call("c1", "read"),
                    a_call("c2", DEPLOY),
                    a_call("c3", DEPLOY),
                    a_call("c4", "notify"),
                )
            ),
            tools,
            "ship both",
            controls=ControlPlane(pre_tool_use=Permissions(gate(_ask_deploy))),
            rules_version="v1",
        )

    assert parked.value.pending == [("approve-c2", "c2"), ("approve-c3", "c3")]
    # the first parked call stays the legacy single-suspension identity
    assert (parked.value.pending_id, parked.value.tool_call_id) == ("approve-c2", "c2")
    assert tools.ran == ["read"]  # execution still stops at the first gate


async def test_a_parked_round_rejects_duplicate_pending_ids_before_persistence() -> None:
    """Every external answer identity in one parked round must route to exactly one call."""

    async def same_pending_id(call: ToolCall) -> dict[str, Any] | None:
        if call["name"] == DEPLOY:
            return {"type": "suspend", "pending_id": "approval-mid"}
        return None

    log = MemorySteps()
    controls = ControlPlane(pre_tool_use=Permissions(gate(same_pending_id)))
    calls = [a_call("c1", DEPLOY), a_call("c2", DEPLOY)]

    with pytest.raises(InvalidSuspension, match="appears twice"):
        await AgentRuntime(store=log).run(
            "duplicate-pending",
            scripted(says("", *calls)),
            Tools(names=[DEPLOY]),
            "ship both",
            controls=controls,
        )

    assert await Orchestrator("duplicate-pending", log).active_continuation() is None


async def test_a_partial_batch_answer_is_recorded_but_executes_nothing() -> None:
    """Contract: partial answers are inputs, not effects.

    The run stays parked, re-raising `AgentSuspended` with the still-undecided requests,
    and no tool runs.
    """
    log = MemorySteps()
    controls = await _parked_batch(log, "run-partial")

    second = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(AgentSuspended) as still:
        await AgentRuntime(store=log).resume(
            "run-partial",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            second,
            controls=controls,
        )

    assert still.value.pending == [("approve-c3", "c3")]
    assert second.ran == []  # incremental execute-while-others-wait is forbidden


async def test_the_last_answer_executes_the_whole_batch_in_model_order() -> None:
    """Only a fully decided batch executes.

    Both approved effects run in model call order, in a different process, and every
    surviving call in history gets an answer.
    """
    log = MemorySteps()
    controls = await _parked_batch(log, "run-decided")
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=log).resume(
            "run-decided",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
            controls=controls,
        )

    # ── a different process answers the last request ──
    final = Tools(names=["read", DEPLOY, "notify"])
    model = scripted(says("done"))
    outcome = await AgentRuntime(store=log).resume(
        "run-decided",
        "approve-c3",
        {"type": "text", "text": "approved"},
        model,
        final,
        controls=controls,
    )

    assert final.ran == [DEPLOY, DEPLOY]
    assert outcome["content"] == "done"
    assistant = next(m for m in model.seen[0] if isinstance(m, AIMessage))
    assert [c["id"] for c in assistant.tool_calls] == ["c1", "c2", "c3"]  # c4 pruned
    answered = [m.tool_call_id for m in model.seen[0] if isinstance(m, ToolMessage)]
    assert answered == ["c1", "c2", "c3"]  # nothing dangles, nothing doubled


async def test_recover_exposes_only_the_still_undecided_pending_ids() -> None:
    """Contract: recovery consults both the parked set and its durable answers.

    Re-gating parked calls would reissue pending ids and orphan answers already in flight —
    recover must re-raise only the original requests that still need an answer.
    """
    log = MemorySteps()
    controls = await _parked_batch(log, "run-recover")
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=log).resume(
            "run-recover",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
            controls=controls,
        )

    waiting = decode_continuation(await Orchestrator("run-recover", log).suspension("c2"))
    assert waiting is not None
    fresh = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(AgentSuspended) as parked:
        await AgentRuntime(store=log).recover(
            "run-recover",
            list(waiting.messages),
            scripted(says("unused")),
            fresh,
            controls=controls,
        )

    assert parked.value.pending == [("approve-c3", "c3")]
    assert fresh.ran == []  # a parked call is never "absent"

    # and the partial answer recorded before the crash still counts
    final = Tools(names=["read", DEPLOY, "notify"])
    await AgentRuntime(store=log).resume(
        "run-recover",
        "approve-c3",
        {"type": "text", "text": "approved"},
        scripted(says("done")),
        final,
        controls=controls,
    )
    assert final.ran == [DEPLOY, DEPLOY]


async def test_recovery_collects_every_approval_from_the_interrupted_round() -> None:
    """Recovery must re-enter the complete gate phase, not stop after its first suspension."""
    log = MemorySteps()
    calls = [a_call("c1", DEPLOY), a_call("c2", DEPLOY)]
    history: list[BaseMessage] = [AIMessage(content="", tool_calls=calls)]
    controls = ControlPlane(pre_tool_use=Permissions(gate(_ask_deploy)))

    async with Orchestrator("run-recover-gates", log) as owner:
        await owner.record_pending(calls, 0)

    with pytest.raises(AgentSuspended) as parked:
        async with Orchestrator("run-recover-gates", log) as owner:
            await owner.recover_pending(
                history,
                Tools(names=[DEPLOY]),
                controls=controls,
            )

    assert parked.value.pending == [("approve-c1", "c1"), ("approve-c2", "c2")]


async def test_a_denied_answer_in_a_batch_refuses_only_that_call() -> None:
    """One deny does not poison the batch.

    The approved call runs, the denied one becomes an error answer the model can read.
    """
    log = MemorySteps()
    controls = await _parked_batch(log, "run-mixed")
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=log).resume(
            "run-mixed",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
            controls=controls,
        )

    final = Tools(names=["read", DEPLOY, "notify"])
    model = scripted(says("done"))
    await AgentRuntime(store=log).resume(
        "run-mixed",
        "approve-c3",
        {"type": "error", "message": "denied by reviewer"},
        model,
        final,
        controls=controls,
    )

    assert final.ran == [DEPLOY]  # only the approved effect executed
    by_id = {m.tool_call_id: m for m in model.seen[0] if isinstance(m, ToolMessage)}
    assert by_id["c3"].status == "error"
    assert by_id["c2"].status == "success"


async def test_compaction_accepts_a_sibling_and_keeps_one_indexed_batch_snapshot() -> None:
    """`compact_suspension` is a field-preserving rewrite, not a reset.

    The active record must keep `call_ids`/`answers` (dropping answers loses a person's
    decision), and every parked call's suspend key resolves through the one active snapshot.
    """
    log = MemorySteps()
    await _parked_batch(log, "run-compact")
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=log).resume(
            "run-compact",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
        )

    async with Orchestrator("run-compact", log) as o:
        await o.compact_suspension("c3", conversation_id="run-compact", leaf_uuid=None)
        active = await o.active_continuation()
        assert active is not None
        assert active["answers"] == {"c2": {"type": "text", "text": "approved"}}
        assert active["call_ids"] == ["c2", "c3"]
        for call_id in ("c2", "c3"):
            raw = await log.read("run-compact", f"suspend:{call_id}")
            assert raw.value == {"active_call_id": "c2"}
            payload = await o.suspension(call_id)
            assert payload is not None
            assert "messages" not in payload  # the snapshot became a cursor
            assert payload["transcript"]["conversation_id"] == "run-compact"


async def test_a_parked_batch_stores_one_snapshot_plus_per_call_indexes() -> None:
    log = MemorySteps()
    await _parked_batch(log, "run-indexed-snapshot")

    active = await log.read("run-indexed-snapshot", "agent:active-suspension")
    assert isinstance(active.value, dict)
    assert "messages" in active.value["continuation"]
    for call_id in ("c2", "c3"):
        per_call = await log.read("run-indexed-snapshot", f"suspend:{call_id}")
        assert per_call.value == {"active_call_id": "c2"}


async def test_a_resume_time_suspension_reparks_the_batch_without_orphaning_answers() -> None:
    """A gate that suspends anew mid-finalize keeps the whole batch.

    Only the reissued call waits, under its new pending id; every other recorded answer
    survives the re-park, and answering the reissue alone finishes the round.
    """

    class ReAsk:
        """Suspend c2 once at resume time, then let everything through."""

        def __init__(self) -> None:
            self.asked = False

        async def __call__(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> Any:
            if call["id"] == "c2" and not self.asked:
                self.asked = True
                return Suspend({"type": "suspend", "pending_id": "re-approve-c2"})
            return Continue()

    log = MemorySteps()
    await _parked_batch(log, "run-reask")
    with pytest.raises(AgentSuspended):
        await AgentRuntime(store=log).resume(
            "run-reask",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
        )

    reask = ReAsk()
    mid = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(AgentSuspended) as reparked:
        await AgentRuntime(store=log).resume(
            "run-reask",
            "approve-c3",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            mid,
            controls=ControlPlane(on_resume=reask),
        )

    assert reparked.value.pending == [("re-approve-c2", "c2")]  # only the reissue waits
    assert mid.ran == []  # nothing executed while the batch went back to waiting

    final = Tools(names=["read", DEPLOY, "notify"])
    model = scripted(says("done"))
    outcome = await AgentRuntime(store=log).resume(
        "run-reask",
        "re-approve-c2",
        {"type": "text", "text": "approved"},
        model,
        final,
        controls=ControlPlane(on_resume=reask),
    )

    assert final.ran == [DEPLOY, DEPLOY]  # c3's earlier answer was not orphaned
    assert outcome["content"] == "done"
    answered = [m.tool_call_id for m in model.seen[0] if isinstance(m, ToolMessage)]
    assert answered == ["c1", "c2", "c3"]


async def test_switching_from_a_repark_preserves_effects_that_already_committed() -> None:
    """A completed sibling is replayed while only the still-parked effect is cancelled."""

    class ReAskSecond:
        async def __call__(self, ctx: Ctx, call: ToolCall, resume: ResumeInput) -> Any:
            if call["id"] == "c3":
                return Suspend({"type": "suspend", "pending_id": "re-approve-c3"})
            return Continue()

    log = MemorySteps()
    await _parked_batch(log, "switch-repark")
    runtime = AgentRuntime(store=log)
    with pytest.raises(AgentSuspended):
        await runtime.resume(
            "switch-repark",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
        )

    executed = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(AgentSuspended):
        await runtime.resume(
            "switch-repark",
            "approve-c3",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            executed,
            controls=ControlPlane(on_resume=ReAskSecond()),
        )
    assert executed.ran == [DEPLOY]

    model = scripted(says("switched"))
    outcome = await runtime.run(
        "switch-repark",
        model,
        Tools(names=["read", DEPLOY, "notify"]),
        "cancel the remaining deployment",
    )

    results = {m.tool_call_id: m for m in model.seen[0] if isinstance(m, ToolMessage)}
    assert results["c2"].status == "success"
    assert results["c3"].status == "error"
    assert outcome["content"] == "switched"


async def test_switching_from_a_batch_emits_one_cancellation_event_per_cancelled_call() -> None:
    log = MemorySteps()
    await _parked_batch(log, "switch-events")
    cancelled: list[str] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == EventType.TOOL_REQUEST_CANCELLED:
            cancelled.append(str(payload["call_id"]))

    await AgentRuntime(store=log, emit=emit).run(
        "switch-events",
        scripted(says("switched")),
        Tools(names=["read", DEPLOY, "notify"]),
        "cancel both deployments",
    )

    assert cancelled == ["c2", "c3"]


async def test_finalize_replays_a_committed_effect_before_revalidating_policy() -> None:
    """A crash after effect commit cannot turn that completed fact into a later denial."""

    class CrashAfterFirstDeploy(MemorySteps):
        crashed = False

        async def finish_effect(
            self, run_id: str, key: str, value: Any, token: int = 0
        ) -> None:
            await super().finish_effect(run_id, key, value, token)
            if key == "c2" and not self.crashed:
                self.crashed = True
                raise RuntimeError("worker crashed after committed deploy")

    log = CrashAfterFirstDeploy()
    await _parked_batch(log, "finalize-commit-wins")
    runtime = AgentRuntime(store=log)
    with pytest.raises(AgentSuspended):
        await runtime.resume(
            "finalize-commit-wins",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
        )

    async def changed_policy(ctx: Ctx, call: ToolCall, resume: ResumeInput) -> Any:
        if call["id"] == "c2" and log.crashed:
            return Deny({"type": "error", "message": "policy changed"})
        return Continue()

    controls = ControlPlane(on_resume=changed_policy)
    first = Tools(names=["read", DEPLOY, "notify"])
    with pytest.raises(RuntimeError, match="worker crashed after committed deploy"):
        await runtime.resume(
            "finalize-commit-wins",
            "approve-c3",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            first,
            controls=controls,
        )

    active = await Orchestrator("finalize-commit-wins", log).active_continuation()
    assert active is not None
    assert active["state"] == "finalizing"
    assert set(active["answers"]) == {"c2", "c3"}  # the final answer preceded every effect

    waiting = decode_continuation(active["continuation"])
    assert waiting is not None
    final = Tools(names=["read", DEPLOY, "notify"])
    model = scripted(says("done"))
    await runtime.recover(
        "finalize-commit-wins",
        list(waiting.messages),
        model,
        final,
        controls=controls,
    )

    by_id = {m.tool_call_id: m for m in model.seen[0] if isinstance(m, ToolMessage)}
    assert (first.ran, final.ran, by_id["c2"].status) == ([DEPLOY], [DEPLOY], "success")


async def test_repeated_final_answer_can_reenter_a_committed_resuming_transition() -> None:
    """A response retry after the finalize transition commits must not fail with a 500."""

    class CrashAfterResumingTransition(MemorySteps):
        crashed = False

        async def commit_transition(
            self, run_id: str, transition: Any, token: int = 0
        ) -> set[str]:
            inserted = await super().commit_transition(run_id, transition, token)
            if not self.crashed and any(
                isinstance(value, dict) and value.get("state") == "resuming"
                for value in transition.controls.values()
            ):
                self.crashed = True
                raise RuntimeError("worker crashed after resuming commit")
            return inserted

    log = CrashAfterResumingTransition()
    await _parked_batch(log, "resume-redelivery")
    runtime = AgentRuntime(store=log)
    with pytest.raises(AgentSuspended):
        await runtime.resume(
            "resume-redelivery",
            "approve-c2",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
        )

    with pytest.raises(RuntimeError, match="worker crashed after resuming commit"):
        await runtime.resume(
            "resume-redelivery",
            "approve-c3",
            {"type": "text", "text": "approved"},
            scripted(says("unused")),
            Tools(names=["read", DEPLOY, "notify"]),
        )

    active = await Orchestrator("resume-redelivery", log).active_continuation()
    assert active is not None and active["state"] == "resuming"
    final = Tools(names=["read", DEPLOY, "notify"])
    outcome = await runtime.resume(
        "resume-redelivery",
        "approve-c3",
        {"type": "text", "text": "approved"},
        scripted(says("done")),
        final,
    )

    assert final.ran == []  # both effects were already committed before the crash
    assert outcome["content"] == "done"


async def test_post_tool_use_crosses_its_durable_boundary_once_per_call() -> None:
    """A replayed result must not re-run `post_tool_use`.

    The hook may write to external stores; its `after:` marker is the durable boundary, so a
    recovery replay of a done step skips a hook that already crossed it.
    """
    log = MemorySteps()
    seen: list[str] = []

    async def audit(ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        seen.append(str(call["id"]))

    controls = ControlPlane(post_tool_use=audit)
    calls = [a_call("c1", "read")]
    async with Orchestrator("hook-once", log) as owner:
        await owner.execute_round(Tools(names=["read"]), calls, lambda: False, controls=controls)
    assert seen == ["c1"]

    async with Orchestrator("hook-once", log) as owner:
        await owner.recover_pending(
            [AIMessage(content="", tool_calls=calls)],
            Tools(names=["read"]),
            controls=controls,
        )
    assert seen == ["c1"]  # the recovery replay skipped the hook

    async with Orchestrator("hook-once", log) as owner:
        await owner.resume_effect(
            Tools(names=["read"]),
            calls[0],
            {"type": "text", "text": "approved"},
            {"type": "suspend", "pending_id": "p1"},
            "",
            controls=controls,
            park=False,
        )
    assert seen == ["c1"]  # and so did the resume-time done-replay


async def test_a_suspension_survives_the_process_and_the_run_continues() -> None:
    """The whole cycle. `on_suspend` writes to the same ledger the steps use — no second store."""
    log = MemorySteps()
    runtime = AgentRuntime(store=log)
    tools = Tools(names=["read", DEPLOY])

    async def ask_about_deploy(call: ToolCall) -> dict[str, Any] | None:
        if call["name"] == DEPLOY:
            return {"type": "suspend", "pending_id": call["id"], "handle": "change-req-88"}
        return None

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-1",
            scripted(says("", a_call("c1", "read"), a_call("c2", DEPLOY))),
            tools,
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask_about_deploy))),
            rules_version="v1",
        )

    assert tools.ran == ["read"]

    # ── a different process, hours later ──
    waiting = decode_continuation(await Orchestrator("run-1", log).suspension("c2"))
    assert waiting is not None
    assert waiting.request["handle"] == "change-req-88"  # the tool's own handle survived
    assert waiting.turn == 0
    assert [c["id"] for c in waiting.completed] == ["c1"]
    shape = [type(m).__name__ for m in waiting.messages]
    assert shape == ["HumanMessage", "AIMessage", "ToolMessage"]

    resumed = Tools(names=["read", DEPLOY])
    outcome = await runtime.resume(
        "run-1",
        "c2",
        {"type": "text", "text": "approved"},
        scripted(says("deployed")),
        resumed,
    )

    assert resumed.ran == [DEPLOY]  # read is not replayed; the approved effect runs now
    assert outcome["content"] == "deployed"


async def test_runtime_resume_revalidates_the_latest_policy_before_the_effect() -> None:
    """An approval is input, not authority: a deny added while waiting still wins."""
    log = MemorySteps()
    runtime = AgentRuntime(store=log)

    async def ask(_call: ToolCall) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approval-c1"}

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-policy-change",
            scripted(says("", a_call("c1", DEPLOY))),
            Tools(names=[DEPLOY]),
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask))),
            rules_version="v1",
        )

    current = PolicyContext(rules=[Rule(effect="deny", tool=DEPLOY)], version="v2")
    resumed = Tools(names=[DEPLOY])
    model = scripted(says("blocked"))
    outcome = await runtime.resume(
        "run-policy-change",
        "approval-c1",
        {"type": "text", "text": "approved"},
        model,
        resumed,
        controls=ControlPlane(on_resume=current.resume_stage(resumed)),
        rules_version=current.version,
    )

    assert resumed.ran == []
    assert outcome["content"] == "blocked"


async def test_the_same_ask_rule_is_satisfied_by_the_human_approval() -> None:
    """Revalidating an unchanged ask must not park the same call forever."""
    log = MemorySteps()
    runtime = AgentRuntime(store=log)
    policy = PolicyContext(rules=[Rule(effect="ask", tool=DEPLOY)], version="v1")
    first = Tools(names=[DEPLOY])

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-same-policy",
            scripted(says("", a_call("c1", DEPLOY))),
            first,
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(policy.stage(first))),
            rules_version=policy.fingerprint,
        )

    resumed = Tools(names=[DEPLOY])
    outcome = await runtime.resume(
        "run-same-policy",
        "c1",
        {"type": "text", "text": "approved"},
        scripted(says("deployed")),
        resumed,
        controls=ControlPlane(on_resume=policy.resume_stage(resumed)),
        rules_version=policy.fingerprint,
    )

    assert resumed.ran == [DEPLOY]
    assert outcome["content"] == "deployed"


async def test_the_resume_re_runs_the_rules_instead_of_trusting_the_approval() -> None:
    """A rule added while the run was suspended must still apply.

    Storing "approved" would let Monday's approval ignore Tuesday's deny rule. So the record keeps
    the call and a `rules_version`, and the resume asks `resolve_rules` again.
    """
    log = MemorySteps()
    call = a_call("c9", DEPLOY)
    was = PolicyContext(rules=[], version="v1")

    o = Orchestrator("run-2", log)
    await o.suspend(
        "c9",
        encode_continuation(
            call, {"type": "suspend", "pending_id": "c9"}, [], [], was.version
        ),
    )
    waiting = decode_continuation(await o.suspension("c9"))
    assert waiting is not None

    # Someone approved. Meanwhile the rules moved on.
    now = PolicyContext(rules=[Rule(effect="deny", tool=DEPLOY)], version="v2")

    assert waiting.rules_version != now.version  # the window is visible, not assumed
    decision = await resolve_rules(
        waiting.call,
        rules=now.rules,
        mode=now.mode,
        subscriber_answer={"type": "allow"},  # the human said yes
        definition={},
    )

    assert decision is not None
    assert decision["type"] == "error"  # and the rule still wins


async def test_a_revoked_rule_re_asks_even_though_the_label_never_moved() -> None:
    """The case a caller-supplied version cannot see, and the reason identity is derived.

    Rules derived per requesting subject change when a role is revoked, and no label anywhere
    changes with them. Comparing labels reads that as "same policy" and runs the effect.
    """
    granted = PolicyContext(rules=[Rule(effect="allow", tool=DEPLOY)], version="v1")
    revoked = PolicyContext(rules=[], version="v1")  # same label, the grant is gone
    tools = Tools(names=[DEPLOY])

    assert granted.version == revoked.version
    decision = await revoked.resume_stage(tools)(
        Ctx(turn=0),
        a_call("c1", DEPLOY),
        ResumeInput(
            answer={"type": "text", "text": "approved"},
            request={"type": "suspend", "pending_id": "c1"},
            suspended_rules_version=granted.fingerprint,
            current_rules_version=revoked.fingerprint,
        ),
    )

    assert isinstance(decision, Suspend)  # asked again, not waved through


async def test_a_relabelled_policy_still_satisfies_the_approval() -> None:
    """The other direction: a label is not authority, so editing one must not re-ask."""
    asked = PolicyContext(rules=[Rule(effect="ask", tool=DEPLOY)], version="v1")
    relabelled = PolicyContext(rules=[Rule(effect="ask", tool=DEPLOY)], version="v2-hotfix")
    tools = Tools(names=[DEPLOY])

    assert asked.fingerprint == relabelled.fingerprint
    decision = await relabelled.resume_stage(tools)(
        Ctx(turn=0),
        a_call("c1", DEPLOY),
        ResumeInput(
            answer={"type": "text", "text": "approved"},
            request={"type": "suspend", "pending_id": "c1"},
            suspended_rules_version=asked.fingerprint,
            current_rules_version=relabelled.fingerprint,
        ),
    )

    assert isinstance(decision, Continue)


async def test_a_fingerprint_is_stable_across_rule_order() -> None:
    """Identity is what the policy *is*. The same rules written in another order are the same."""
    one = PolicyContext(rules=[Rule("allow", "read"), Rule("deny", DEPLOY, "prod")])
    other = PolicyContext(rules=[Rule("deny", DEPLOY, "prod"), Rule("allow", "read")])

    assert one.fingerprint == other.fingerprint
    # and a tool-wide rule is not the same rule as one scoped to every input of that tool
    assert PolicyContext(rules=[Rule("ask", DEPLOY)]).fingerprint != PolicyContext(
        rules=[Rule("ask", DEPLOY, "")]
    ).fingerprint


async def test_a_suspension_and_its_events_name_who_it_was_for() -> None:
    """"Whose deploy was waiting" must be answerable from the record, not reconstructed later."""
    log = MemorySteps()
    seen: list[dict[str, Any]] = []

    async def sink(event_type: str, payload: dict[str, Any]) -> None:
        if event_type in {"pre_tool_use", "permission_request"}:
            seen.append({"event": str(event_type), **payload})

    runtime = AgentRuntime(store=log, emit=sink)
    policy = PolicyContext(rules=[Rule(effect="ask", tool=DEPLOY)])
    tools = Tools(names=[DEPLOY])

    with pytest.raises(AgentSuspended):
        await runtime.run(
            "run-subject",
            scripted(says("", a_call("c1", DEPLOY))),
            tools,
            "ship it",
            controls=ControlPlane(pre_tool_use=Permissions(policy.stage(tools))),
            rules_version=policy.fingerprint,
            subject="svc-deployer@example",
        )

    assert [item["subject"] for item in seen] == ["svc-deployer@example"] * 2
    assert seen[-1]["request"]["subject"] == "svc-deployer@example"

    parked = decode_continuation(await Orchestrator("run-subject", log).suspension("c1"))
    assert parked is not None
    assert parked.subject == "svc-deployer@example"


async def test_an_unnamed_subject_is_absent_rather_than_empty() -> None:
    """A framework that stamped `""` would put a name it invented into an audit record."""
    seen: list[dict[str, Any]] = []

    async def sink(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "pre_tool_use":
            seen.append(payload)

    runtime = AgentRuntime(store=MemorySteps(), emit=sink)
    tools = Tools(names=[DEPLOY])

    await runtime.run(
        "run-anon", scripted(says("", a_call("c1", DEPLOY)), says("done")), tools, "go"
    )

    assert seen and all("subject" not in payload for payload in seen)


async def test_nothing_parked_reads_as_nothing() -> None:
    o = Orchestrator("run-3", MemorySteps())
    assert decode_continuation(await o.suspension("never-suspended")) is None


def test_a_legacy_tool_originated_suspension_cannot_be_replayed_as_permission() -> None:
    payload = encode_continuation(
        a_call("c1", DEPLOY),
        {"type": "suspend", "pending_id": "p1"},
        [],
        [],
    )
    payload.pop("origin")
    payload["kind"] = "elicitation"

    with pytest.raises(ValueError, match="tool-originated legacy suspension"):
        decode_continuation(payload)


async def test_a_policy_context_is_one_owned_thing() -> None:
    """A permissive hook cannot bypass rules owned by the same policy context."""
    tools = Tools(names=[DEPLOY])
    policy = PolicyContext(rules=[Rule(effect="deny", tool=DEPLOY)], version="v3")
    seen: list[str] = []

    async def permissive(call: ToolCall) -> dict[str, Any] | None:
        seen.append("hook")
        return {"type": "allow"}

    chain = Permissions(gate(permissive), policy.stage(tools))
    decision = await chain(Ctx(turn=0), a_call("c1", DEPLOY))

    assert seen == ["hook"]
    assert isinstance(decision, Deny)  # the rules had the last word
    assert decision.result["type"] == "error"


@pytest.mark.parametrize("effect", ["deny", "ask"])
async def test_an_approval_cannot_outrank_an_immune_rule(effect: str) -> None:
    """Both halves of the immune region hold across a suspension, not just deny."""
    decision = await resolve_rules(
        a_call("c1", DEPLOY),
        rules=[Rule(effect=effect, tool=DEPLOY)],  # type: ignore[arg-type]
        mode="bypass",
        subscriber_answer={"type": "allow"},
        definition={},
    )

    assert decision is not None
    assert decision["type"] == {"deny": "error", "ask": "suspend"}[effect]
