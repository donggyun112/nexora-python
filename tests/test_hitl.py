"""Human-in-the-loop, end to end: stop for a person, survive the process, continue.

The window here is days. Everything in this file exists because of that: the snapshot has to
outlive the worker, and the decision has to be re-derived rather than replayed.
"""

from typing import Any

import pytest
from nexora import AgentRuntime
from nexora.contracts import ToolCall
from nexora.controls import (
    Continue,
    ControlPlane,
    Ctx,
    Deny,
    Permissions,
    ResumeInput,
    Suspend,
    gate,
)
from nexora.history import decode_continuation, encode_continuation
from nexora.orchestrator import AgentSuspended, MemorySteps, Orchestrator
from nexora_permissions import PolicyContext, Rule, resolve_rules

from tests.test_loop import Tools, a_call, says, scripted

DEPLOY = "deploy"


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
