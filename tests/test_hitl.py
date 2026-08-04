"""Human-in-the-loop, end to end: stop for a person, survive the process, continue.

The window here is days. Everything in this file exists because of that: the snapshot has to
outlive the worker, and the decision has to be re-derived rather than replayed.
"""

from typing import Any

import pytest

from nexora import AgentRuntime
from nexora.contracts import ToolCall
from nexora.controls import ControlPlane, Permissions, gate
from nexora.history import decode_continuation, encode_continuation
from nexora.orchestrator import AgentSuspended, MemorySteps, Orchestrator, PermissionChain
from nexora.permissions import PolicyContext, Rule, resolve_rules
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
            controls=ControlPlane(pre_tool_use=Permissions(gate(policy.stage(first)))),
            rules_version=policy.version,
        )

    resumed = Tools(names=[DEPLOY])
    outcome = await runtime.resume(
        "run-same-policy",
        "c1",
        {"type": "text", "text": "approved"},
        scripted(says("deployed")),
        resumed,
        controls=ControlPlane(on_resume=policy.resume_stage(resumed)),
        rules_version=policy.version,
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
    """Rules, mode and version travelled separately and so belonged to nobody. As a stage in a
    chain, a hook's `allow` still does not end anything — the rules run after it and re-validate."""
    tools = Tools(names=[DEPLOY])
    policy = PolicyContext(rules=[Rule(effect="deny", tool=DEPLOY)], version="v3")
    seen: list[str] = []

    async def permissive(call: ToolCall) -> dict[str, Any] | None:
        seen.append("hook")
        return {"type": "allow"}

    chain = PermissionChain(permissive, policy.stage(tools))
    decision = await chain.resolve(a_call("c1", DEPLOY))

    assert seen == ["hook"]
    assert decision is not None
    assert decision["type"] == "error"  # the rules had the last word


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
