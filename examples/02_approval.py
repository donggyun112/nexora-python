"""Stop for a person, let the worker exit, resume days later.

    uv run python examples/02_approval.py

`pre_tool_use` is an awaited call whose return value decides. `Suspend` means the run parks: the
continuation is written to the ledger, `AgentSuspended` is raised, and no generator, worker or
lease stays alive while the approval is outstanding.

The answer is an input to the next decision and never the decision itself. `on_resume` re-decides
under the rules in force *now*, because a suspension's window is days and rules move inside it —
here the same approval is refused once the policy has changed.
"""

import asyncio
from typing import Any

from _scripted import Files, calling, says, scripted
from nexora import (
    AgentRuntime,
    Continue,
    ControlPlane,
    Ctx,
    Deny,
    MemorySteps,
    Permissions,
    ResumeInput,
    gate,
)
from nexora.controls import ToolDecision
from nexora.orchestrator import AgentSuspended


async def ask_before_deleting(call: dict[str, Any]) -> dict[str, Any] | None:
    """A `pre_tool_use` stage. `None` allows; a `suspend` result asks a person."""
    if call["name"] == "delete":
        return {"type": "suspend", "pending_id": f"approve-{call['id']}"}
    return None


def revalidate(*, deleting_still_allowed: bool) -> Any:
    """An `on_resume` stage: the human said yes, but does the current policy agree?"""

    async def stage(ctx: Ctx, call: dict[str, Any], resume: ResumeInput) -> ToolDecision:
        if resume.answer.get("decision") != "approve":
            return Deny({"type": "error", "message": "a person refused this"})
        if not deleting_still_allowed:
            return Deny({"type": "error", "message": "policy changed while you were away"})
        return Continue()

    return stage


async def main() -> None:
    store = MemorySteps()  # one store, so the second attempt sees the first one's ledger
    files = Files({"stale.md": "old"})
    controls = ControlPlane(
        pre_tool_use=Permissions(gate(ask_before_deleting)),
        on_resume=revalidate(deleting_still_allowed=True),
    )

    # ── attempt one: the model asks to delete, the gate parks the run ────────
    asking = scripted(says("", calling("c1", "delete", path="stale.md")))
    try:
        await AgentRuntime(store=store).run(
            "run-2", asking, files, "delete stale.md", controls=controls
        )
        raise AssertionError("unreachable: the gate suspends")
    except AgentSuspended as parked:
        pending_id = parked.pending_id  # bound here: Python unbinds `parked` after the block
        print(f"parked      pending_id={pending_id}")
        print(f"tools ran   {files.ran}  ← the delete never happened")
        assert files.ran == [], "a suspended call must not have executed"

    # ── the worker is gone. Whenever the answer arrives, a new attempt resumes ──
    answered = await AgentRuntime(store=store).resume(
        "run-2",
        pending_id,
        {"decision": "approve"},
        scripted(says("Deleted stale.md.")),
        files,
        controls=controls,
    )
    print(f"resumed     {answered['content']}")
    print(f"tools ran   {files.ran}")
    assert files.ran == ["delete"], "the approved call runs exactly once"

    # ── the same approval under a policy that has since said no ─────────────
    refusing = ControlPlane(
        pre_tool_use=Permissions(gate(ask_before_deleting)),
        on_resume=revalidate(deleting_still_allowed=False),
    )
    fresh_store, fresh_files = MemorySteps(), Files({"stale.md": "old"})
    try:
        await AgentRuntime(store=fresh_store).run(
            "run-3",
            scripted(says("", calling("c1", "delete", path="stale.md"))),
            fresh_files,
            "delete stale.md",
            controls=refusing,
        )
    except AgentSuspended as parked_again:
        outcome = await AgentRuntime(store=fresh_store).resume(
            "run-3",
            parked_again.pending_id,
            {"decision": "approve"},
            scripted(says("I could not delete it.")),
            fresh_files,
            controls=refusing,
        )
        print(f"revalidated {outcome['content']}")
        print(f"tools ran   {fresh_files.ran}  ← approved by a person, refused by policy")
        assert fresh_files.ran == [], "current policy outranks the stored approval"


if __name__ == "__main__":
    asyncio.run(main())
