"""The agent loop with no ledger at all — no orchestrator, no store, no durability.

    uv run python examples/06_bare_loop.py

The default `AgentRuntime` drives `react_loop` with its direct `execute_calls`, so the durable
boundary is optional rather than something the loop depends on. Nothing in
`semora.orchestrator` participates.

You keep the whole control plane: `on_inputs`, `before_model`, `pre_tool_use`, `post_tool_use`,
`before_finish`, the event stream, steers, and the stop hook. What you give up is the three things
the ledger is for — call-id idempotency, crash recovery, and parking a call for a person.

That last one is a hard edge rather than a missing nicety, and this file ends by showing it: a gate
that answers `Suspend` has nowhere to write its continuation, so the round refuses instead of
pretending it parked.
"""

import asyncio
from typing import Any

from _scripted import Files, calling, says, scripted
from semora import (
    AgentRuntime,
    ControlPlane,
    Ctx,
    FinishPolicy,
    Halt,
    Permissions,
    Proceed,
    gate,
)
from semora.contracts import StopReason, ToolCall
from semora.tools import RoundSuspended


async def no_deleting(call: ToolCall) -> dict[str, Any] | None:
    """A `pre_tool_use` stage. Denying needs no ledger — the model just sees a failed call."""
    if call["name"] == "delete":
        return {"type": "error", "message": "this loop does not delete things"}
    return None


async def wants_a_citation(ctx: Ctx, reason: StopReason) -> Any:
    """A `before_finish` gate, also ledger-free: it only sends the loop around again."""
    if "notes.md" not in ctx.text:
        return Proceed([])
    return Halt(reason)


async def cap(turn: int, text: str, calls: list[dict[str, Any]]) -> bool:
    """The loop has no built-in iteration limit; how long an agent may run is the caller's call."""
    return turn >= 5


async def main() -> None:
    events: list[dict[str, Any]] = []
    model = scripted(
        says("", calling("c1", "delete", path="notes.md"), calling("c2", "read", path="notes.md")),
        says("나는 파일을 읽었다"),  # no citation — the verifier objects once
        says("notes.md 를 읽었다"),
    )

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    await AgentRuntime().run(
        "bare-example",
        model,
        Files(),
        controls=ControlPlane(
            pre_tool_use=Permissions(gate(no_deleting)),
            before_finish=FinishPolicy(wants_a_citation),
        ),
        should_stop_after_turn=cap,
        on_event=collect,
    )

    refused = next(e for e in events if e["type"] == "tool_result" and e["is_error"])
    print(f"denied      {refused['name']} · {refused['result']['message']}")
    print(f"answer      {events[-1]['content']!r} · {events[-1]['stop_reason']}")
    print(f"events      {[e['type'] for e in events]}")

    assert refused["executed"] is False, "a denied call must not have run"
    assert events[-1]["content"] == "notes.md 를 읽었다", "before_finish sent it around once"

    # ── the one thing that needs a ledger ───────────────────────────────────
    async def ask_a_person(call: ToolCall) -> dict[str, Any]:
        return {"type": "suspend", "pending_id": "approve-1"}

    try:
        await AgentRuntime().run(
            "bare-suspension",
            scripted(says("", calling("c9", "read", path="notes.md"))),
            Files(),
            controls=ControlPlane(pre_tool_use=Permissions(gate(ask_a_person))),
        )
        raise AssertionError("unreachable: there is nowhere to write the continuation")
    except RoundSuspended as refused_to_park:
        print(f"\nsuspend     {str(refused_to_park)[:66]}…")
        print("            ← parking is the one control decision a bare loop cannot honour")


if __name__ == "__main__":
    asyncio.run(main())
