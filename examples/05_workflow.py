"""A durable workflow with an agent inside one of its steps, and a human in the middle.

    uv run python examples/05_workflow.py

This is the shape a workflow engine usually asks for a DSL to express. Here it is ordinary Python:
each `orchestrator.run(name, fn)` is a step that happens once ever, and `signal(name)` ends the
attempt when the answer it needs does not exist yet.

The agent is not a special case. `run_agent(react_loop(...))` collapses a whole agent run to one
value, so it is a step like any other — which is what stops a doctor's sign-off from re-running the
draft, and stops a retry from sending the prescription twice.

Replay is the mechanism, so the function runs top to bottom on every attempt. What makes that safe
is that a finished step returns its recorded value instead of calling its function again — printed
below as the second attempt does no work the first one already did.
"""

import asyncio
from datetime import date
from typing import Any

from _scripted import Files, Scripted, calling, says, scripted
from nexora import MemorySteps, react_loop
from nexora.orchestrator import Orchestrator, Suspended, run_agent

SENT: list[str] = []
"""Every external effect this workflow performed. Nothing here may appear twice."""


async def send_to_pharmacy(meds: str) -> str:
    SENT.append(f"pharmacy:{meds}")
    return "accepted"


async def send_invoice(amount: str) -> str:
    SENT.append(f"invoice:{amount}")
    return "sent"


async def discharge(orchestrator: Orchestrator, patient_id: str, model: Any) -> dict[str, Any]:
    """One durable workflow. Every line is a step, including the agent."""
    since = await orchestrator.run("day", lambda: date(2026, 8, 6).isoformat())

    plan = await orchestrator.run(
        "draft",
        lambda: run_agent(
            react_loop(model, Files({"notes.md": f"{patient_id} stable since {since}"}))
        ),
    )

    # Ends the attempt if no answer has been written yet. No worker, no lease, no timeout.
    signed_off = await orchestrator.signal(f"signoff:{patient_id}")

    await orchestrator.run("meds", lambda: send_to_pharmacy("amoxicillin 500mg"))
    await orchestrator.run("bill", lambda: send_invoice("120.00"))
    return {"plan": plan["content"], "signed_off": signed_off}


class Explodes(Scripted):
    """Fail if a replay calls the model instead of reusing the draft step."""

    def __init__(self) -> None:
        super().__init__(messages=iter(()))

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the draft step re-ran the agent instead of reusing its result")


def a_model() -> Any:
    return scripted(
        says("", calling("c1", "read", path="notes.md")),
        says("Discharge plan: continue antibiotics, review in two weeks."),
    )


async def main() -> None:
    log = MemorySteps()
    drafted = a_model()

    # ── attempt one: runs up to the sign-off, then stops ─────────────────────
    try:
        await discharge(Orchestrator("discharge-7", log), "patient-7", drafted)
        raise AssertionError("unreachable: the signal has no answer yet")
    except Suspended as waiting:
        print(f"attempt 1   stopped at signal {waiting.signal!r}")
        print(f"            effects sent {SENT}")

    assert SENT == [], "nothing past the sign-off may have happened"
    assert (await log.read("discharge-7", "draft")).status == "done"

    # ── the doctor answers, from outside the run ─────────────────────────────
    await Orchestrator("discharge-7", log).resolve("signoff:patient-7", {"by": "dr-kim"})
    print("            sign-off written by an outsider — no worker was waiting")

    # ── attempt two: replays, reuses, and finishes ───────────────────────────
    outcome = await discharge(Orchestrator("discharge-7", log), "patient-7", Explodes())

    print(f"attempt 2   plan  {outcome['plan']!r}")
    print(f"            signed_off {outcome['signed_off']}")
    print(f"            effects sent {SENT}")

    assert SENT == ["pharmacy:amoxicillin 500mg", "invoice:120.00"], "each effect exactly once"

    # ── a third attempt does nothing at all ─────────────────────────────────
    await discharge(Orchestrator("discharge-7", log), "patient-7", Explodes())
    print(f"attempt 3   effects sent {SENT}  ← unchanged: every step was already done")
    assert SENT == ["pharmacy:amoxicillin 500mg", "invoice:120.00"]


if __name__ == "__main__":
    asyncio.run(main())
