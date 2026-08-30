"""Where to put policy, and which seam actually reaches the durable copy.

    uv run python examples/04_control_plane.py

Two masks that look alike and are not. `on_inputs` screens what enters model context, so the model
and the transcript see the redacted text. It does not reach the ledger: a tool result was recorded
before any control point exists. Redacting *that* means wrapping the executor, which the
orchestrator nests inside the durable step — so the recorded value is already masked.

The third piece is `before_finish`: a verifier that says "not done yet" and sends the run around
again instead of letting it stop.
"""

import asyncio
import re
from typing import Any

from _scripted import Files, calling, says, scripted
from langchain_core.messages import HumanMessage
from semora import (
    AgentRuntime,
    ControlPlane,
    Ctx,
    FinishPolicy,
    Halt,
    Ingress,
    MemorySteps,
    PendingInput,
    Proceed,
)
from semora.controls import TurnDecision

SSN = re.compile(r"\d{3}-\d{2}-\d{4}")


class Redacting(Files):
    """Mask tool results before the durable step records them."""

    async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
        result = await super().execute(name, call_id, args)
        return {**result, "text": SSN.sub("***-**-****", str(result["text"]))}


async def mask_inputs(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput]:
    """An `on_inputs` screen. Chains in order, and each sees the previous one's output."""
    return [
        PendingInput(
            item.kind,
            item.message.model_copy(
                update={"content": SSN.sub("***-**-****", str(item.message.content))}
            ),
            item.origin_id,
        )
        for item in inputs
    ]


def needs_a_citation() -> Any:
    """A `before_finish` gate. `Proceed` vetoes the ending; silence lets it end."""
    asked = 0

    async def verify(ctx: Ctx, reason: str) -> TurnDecision:
        nonlocal asked
        asked += 1
        if asked == 1 and "notes.md" not in ctx.text:
            return Proceed([HumanMessage("name the file you read")])
        return Halt(reason)  # anything but Proceed means "no objection"

    return verify


async def main() -> None:
    store = MemorySteps()
    files = Redacting()
    model = scripted(
        says("", calling("c1", "read", path="notes.md")),
        says("There is a number in the file."),  # no citation — the verifier objects
        says("The number lives in notes.md."),
    )

    outcome = await AgentRuntime(store=store).run(
        "run-5",
        model,
        files,
        "summarise notes.md for 123-45-6789",
        controls=ControlPlane(
            on_inputs=Ingress(mask_inputs),
            before_finish=FinishPolicy(needs_a_citation()),
        ),
    )

    ledgered = (await store.read("run-5", "c1")).value
    prompt = (await store.list_inputs("run-5"))[0].value["message"]["data"]["content"]

    print(f"answer        {outcome['content']}  ← the verifier sent it around once")
    print(f"tool result   {ledgered['text']!r}  ← masked in the ledger, by the Tools wrapper")
    print(f"queued prompt {prompt!r}")
    print("              ↑ on_inputs masks what the model sees, not what the inbox stored:")
    print("                mask at your edge before submit() if the durable copy matters.")

    assert "123-45-6789" not in ledgered["text"], "the Tools wrapper redacts the durable copy"
    assert "123-45-6789" in prompt, "on_inputs does not reach the input ledger — the finding"
    assert outcome["content"].endswith("notes.md."), "before_finish sent the run around once"


if __name__ == "__main__":
    asyncio.run(main())
