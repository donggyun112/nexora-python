"""One turn: the model asks for a tool, the runtime runs it, the model answers.

    uv run python examples/01_minimal.py

Every tool call crosses the durable boundary even here. `MemorySteps` is a `StepLog` that dies
with the process — enough to see the shape before paying for a database.
"""

import asyncio

from _scripted import Files, calling, says, scripted

from nexora import AgentRuntime
from nexora.orchestrator import MemorySteps


async def main() -> None:
    model = scripted(
        says("", calling("c1", "read", path="notes.md")),
        says("The file mentions a social security number."),
    )
    files = Files()
    runtime = AgentRuntime(store=MemorySteps())

    outcome = await runtime.run("run-1", model, files, "what is in notes.md?")

    print(f"answer      {outcome['content']}")
    print(f"stop reason {outcome['stop_reason']}")
    print(f"tools ran   {files.ran}")

    # The examples assert what their prose claims, so a contract change breaks them loudly.
    assert files.ran == ["read"], "the model asked for one tool and it ran once"
    assert outcome["stop_reason"] == "completed"


if __name__ == "__main__":
    asyncio.run(main())
