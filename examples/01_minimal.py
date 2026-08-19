"""One turn: the model asks for a tool, the runtime runs it, the model answers.

    uv run python examples/01_minimal.py

This is the default `AgentRuntime`: no ledger. The tool still runs once because the process
lived. Crash recovery and permission parking start at `examples/03_recovery.py` and
`examples/02_approval.py`.
"""

import asyncio

from _scripted import Files, calling, says, scripted
from nexora import AgentRuntime


async def main() -> None:
    model = scripted(
        says("", calling("c1", "read", path="notes.md")),
        says("The file mentions a social security number."),
    )
    files = Files()
    runtime = AgentRuntime()

    outcome = await runtime.run("run-1", model, files, "what is in notes.md?")

    print(f"answer      {outcome['content']}")
    print(f"stop reason {outcome['stop_reason']}")
    print(f"tools ran   {files.ran}")

    # The examples assert what their prose claims, so a contract change breaks them loudly.
    assert files.ran == ["read"], "the model asked for one tool and it ran once"
    assert outcome["stop_reason"] == "completed"


if __name__ == "__main__":
    asyncio.run(main())
