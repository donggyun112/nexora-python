"""A worker dies mid-round. Recovery finishes it without asking the model again.

    uv run python examples/03_recovery.py

The model asked for two writes. The first committed, then the process went away. The ledger keys
every call by its `call_id`, so recovery can tell the three cases apart: `done` results are
restored, `absent` calls run, and a call that started and never finished is `Indeterminate` —
`retry_running=False` surfaces that instead of guessing, because only the caller knows whether
repeating that particular effect is safe.

What recovery does *not* do is replay the model turn. The tool calls are already in the transcript;
re-deriving them would cost tokens and might produce different calls.
"""

import asyncio

from _scripted import Files, calling, says, scripted
from langchain_core.messages import AIMessage, HumanMessage
from nexora import AgentRuntime
from nexora.orchestrator import MemorySteps, Orchestrator


async def main() -> None:
    store = MemorySteps()
    files = Files()
    requested = [calling("c1", "write", path="a.md", text="one"), calling("c2", "write",
                                                                         path="b.md", text="two")]

    # The transcript the dead worker had already committed: a prompt and the model's two calls.
    transcript = [HumanMessage("write both files"), AIMessage(content="", tool_calls=requested)]

    # ── the crash: `aborted()` trips once the first write is committed ───────
    crashed = Orchestrator("run-4", store)
    await crashed.execute_round(files, requested, lambda: files.ran == ["write"])
    print(f"before crash  tools ran {files.ran}, files {sorted(files.contents)}")

    # ── a new attempt, with a model that would answer differently if asked ───
    never_asked = scripted(says("I have no idea what happened."))
    outcome = await AgentRuntime(store=store).recover(
        "run-4", transcript, never_asked, files, retry_running=False
    )

    print(f"after recover tools ran {files.ran}  ← 'write' twice, not three times")
    print(f"              files {sorted(files.contents)}")
    print(f"              answer {outcome['content']!r}")
    print("              the recovery model was consulted once, for the answer only")

    assert files.ran == ["write", "write"], "the committed write must not run a second time"
    assert sorted(files.contents) == ["a.md", "b.md", "notes.md"], "c2 still had to run"


if __name__ == "__main__":
    asyncio.run(main())
