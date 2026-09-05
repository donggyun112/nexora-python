"""A class is an agent, and an instance is one run. One `main` carries a run from any state.

    uv run python examples/reviewer.py

`main(branch_id)` looks at the run's durable state and takes the next step: start it, finish a round
a dead worker left behind, or resume a parked call with a person's answer. Call it again after a
crash, in another process, and it picks up where the ledger says. No API key: the model is
scripted.
"""

import asyncio
import contextlib
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from semora import Agent, MemorySteps, MemoryTranscript, tool
from semora.controls import Continue, Ctx, ResumeInput, Suspend, ToolDecision


def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart("run_tests", {}, tool_call_id="c1"),
                ToolCallPart("write", {"path": "REVIEW.md", "text": "LGTM"}, tool_call_id="c2"),
            ]
        )
    return ModelResponse(parts=[TextPart("Reviewed. Tests pass; notes are in REVIEW.md.")])


class Reviewer(Agent):
    """Reviews a repository: runs the tests, then asks before it writes."""

    llm = FunctionModel(scripted)  # swap for "openai:gpt-5"
    store = MemorySteps()  # the ledger: every effect once, a gate may park the run
    transcript = MemoryTranscript()  # the committed conversation
    retry_running = True  # a test run that started and never reported may be run again

    def __init__(self, repo: Path, **kwargs: object) -> None:
        self.repo = repo
        self.touched: list[str] = []
        self.hold: asyncio.Event | None = None  # the example's crash switch
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def prompt(self) -> str:
        return f"Review the repository at {self.repo}. Run the tests before you judge."

    @tool
    async def run_tests(self) -> str:
        """Run the test suite. Slow."""
        if self.hold is not None:
            await self.hold.wait()
        return "42 passed"

    @tool
    async def write(self, path: str, text: str) -> str:
        """Write one file. An effect: it happens once, or a person is asked first."""
        self.touched.append(path)
        return f"wrote {path}"

    async def pre_tool_use(self, ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        if call.tool_name == "write":
            return Suspend({"pending_id": f"approve-{call.tool_call_id}"})
        return Continue()

    async def on_resume(self, ctx: Ctx, call: ToolCallPart, resume: ResumeInput) -> ToolDecision:
        return Continue()  # the answer is an input; the rules in force now decide


async def ask_person(pending_id: str) -> dict[str, str]:
    print(f"  a person is asked about {pending_id} ... approved")
    return {"type": "approve"}


async def main(branch_id: str, reviewer: Reviewer | None = None) -> None:
    """Carry the run one step further, from whatever state it is in."""
    reviewer = reviewer or Reviewer(Path("."), branch_id=branch_id)
    state = await reviewer.state()
    print(f"{branch_id} is {state}")

    match state:
        case "interrupted":  # a worker died mid-round
            outcome = await reviewer.recover()  # committed calls replay; the rest run
        case "waiting":  # a person's answer is owed
            outcome = await reviewer.resume(await ask_person((await reviewer.pending())[0][0]))
        case _:  # fresh, or completed and continuing the conversation
            outcome = await reviewer.run("review this repository")

    while outcome.suspended:  # a park is an outcome; answer it and go on
        outcome = await reviewer.resume(await ask_person(outcome.pending_id or ""))

    print(f"  answer: {outcome.output!r}; written: {reviewer.touched}")


async def demo() -> None:
    # Process A starts the run and dies while the tests are running.
    doomed = Reviewer(Path("."), branch_id="review-1")
    doomed.hold = asyncio.Event()
    worker = asyncio.create_task(main("review-1", doomed))
    while (await Reviewer.store.read("review-1", "tool:c1")).status != "running":
        await asyncio.sleep(0)
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    print("  ... process A died with c1 running\n")

    # Process B calls the same main. It recovers the round, parks on the write, resumes, answers.
    await main("review-1")


if __name__ == "__main__":
    asyncio.run(demo())
