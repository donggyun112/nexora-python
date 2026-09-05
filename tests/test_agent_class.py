"""A class is an agent.

`__init__` assembles it, methods are its tools and its default policy, and the runtime it runs
under is on the class.
"""

import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from semora import Agent, AgentRuntime, ControlPlane, MemorySteps, Outcome, Prompt, tool
from semora.controls import (
    Continue,
    Ctx,
    Deny,
    Permissions,
    ResumeInput,
    Suspend,
    ToolDecision,
)
from semora_store import MemoryTranscript


async def web_search(query: str) -> str:
    """An implementation written elsewhere."""
    return f"results for {query}"


async def run_command(ctx: RunContext[None], command: str) -> str:
    """A toolset tool that takes the run context."""
    return f"ran {command}"


shell = FunctionToolset(tools=[run_command])

Round = list[tuple[str, dict[str, str]]]


def scripted(*rounds: Round) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = sum(isinstance(m, ModelResponse) for m in messages)
        if index < len(rounds):
            return ModelResponse(
                parts=[
                    ToolCallPart(name, args, tool_call_id=f"c{index}{n}")
                    for n, (name, args) in enumerate(rounds[index])
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(model)


class Reviewer(Agent):
    """Reviews a repository. The class body is the whole definition."""

    uses = (web_search, shell)

    def __init__(
        self,
        repo: Path,
        *rounds: Round,
        runtime: AgentRuntime | None = None,
        run_id: str | None = None,
    ) -> None:
        self.repo = repo
        self.touched: list[str] = []
        self.asked: list[tuple[str, str]] = []
        self.llm = scripted(*rounds)
        super().__init__(runtime=runtime, run_id=run_id)

    def prompt(self) -> str:
        return f"Review the repository at {self.repo}. Read before you judge."

    @tool(concurrency_safe=True)
    async def read(self, path: str) -> str:
        """Read one file."""
        return f"<{path}>"

    @tool
    async def write(self, path: str, text: str) -> str:
        """Write one file. An effect."""
        self.touched.append(path)
        return f"wrote {path}"

    async def pre_tool_use(self, ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        args = call.args_as_dict()
        if call.tool_name == "write" and args["path"].endswith(".env"):
            return Deny("policy: .env is protected")
        if call.tool_name == "write":
            return Suspend({"pending_id": f"approve-{call.tool_call_id}", "path": args["path"]})
        return Continue()

    async def on_resume(self, ctx: Ctx, call: ToolCallPart, resume: ResumeInput) -> ToolDecision:
        self.asked.append((resume.suspended_rules_version, resume.current_rules_version))
        if resume.current_rules_version != resume.suspended_rules_version:
            return Deny("policy changed while you decided")
        return Continue()


async def test_own_methods_and_external_implementations_share_one_ledger() -> None:
    store = MemorySteps()
    agent = Reviewer(
        Path("/repo"),
        [
            ("read", {"path": "a"}),
            ("web_search", {"query": "x"}),
            ("run_command", {"command": "ls"}),
        ],
        runtime=AgentRuntime(store),
    )

    outcome = await agent.run("review", run_id="run-1")

    assert outcome.output == "done"
    assert [(await store.read("run-1", f"tool:c0{n}")).value["value"] for n in range(3)] == [
        "<a>",
        "results for x",
        "ran ls",
    ]


async def test_the_default_runtime_records_nothing_and_still_gates() -> None:
    agent = Reviewer(Path("/repo"), [("write", {"path": ".env", "text": ""})])

    outcome = await agent.run("clear .env")  # no run id, no ledger: Pydantic AI plus the policy

    assert agent.touched == [], "the class's pre_tool_use denied it"
    assert outcome.output == "done"


async def test_a_suspension_parks_and_the_class_re_decides_on_resume() -> None:
    store, transcript = MemorySteps(), MemoryTranscript()
    runtime = AgentRuntime(store, transcript=transcript)
    agent = Reviewer(
        Path("/repo"), [("write", {"path": "notes.md", "text": "hi"})], runtime=runtime
    )

    parked = await agent.run("write notes", run_id="run-3", rules_version="v1")
    assert parked.suspended and agent.touched == []

    # Another process, hours later: a fresh instance of the same class, bound to the same run.
    later = Reviewer(
        Path("/repo"), runtime=AgentRuntime(store, transcript=transcript), run_id="run-3"
    )
    outcome = await later.resume({"type": "approve"}, rules_version="v2")

    assert later.asked == [("v1", "v2")]
    assert later.touched == [], "the rules changed while the person decided; the class refused"
    assert outcome.output == "done"


async def test_an_explicit_control_plane_replaces_the_class_policy() -> None:
    agent = Reviewer(
        Path("/repo"),
        [("write", {"path": "notes.md", "text": "hi"})],
        runtime=AgentRuntime(MemorySteps()),
    )

    outcome = await agent.run("write notes", run_id="run-4", controls=ControlPlane())

    assert agent.touched == ["notes.md"], "no gate: the class's Suspend never ran"
    assert outcome.output == "done"


async def test_the_store_declared_in_the_class_body_is_the_ledger() -> None:
    class Durable(Reviewer):
        store = MemorySteps()

    agent = Durable(Path("/repo"), [("read", {"path": "a"})])
    outcome = await agent.run("go", run_id="run-5")

    assert outcome.output == "done"
    assert (await Durable.store.read("run-5", "tool:c00")).status == "done"


async def test_the_class_body_is_what_the_model_sees() -> None:
    agent = Reviewer(Path("/repo"))

    assert agent.name == "Reviewer"
    assert agent.description == "Reviews a repository. The class body is the whole definition."
    names = {t.name for t in agent._function_toolset.tools.values()}
    assert {"read", "write", "web_search"} <= names


async def test_a_store_factory_is_called_once_per_class_on_first_use() -> None:
    made: list[MemorySteps] = []

    def open_store() -> MemorySteps:
        made.append(MemorySteps())
        return made[-1]

    class Lazy(Reviewer):
        store = staticmethod(open_store)

    assert made == [], "declaring the class opened nothing"
    first = Lazy(Path("/repo"), [("read", {"path": "a"})])
    second = Lazy(Path("/repo"), [("read", {"path": "a"})])
    await first.run("go", run_id="run-6")

    assert len(made) == 1, "one store for the class, opened when the first agent was made"
    assert (await made[0].read("run-6", "tool:c00")).status == "done"
    assert second.runtime is first.runtime


async def test_a_composed_control_can_be_a_class_attribute() -> None:
    async def no_env(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        if call.args_as_dict().get("path", "").endswith(".env"):
            return Deny("policy: .env is protected")
        return Continue()

    async def ask_writes(ctx: Ctx, call: ToolCallPart) -> ToolDecision:
        if call.tool_name == "write":
            return Suspend({"pending_id": f"approve-{call.tool_call_id}"})
        return Continue()

    class Composed(Reviewer):
        pre_tool_use = Permissions(ask_writes, no_env)  # denial wins whatever the order

    agent = Composed(
        Path("/repo"), [("write", {"path": ".env", "text": ""})], runtime=AgentRuntime()
    )
    outcome = await agent.run("clear .env")

    assert agent.touched == [] and outcome.output == "done"


def test_uses_refuses_what_it_cannot_place() -> None:
    class Broken(Reviewer):
        uses = ("web_search",)  # type: ignore[assignment]  # deliberately wrong

    with pytest.raises(TypeError, match="uses holds"):
        Broken(Path("/repo"))


class Deployer(Reviewer):
    """A tool declared `requires_approval=True`, and no gate that says anything about it."""

    @tool(requires_approval=True)
    async def deploy(self, target: str) -> str:
        """Ship it. An effect a person must approve."""
        self.touched.append(target)
        return f"deployed {target}"


async def test_requires_approval_parks_through_the_gate_and_resumes() -> None:
    """Pydantic AI's own deferral carries no pending_id; routed through the gate it does."""
    store = MemorySteps()
    agent = Deployer(Path("/repo"), [("deploy", {"target": "prod"})], runtime=AgentRuntime(store))

    parked = await agent.run("ship", run_id="run-7", rules_version="v1")

    assert parked.suspended and parked.pending == (("c00", "c00"),)
    assert agent.touched == [], "parked before the effect"

    outcome = await agent.resume({"type": "approve"}, rules_version="v1")

    assert agent.asked == [("v1", "v1")], "the answer went through on_resume, not around it"
    assert agent.touched == ["prod"] and outcome.output == "done"
    assert (await store.read("run-7", "tool:c00")).value["value"] == "deployed prod"


def test_run_sync_drives_the_runtime() -> None:
    """Pydantic AI's run_sync forwards output_type=None; that must not collide with ours."""
    store = MemorySteps()
    agent = Reviewer(Path("/repo"), [("read", {"path": "a"})], runtime=AgentRuntime(store))

    outcome = cast(Outcome, agent.run_sync("go", run_id="run-8"))

    assert outcome.output == "done" and agent.last is outcome
    assert asyncio.run(store.read("run-8", "tool:c00")).status == "done"


async def test_a_queued_prompt_is_a_receipt_not_the_last_outcome() -> None:
    """A dispatch behind a live lease returns an enqueue receipt; `last` stays an Outcome."""
    store = MemorySteps()
    runtime = AgentRuntime(store, transcript=MemoryTranscript())
    agent = Reviewer(Path("/repo"), runtime=runtime, run_id="run-9")
    await store.acquire("run-9", "another-worker", 60)

    receipt = await agent.dispatch(Prompt("also this", prompt_id="p1"))

    assert receipt == {"type": "enqueued", "input_id": "p1"}
    assert agent.last is None


async def test_an_instance_is_one_run() -> None:
    agent = Reviewer(Path("/repo"), [("read", {"path": "a"})], runtime=AgentRuntime(MemorySteps()))

    first = await agent.run("go")
    assert first.output == "done" and agent.run_id is not None and agent.last is first

    with pytest.raises(RuntimeError, match="bound to run"):
        await agent.run("again", run_id="another")
