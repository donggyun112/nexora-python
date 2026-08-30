"""Progressive prompt, skill, and tool disclosure semantics."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from semora.engines.plain import react_loop
from semora.prompts import SystemPrompt, prompt_section, volatile_prompt_section
from semora.skills import (
    DirectorySkillSource,
    Skill,
    SkillMetadata,
    SkillRegistry,
    SkillTools,
)
from semora.tool_search import DeferredTools

from tests.test_loop import Llm, Tools, a_call, says, scripted


class BindingLlm(Llm):
    """Record each round's model-visible function names."""

    bound_names: list[list[str]] = []  # noqa: RUF012 - pydantic fake, set per instance
    bound_tools: list[list[dict[str, Any]]] = []  # noqa: RUF012 - pydantic fake

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Keep the fake stream behavior while recording normalized schemas."""
        definitions = list(tools)
        self.bound_names.append(
            [
                str(tool.get("function", {}).get("name", tool.get("name", "")))
                for tool in definitions
            ]
        )
        self.bound_tools.append(definitions)
        return self


class StoredSkillSource:
    """Expose DB-shaped metadata and record full-body fetches."""

    def __init__(self, skill: Skill) -> None:
        self.skill = skill
        self.loads: list[str] = []

    async def list(self) -> list[SkillMetadata]:
        return [SkillMetadata(self.skill.name, self.skill.description, revision="42")]

    async def load(self, name: str) -> Skill | None:
        self.loads.append(name)
        return self.skill if name == self.skill.name else None


def write_skill(root: Path, name: str, description: str, body: str) -> None:
    """Write the directory-form skill layout supported by the runtime."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}",
        encoding="utf-8",
    )


async def test_a_cached_prompt_section_computes_once_across_model_rounds() -> None:
    """Claude Code `systemPromptSection` — ordinary sections survive model rounds."""
    computations = 0

    def identity() -> str:
        nonlocal computations
        computations += 1
        return "stable identity"

    prompt = SystemPrompt([prompt_section("identity", identity)])
    model = scripted(says("", a_call("c1", "read")), says("done"))

    async for _ in react_loop(model, Tools(), system_prompt=prompt):
        pass

    assert computations == 1


async def test_a_volatile_prompt_section_reaches_each_model_round() -> None:
    """Claude Code `DANGEROUS_uncachedSystemPromptSection` — changes are model-visible."""
    computations = 0

    def runtime() -> str:
        nonlocal computations
        computations += 1
        return f"runtime {computations}"

    prompt = SystemPrompt(
        [volatile_prompt_section("runtime", runtime, reason="the runtime changed")]
    )
    model = scripted(says("", a_call("c1", "read")), says("done"))

    async for _ in react_loop(model, Tools(), system_prompt=prompt):
        pass

    assert [str(turn[0].content) for turn in model.seen] == ["runtime 1", "runtime 2"]


async def test_a_later_skill_source_overrides_an_earlier_one(tmp_path: Path) -> None:
    """Claude Code `getSkillDirCommands` — source precedence selects one definition."""
    user = tmp_path / "user"
    project = tmp_path / "project"
    write_skill(user, "review", "user version", "old procedure")
    write_skill(project, "review", "project version", "new procedure")

    registry = SkillRegistry([DirectorySkillSource(user), DirectorySkillSource(project)])
    skill = await registry.load("review")

    assert skill is not None
    assert skill.body == "new procedure"


async def test_the_skill_catalog_does_not_disclose_the_body(tmp_path: Path) -> None:
    """Claude Code `formatCommandsWithinBudget` — listings are discovery metadata only."""
    write_skill(tmp_path, "review", "Review a change", "SECRET FULL PROCEDURE")

    catalog = await SkillRegistry([DirectorySkillSource(tmp_path)]).catalog()

    assert "review" in catalog and "Review a change" in catalog
    assert "SECRET FULL PROCEDURE" not in catalog


async def test_catalog_discovery_does_not_load_a_database_skill_body() -> None:
    """`SkillSource.list` is the metadata boundary; discovery must not call `load`."""
    source = StoredSkillSource(Skill("review", "Review a change", "SECRET"))

    await SkillRegistry([source]).catalog()

    assert source.loads == []


async def test_the_skill_catalog_appears_once_in_the_model_request() -> None:
    """Claude Code `SkillTool` — its schema alone owns catalog disclosure."""
    description = "Review a change without disclosing the complete procedure"
    source = StoredSkillSource(Skill("review", description, "SECRET"))
    model = BindingLlm(messages=iter((says("done"),)))
    model.seen = []
    model.bound_names = []
    model.bound_tools = []

    async for _ in react_loop(
        model,
        SkillTools(Tools(), SkillRegistry([source])),
        system_prompt=SystemPrompt([prompt_section("identity", "You are Semora.")]),
    ):
        pass

    visible = "\n".join(str(message.content) for message in model.seen[0])
    visible += json.dumps(model.bound_tools[0])
    assert visible.count(description) == 1


async def test_invoking_a_skill_injects_its_body_after_the_tool_answer() -> None:
    """Claude Code `SkillTool.call` — full instructions enter context only on invocation."""
    source = StoredSkillSource(Skill("review", "Review a change", "FOLLOW THIS PROCEDURE"))
    tools = SkillTools(Tools(), SkillRegistry([source]))
    model = scripted(says("", a_call("s1", "skill", {"skill": "review"})), says("done"))

    async for _ in react_loop(model, tools):
        pass

    first = "\n".join(str(message.content) for message in model.seen[0])
    second = "\n".join(str(message.content) for message in model.seen[1])
    assert "FOLLOW THIS PROCEDURE" not in first
    assert "FOLLOW THIS PROCEDURE" in second
    assert any(isinstance(message, HumanMessage) for message in model.seen[1])
    assert source.loads == ["review"]


async def test_deferred_schema_appears_only_after_tool_search() -> None:
    """Claude Code API filtering — a discovered deferred tool joins the next request."""
    tools = DeferredTools(
        Tools(names=["read", "mcp__github__issue"]),
        deferred={"mcp__github__issue"},
    )
    model = BindingLlm(
        messages=iter(
            (
                says("", a_call("search-1", "tool_search", {"query": "select:mcp__github__issue"})),
                says("done"),
            )
        )
    )
    model.seen = []
    model.bound_names = []
    model.bound_tools = []

    async for _ in react_loop(model, tools):
        pass

    assert model.bound_names == [
        ["read", "tool_search"],
        ["mcp__github__issue", "read", "tool_search"],
    ]


async def test_a_fresh_deferred_wrapper_recovers_tool_references_from_history() -> None:
    """Claude Code `extractDiscoveredToolNames` — transcript history restores exposure."""
    original = DeferredTools(Tools(names=["mcp__github__issue"]))
    result = await original.execute(
        "tool_search", "search-1", {"query": "select:mcp__github__issue"}
    )
    recovered = DeferredTools(Tools(names=["mcp__github__issue"]))

    await recovered.prepare([ToolMessage(str(result["text"]), tool_call_id="search-1")])

    assert [definition["name"] for definition in recovered.list()] == [
        "mcp__github__issue",
        "tool_search",
    ]
