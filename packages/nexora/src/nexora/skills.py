"""Source-neutral skill discovery and progressive disclosure.

Sources list lightweight metadata and load full instructions only when the ``skill`` tool is
invoked.  Filesystem discovery is one adapter, not a registry responsibility.
"""

from __future__ import annotations

import asyncio
import builtins
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from inspect import isawaitable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import BaseMessage, DynamicTools, Tools

__all__ = [
    "DirectorySkillSource",
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    "SkillSource",
    "SkillTools",
]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_DEFAULT_CATALOG_BUDGET = 8_000
_DESCRIPTION_LIMIT = 250


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """The model-visible part of a skill returned by source discovery."""

    name: str
    description: str
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class Skill:
    """Full instructions loaded on demand from an arbitrary source."""

    name: str
    description: str
    body: str
    origin: str | None = None
    resource_base: str | None = None
    allowed_tools: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    def context(self, arguments: str = "") -> str:
        """Render the full on-demand context injected after the tool result."""
        body = self.body
        prefix = ""
        if self.resource_base is not None:
            body = body.replace("${NEXORA_SKILL_ROOT}", self.resource_base)
            body = body.replace("${NEXORA_SKILL_DIR}", self.resource_base)
            prefix = f"Resource base for this skill: {self.resource_base}\n\n"
        body = body.replace("${ARGUMENTS}", arguments).replace("$ARGUMENTS", arguments)
        argument_block = f"\n\nArguments: {arguments}" if arguments else ""
        return f"{prefix}{body}{argument_block}"


@runtime_checkable
class SkillSource(Protocol):
    """Metadata-first store for directory, database, API, or package-backed skills."""

    async def list(self) -> Sequence[SkillMetadata]:
        """List discovery metadata without loading complete skill bodies."""
        ...

    async def load(self, name: str) -> Skill | None:
        """Load one exact skill body, or return ``None`` if it disappeared."""
        ...


class DirectorySkillSource:
    """Read directory-form ``SKILL.md`` files through the generic source contract."""

    def __init__(self, root: str | Path) -> None:
        """Resolve the root once; source callers remain responsible for user expansion."""
        self.root = Path(root).resolve()
        self._locations: dict[str, Path] | None = None

    async def list(self) -> Sequence[SkillMetadata]:
        """Read frontmatter only; instruction bodies remain unread until ``load``."""
        metadata, locations = await asyncio.to_thread(self._discover)
        self._locations = locations
        return metadata

    async def load(self, name: str) -> Skill | None:
        """Read and parse one previously discovered skill file."""
        if self._locations is None:
            await self.list()
        assert self._locations is not None
        path = self._locations.get(name)
        if path is None:
            return None
        try:
            return await asyncio.to_thread(_parse_skill, path)
        except (OSError, ValueError):
            return None

    def _discover(self) -> tuple[tuple[SkillMetadata, ...], dict[str, Path]]:
        found: dict[str, SkillMetadata] = {}
        locations: dict[str, Path] = {}
        if not self.root.is_dir():
            return (), locations
        for root, directories, files in os.walk(self.root, followlinks=False):
            directories[:] = sorted(
                directory for directory in directories if not (Path(root) / directory).is_symlink()
            )
            if "SKILL.md" not in files:
                continue
            path = Path(root) / "SKILL.md"
            if path.is_symlink():
                continue
            try:
                metadata = _read_metadata(path)
            except (OSError, ValueError):
                continue
            found[metadata.name] = metadata
            locations[metadata.name] = path.resolve()
        return tuple(found[name] for name in sorted(found)), locations


class SkillRegistry:
    """Merge ordered metadata sources; a later source overrides an earlier one by name."""

    def __init__(
        self,
        sources: Sequence[SkillSource | str | Path],
        *,
        catalog_char_budget: int = _DEFAULT_CATALOG_BUDGET,
    ) -> None:
        """Configure source precedence and the maximum model-visible catalog size."""
        if catalog_char_budget < 256:
            raise ValueError("skill catalog budget must be at least 256 characters")
        self._sources = tuple(_source(source) for source in sources)
        self._catalog_char_budget = catalog_char_budget
        self._metadata: dict[str, SkillMetadata] | None = None
        self._owners: dict[str, SkillSource] = {}

    async def refresh(self) -> tuple[SkillMetadata, ...]:
        """Refresh metadata from every source without loading any instruction body."""
        found: dict[str, SkillMetadata] = {}
        owners: dict[str, SkillSource] = {}
        for source in self._sources:
            for metadata in await source.list():
                _validate_metadata(metadata)
                found[metadata.name] = metadata
                owners[metadata.name] = source
        self._metadata = found
        self._owners = owners
        return self.snapshot()

    def clear(self) -> None:
        """Forget discovery metadata so the next access refreshes every source."""
        self._metadata = None
        self._owners = {}

    async def list(self) -> tuple[SkillMetadata, ...]:
        """Return deterministic discovery metadata, refreshing once when needed."""
        if self._metadata is None:
            await self.refresh()
        return self.snapshot()

    def snapshot(self) -> tuple[SkillMetadata, ...]:
        """Return already-fetched metadata without performing source I/O."""
        if self._metadata is None:
            return ()
        return tuple(self._metadata[name] for name in sorted(self._metadata))

    async def load(self, name: str) -> Skill | None:
        """Load a full skill from the source that won metadata precedence."""
        await self.list()
        owner = self._owners.get(name)
        if owner is None:
            return None
        skill = await owner.load(name)
        if skill is None:
            return None
        if skill.name != name:
            raise ValueError(f"skill source returned {skill.name!r} for {name!r}")
        return skill

    async def catalog(self) -> str:
        """Fetch metadata and build a bounded discovery-only catalog."""
        await self.list()
        return self.catalog_snapshot()

    def catalog_snapshot(self) -> str:
        """Build a catalog from fetched metadata without source I/O."""
        skills = self.snapshot()
        if not skills:
            return ""
        head = "<available_skills>\n"
        tail = "\n</available_skills>"
        entries = [
            f"  - {escape(skill.name)}: {escape(skill.description[:_DESCRIPTION_LIMIT])}"
            for skill in skills
        ]
        full = head + "\n".join(entries) + tail
        if len(full) <= self._catalog_char_budget:
            return full

        names = [f"  - {escape(skill.name)}" for skill in skills]
        kept: list[str] = []
        for entry in names:
            remaining = len(skills) - len(kept) - 1
            suffix = f"\n  ... {remaining} more" if remaining > 0 else ""
            candidate = head + "\n".join([*kept, entry]) + suffix + tail
            if len(candidate) > self._catalog_char_budget:
                break
            kept.append(entry)
        omitted = len(skills) - len(kept)
        suffix = f"\n  ... {omitted} more" if omitted else ""
        return head + "\n".join(kept) + suffix + tail

class SkillTools:
    """Add an on-demand ``skill`` tool to another tool collection."""

    def __init__(self, inner: Tools, registry: SkillRegistry) -> None:
        """Compose the registry over an existing tool collection."""
        if any(definition.get("name") == "skill" for definition in inner.list()):
            raise ValueError("the wrapped tool collection already defines 'skill'")
        self._inner = inner
        self._registry = registry

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Load a skill or delegate an ordinary tool call."""
        if name != "skill":
            return await self._inner.execute(name, call_id, arguments)
        if not isinstance(arguments, Mapping) or not isinstance(arguments.get("skill"), str):
            return {"type": "error", "message": "skill requires a string 'skill' argument"}
        skill_name = str(arguments["skill"]).removeprefix("/").strip()
        skill = await self._registry.load(skill_name)
        if skill is None:
            return {"type": "error", "message": f"unknown skill: {skill_name}"}
        raw_args = arguments.get("args", "")
        if not isinstance(raw_args, str):
            return {"type": "error", "message": "skill 'args' must be a string"}
        metadata: dict[str, Any] = {
            "kind": "skill",
            "name": skill.name,
            "allowed_tools": list(skill.allowed_tools),
        }
        if skill.origin is not None:
            metadata["origin"] = skill.origin
        return {
            "type": "text",
            "text": f"Loaded skill {skill.name}.",
            "context_messages": [
                {
                    "content": skill.context(raw_args),
                    "metadata": metadata,
                }
            ],
        }

    def get(self, name: str) -> dict[str, Any] | None:
        """Return one currently available tool definition."""
        if name == "skill":
            return self._definition()
        return self._inner.get(name)

    def list(self) -> list[dict[str, Any]]:
        """Expose ordinary tools plus the fetched skill discovery schema."""
        return [*self._inner.list(), self._definition()]

    async def prepare(self, messages: builtins.list[BaseMessage]) -> None:
        """Fetch skill metadata and forward dynamic exposure reconstruction."""
        await self._registry.list()
        if isinstance(self._inner, DynamicTools):
            prepared = self._inner.prepare(messages)
            if isawaitable(prepared):
                await prepared

    def get_context(self) -> Any:
        """Forward the workspace context when the wrapped collection supports it."""
        get_context = getattr(self._inner, "get_context", None)
        if get_context is None:
            raise TypeError("wrapped tools do not expose a workspace context")
        return get_context()

    def with_context(self, context: Any) -> SkillTools:
        """Rebind the wrapped collection without rebuilding the skill registry."""
        with_context = getattr(self._inner, "with_context", None)
        if with_context is None:
            raise TypeError("wrapped tools cannot be rebound to a workspace context")
        return SkillTools(with_context(context), self._registry)

    def _definition(self) -> dict[str, Any]:
        catalog = self._registry.catalog_snapshot()
        return {
            "name": "skill",
            "description": (
                "Load a matching skill before doing the task. The tool injects the full "
                "instructions into the next model round; do not guess unlisted names.\n\n"
                f"{catalog}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "Exact skill name"},
                    "args": {"type": "string", "description": "Optional skill arguments"},
                },
                "required": ["skill"],
                "additionalProperties": False,
            },
            "is_exclusive": True,
        }


def _source(source: SkillSource | str | Path) -> SkillSource:
    if isinstance(source, str | Path):
        return DirectorySkillSource(source)
    return source


def _validate_metadata(metadata: SkillMetadata) -> None:
    if not _NAME.fullmatch(metadata.name):
        raise ValueError(f"invalid skill name: {metadata.name!r}")


def _read_metadata(path: Path) -> SkillMetadata:
    lines: list[str] = []
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        if first.strip() != "---":
            raise ValueError("SKILL.md has no YAML frontmatter")
        for line in stream:
            if line.strip() == "---":
                break
            lines.append(line.rstrip("\n"))
        else:
            raise ValueError("SKILL.md frontmatter is not closed")
    frontmatter = _frontmatter_values(lines)
    name = str(frontmatter.get("name") or path.parent.name).strip()
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")
    return SkillMetadata(
        name=name,
        description=str(frontmatter.get("description") or "").strip(),
        revision=str(path.stat().st_mtime_ns),
    )


def _parse_skill(path: Path) -> Skill:
    content = path.read_text(encoding="utf-8")
    frontmatter, body = _frontmatter(content)
    name = str(frontmatter.get("name") or path.parent.name).strip()
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")
    description = str(frontmatter.get("description") or "").strip()
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        origin=path.resolve().as_uri(),
        resource_base=str(path.resolve().parent),
        allowed_tools=_string_tuple(frontmatter.get("allowed-tools")),
        paths=_string_tuple(frontmatter.get("paths")),
    )


def _frontmatter(content: str) -> tuple[dict[str, Any], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md has no YAML frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as error:
        raise ValueError("SKILL.md frontmatter is not closed") from error
    return _frontmatter_values(lines[1:end]), "\n".join(lines[end + 1 :])


def _frontmatter_values(lines: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_list: str | None = None
    for line in lines:
        stripped = line.strip()
        if current_list is not None and stripped.startswith("- "):
            value = values[current_list]
            assert isinstance(value, list)
            value.append(_scalar(stripped[2:]))
            continue
        if ":" not in line:
            current_list = None
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if not raw:
            values[key] = []
            current_list = key
            continue
        current_list = None
        values[key] = _scalar(raw)
    return values


def _scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_scalar(item) for item in inner.split(",")]
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)
