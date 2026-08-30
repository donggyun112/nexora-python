"""Provider-neutral deferred tool exposure.

The model initially receives complete schemas only for ordinary tools and ``tool_search``. Deferred
tools are announced by name in that tool's description; selecting/searching one records a compact
activation marker in the transcript, from which a fresh wrapper can reconstruct the active set.
"""

from __future__ import annotations

import builtins
import json
import re
from collections.abc import Mapping
from inspect import isawaitable
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from .contracts import DynamicTools, Tools

__all__ = ["DeferredTools"]

_TOOL_NAME = "tool_search"
_ACTIVATED = re.compile(r"<activated_tools>(.*?)</activated_tools>", re.DOTALL)


class DeferredTools:
    """Hide selected tool schemas until ``tool_search`` activates them."""

    def __init__(
        self,
        inner: Tools,
        *,
        deferred: set[str] | None = None,
        initially_active: set[str] | None = None,
        _discovered: set[str] | None = None,
    ) -> None:
        """Configure explicit deferrals and session-local activation state."""
        if any(definition.get("name") == _TOOL_NAME for definition in inner.list()):
            raise ValueError(f"the wrapped tool collection already defines {_TOOL_NAME!r}")
        self._inner = inner
        self._explicit = frozenset(deferred or ())
        self._initially_active = frozenset(initially_active or ())
        self._discovered = _discovered if _discovered is not None else set()

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Search for a schema or execute an already exposed tool."""
        if name == _TOOL_NAME:
            return self._search(arguments)
        definition = self._definition(name)
        if definition is None:
            return {"type": "error", "message": f"tool is not available: {name}"}
        if self._is_deferred(definition) and name not in self._active_names():
            return {
                "type": "error",
                "message": f"tool {name!r} is deferred; load it with {_TOOL_NAME} first",
            }
        return await self._inner.execute(name, call_id, arguments)

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a definition only when the model is allowed to call it."""
        if name == _TOOL_NAME:
            return self._search_definition() if self._deferred_definitions() else None
        definition = self._definition(name)
        if definition is None:
            return None
        if self._is_deferred(definition) and name not in self._active_names():
            return None
        return definition

    def list(self) -> builtins.list[dict[str, Any]]:
        """List non-deferred and previously discovered schemas."""
        active = self._active_names()
        definitions = [
            definition
            for definition in self._inner.list()
            if not self._is_deferred(definition) or str(definition.get("name")) in active
        ]
        if self._deferred_definitions():
            definitions.append(self._search_definition())
        return definitions

    async def prepare(self, messages: builtins.list[BaseMessage]) -> None:
        """Recover activated names from durable tool results in the transcript."""
        if isinstance(self._inner, DynamicTools):
            prepared = self._inner.prepare(messages)
            if isawaitable(prepared):
                await prepared
        available = {str(definition.get("name")) for definition in self._inner.list()}
        for message in messages:
            if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
                continue
            for match in _ACTIVATED.finditer(message.content):
                try:
                    names = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(names, list):
                    self._discovered.update(
                        name for name in names if isinstance(name, str) and name in available
                    )

    def reset(self) -> None:
        """Clear session-local discoveries while retaining explicit initial tools."""
        self._discovered.clear()

    def get_context(self) -> Any:
        """Forward the workspace context when the wrapped collection supports it."""
        get_context = getattr(self._inner, "get_context", None)
        if get_context is None:
            raise TypeError("wrapped tools do not expose a workspace context")
        return get_context()

    def with_context(self, context: Any) -> DeferredTools:
        """Rebind the wrapped tools and retain this conversation's discoveries."""
        with_context = getattr(self._inner, "with_context", None)
        if with_context is None:
            raise TypeError("wrapped tools cannot be rebound to a workspace context")
        return DeferredTools(
            with_context(context),
            deferred=set(self._explicit),
            initially_active=set(self._initially_active),
            _discovered=self._discovered,
        )

    def _search(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, Mapping) or not isinstance(arguments.get("query"), str):
            return {"type": "error", "message": "tool_search requires a string 'query'"}
        query = str(arguments["query"]).strip()
        raw_limit = arguments.get("max_results", 5)
        if not isinstance(raw_limit, int) or isinstance(raw_limit, bool):
            return {"type": "error", "message": "tool_search 'max_results' must be an integer"}
        maximum = min(max(raw_limit, 1), 20)
        matches = self._matches(query, maximum)
        self._discovered.update(matches)
        encoded = json.dumps(matches, separators=(",", ":"), ensure_ascii=False)
        visible = (
            "Activated deferred tools:\n" + "\n".join(f"- {name}" for name in matches)
            if matches
            else "No matching deferred tools found."
        )
        return {"type": "text", "text": f"{visible}\n<activated_tools>{encoded}</activated_tools>"}

    def _matches(self, query: str, maximum: int) -> builtins.list[str]:
        definitions = self._deferred_definitions()
        by_name = {str(definition.get("name")): definition for definition in definitions}
        if query.lower().startswith("select:"):
            requested = [item.strip() for item in query[7:].split(",") if item.strip()]
            lowered = {name.lower(): name for name in by_name}
            selected = dict.fromkeys(
                lowered[item.lower()] for item in requested if item.lower() in lowered
            )
            return list(selected)[:maximum]

        terms = [term.lower() for term in query.split() if term]
        required = [term[1:] for term in terms if term.startswith("+") and len(term) > 1]
        optional = [term for term in terms if not term.startswith("+")]
        scored: builtins.list[tuple[int, str]] = []
        for name, definition in by_name.items():
            haystack = " ".join(
                (
                    name,
                    str(definition.get("description", "")),
                    str(definition.get("search_hint", "")),
                )
            ).lower()
            if required and not all(term in haystack for term in required):
                continue
            score = sum(
                10 if term in name.lower() else 2
                for term in [*required, *optional]
                if term in haystack
            )
            if score:
                scored.append((score, name))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))[:maximum]
        return [name for _score, name in ranked]

    def _active_names(self) -> set[str]:
        return set(self._initially_active) | self._discovered

    def _definition(self, name: str) -> dict[str, Any] | None:
        return next(
            (definition for definition in self._inner.list() if definition.get("name") == name),
            None,
        )

    def _deferred_definitions(self) -> builtins.list[dict[str, Any]]:
        return [definition for definition in self._inner.list() if self._is_deferred(definition)]

    def _is_deferred(self, definition: dict[str, Any]) -> bool:
        if definition.get("always_load") is True:
            return False
        name = str(definition.get("name", ""))
        return bool(
            name in self._explicit
            or name.startswith("mcp__")
            or definition.get("is_mcp") is True
            or definition.get("should_defer") is True
            or definition.get("defer_loading") is True
        )

    def _search_definition(self) -> dict[str, Any]:
        names = sorted(str(definition.get("name")) for definition in self._deferred_definitions())
        return {
            "name": _TOOL_NAME,
            "description": (
                "Load complete schemas for deferred tools. Until selected, only these names are "
                "known and the tools cannot be called. Use 'select:<name>' for exact selection "
                "or capability keywords to search.\n\n<available_deferred_tools>\n"
                + "\n".join(names)
                + "\n</available_deferred_tools>"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "is_exclusive": True,
            "is_concurrency_safe": True,
        }
