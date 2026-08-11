"""Cache-stable system-prompt composition.

Claude Code's ``systemPromptSection`` is the behavioral reference: ordinary sections compute
once until explicitly cleared, while a volatile section opts into recomputation at every model
round.  The prompt stays an ordered list until rendering so callers control cache-stable prefixes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from inspect import isawaitable

type SectionResult = str | None
type SectionCompute = Callable[[], SectionResult | Awaitable[SectionResult]]

__all__ = [
    "PromptSection",
    "SystemPrompt",
    "prompt_section",
    "volatile_prompt_section",
]


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One named prompt fragment and its cache behavior."""

    name: str
    compute: SectionCompute
    volatile: bool = False


def prompt_section(name: str, value: str | SectionCompute) -> PromptSection:
    """Create a section computed once and cached until ``SystemPrompt.clear``."""
    return PromptSection(name, _compute(value))


def volatile_prompt_section(
    name: str,
    compute: SectionCompute,
    *,
    reason: str,
) -> PromptSection:
    """Create a section deliberately recomputed every round.

    ``reason`` is required because a changing system-prompt prefix invalidates provider caches.
    """
    if not reason.strip():
        raise ValueError("a volatile prompt section requires a cache-breaking reason")
    return PromptSection(name, compute, volatile=True)


class SystemPrompt:
    """Render ordered prompt sections with explicit cache invalidation."""

    def __init__(
        self,
        sections: Sequence[PromptSection],
        *,
        separator: str = "\n\n---\n\n",
    ) -> None:
        """Keep declaration order and reject cache-key collisions."""
        names = [section.name for section in sections]
        if len(set(names)) != len(names):
            raise ValueError("system prompt section names must be unique")
        self._sections = tuple(sections)
        self._separator = separator
        self._cache: dict[str, SectionResult] = {}

    async def render(self) -> str:
        """Resolve sections in declaration order and join non-empty fragments."""
        values: list[str] = []
        for section in self._sections:
            if not section.volatile and section.name in self._cache:
                value = self._cache[section.name]
            else:
                computed = section.compute()
                value = await computed if isawaitable(computed) else computed
                self._cache[section.name] = value
            if value is not None and value.strip():
                values.append(value)
        return self._separator.join(values)

    def clear(self, name: str | None = None) -> None:
        """Invalidate one section or the complete prompt cache."""
        if name is None:
            self._cache.clear()
            return
        if name not in {section.name for section in self._sections}:
            raise KeyError(name)
        self._cache.pop(name, None)


def _compute(value: str | SectionCompute) -> SectionCompute:
    if callable(value):
        return value
    return lambda: value
