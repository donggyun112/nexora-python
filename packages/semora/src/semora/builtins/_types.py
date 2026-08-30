"""Shared contracts and state for Semora's built-in tools."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..workspace import ToolContext, WorkspaceSession

ToolResult = dict[str, object]


def text_result(text: str) -> ToolResult:
    """Build a text tool result."""
    return {"type": "text", "text": text}


def error_result(message: str) -> ToolResult:
    """Build an error tool result."""
    return {"type": "error", "message": message}


def require_workspace(context: ToolContext) -> WorkspaceSession | None:
    """Return the active workspace, if the runtime supplied one."""
    return context.workspace


@dataclass(frozen=True, slots=True)
class ExecToolOptions:
    """Security and resource policy for the ``Bash`` built-in.

    An empty ``allow_list`` disables command execution. ``("*",)`` permits every bare
    executable and is intended only when the selected workspace is the real OS sandbox.
    """

    allow_list: tuple[str, ...] = ()
    allow_shell: bool = False
    env_allow_list: tuple[str, ...] = ()
    default_timeout_ms: int = 120_000
    require_isolation: bool = True
    allowed_domains: tuple[str, ...] | None = ()


class WebFetchSummarizer(Protocol):
    """Optionally apply a caller-owned model to fetched page text."""

    async def summarize(self, content: str, prompt: str) -> str:
        """Return the prompt-specific summary."""
        ...


@dataclass(frozen=True, slots=True)
class WebFetchResponse:
    """Transport-neutral HTTP response consumed by ``web_fetch``."""

    status: int
    reason: str
    url: str
    headers: Mapping[str, str]
    body: bytes


class WebFetchTransport(Protocol):
    """Minimal injectable HTTP seam for ``web_fetch``."""

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        """Fetch at most ``max_bytes`` from ``url`` and follow redirects."""
        ...


@dataclass(frozen=True, slots=True)
class WebFetchToolOptions:
    """Caching, transport, and optional summarization for ``web_fetch``."""

    transport: WebFetchTransport | None = None
    summarizer: WebFetchSummarizer | None = None
    cache_ttl_ms: int = 15 * 60 * 1000
    max_bytes: int = 5 * 1024 * 1024
    fetch_timeout_ms: int = 30_000
    now: Callable[[], float] = time.time


@dataclass(slots=True)
class BuiltinToolState:
    """State shared by context-bound copies of one built-in tool collection."""

    file_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    read_files: dict[str, tuple[int, int, int | None, int | None]] = field(default_factory=dict)
    web_cache: dict[str, tuple[float, str]] = field(default_factory=dict)
    search_engines: dict[str, str] = field(default_factory=dict)

    def file_lock(self, key: str) -> asyncio.Lock:
        """Return the stable in-process serialization lock for one file."""
        lock = self.file_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.file_locks[key] = lock
        return lock


def tool_environment(extra_names: Sequence[str]) -> dict[str, str]:
    """Build the scrubbed child environment from TS ``buildToolEnv`` semantics."""
    names = {"PATH", "HOME", "LANG", "LC_ALL", *extra_names}
    return {name: value for name in names if (value := os.environ.get(name)) is not None}


Handler = Callable[[str, object, ToolContext, BuiltinToolState], Awaitable[ToolResult]]
