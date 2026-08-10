"""TS-compatible core built-in tool bundle.

The bundle intentionally contains file/process tools plus ``web_fetch``. It does not include
``web_search``; applications can add their own search provider when they need one.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from ..workspace import ToolContext
from ._exec import exec_is_read_only, exec_tool
from ._files import edit_tool, read_tool, write_tool
from ._search import glob_tool, grep_tool
from ._types import (
    BuiltinToolState,
    ExecToolOptions,
    Handler,
    WebFetchResponse,
    WebFetchSummarizer,
    WebFetchToolOptions,
    WebFetchTransport,
    error_result,
)
from ._web import UrllibWebFetchTransport, web_fetch_tool

__all__ = [
    "BuiltinTools",
    "ExecToolOptions",
    "UrllibWebFetchTransport",
    "WebFetchResponse",
    "WebFetchSummarizer",
    "WebFetchToolOptions",
    "WebFetchTransport",
    "builtin_tools",
]


class BuiltinTools:
    """Context-aware implementation of the TS sandbox bundle plus ``web_fetch``."""

    def __init__(
        self,
        *,
        context: ToolContext | None = None,
        exec_options: ExecToolOptions | None = None,
        web_fetch_options: WebFetchToolOptions | None = None,
        _state: BuiltinToolState | None = None,
    ) -> None:
        """Configure tool policy independently from a turn's workspace session."""
        self._context = context or ToolContext(workdir=".")
        self._exec_options = exec_options or ExecToolOptions()
        self._web_fetch_options = web_fetch_options or WebFetchToolOptions()
        self._state = _state or BuiltinToolState()
        self._handlers: dict[str, Handler] = {
            "read": read_tool,
            "write": write_tool,
            "edit": edit_tool,
            "glob": glob_tool,
            "grep": grep_tool,
            "Bash": partial(exec_tool, options=self._exec_options),
            "web_fetch": partial(web_fetch_tool, options=self._web_fetch_options),
        }
        self._definitions = _definitions(self._exec_options.allow_shell)

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Execute a named built-in against the currently bound context."""
        handler = self._handlers.get(name)
        if handler is None:
            return error_result(f"Unknown tool: {name}")
        return dict(await handler(call_id, arguments, self._context, self._state))

    def get(self, name: str) -> dict[str, Any] | None:
        """Return one model-visible tool definition."""
        definition = self._definitions.get(name)
        return dict(definition) if definition is not None else None

    def list(self) -> list[dict[str, Any]]:
        """Return the stable core set, deliberately excluding ``web_search``."""
        return [dict(definition) for definition in self._definitions.values()]

    def get_context(self) -> ToolContext:
        """Return the immutable execution context currently bound to the tools."""
        return self._context

    def with_context(self, context: ToolContext) -> BuiltinTools:
        """Bind a runtime workspace while retaining locks and fetch cache across turns."""
        return BuiltinTools(
            context=context,
            exec_options=self._exec_options,
            web_fetch_options=self._web_fetch_options,
            _state=self._state,
        )


def builtin_tools(
    *,
    exec_options: ExecToolOptions | None = None,
    web_fetch_options: WebFetchToolOptions | None = None,
) -> BuiltinTools:
    """Create the standard file/process/fetch tool collection."""
    return BuiltinTools(exec_options=exec_options, web_fetch_options=web_fetch_options)


def _definitions(allow_shell: bool) -> dict[str, dict[str, Any]]:
    return {
        "read": {
            "name": "read",
            "description": (
                "Read a file or directory from the workspace. Text is line-numbered and paged; "
                "images return inline and Jupyter notebooks return readable cell text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"},
                    "pages": {"type": "string"},
                },
                "required": ["path"],
            },
            "is_read_only": True,
            "is_concurrency_safe": True,
        },
        "write": {
            "name": "write",
            "description": "Create or replace a UTF-8 file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            "is_read_only": False,
            "is_concurrency_safe": False,
        },
        "edit": {
            "name": "edit",
            "description": "Replace an exact string in a UTF-8 workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            "is_read_only": False,
            "is_concurrency_safe": False,
        },
        "grep": {
            "name": "grep",
            "description": (
                "Search workspace file contents with ripgrep, falling back to system grep. "
                "Supports content, files_with_matches, and count output modes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "type": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                    },
                    "-A": {"type": "number"},
                    "-B": {"type": "number"},
                    "-C": {"type": "number"},
                    "context": {"type": "number"},
                    "-n": {"type": "boolean"},
                    "-i": {"type": "boolean"},
                    "head_limit": {"type": "number"},
                    "offset": {"type": "number"},
                    "multiline": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
            "is_read_only": True,
            "is_concurrency_safe": True,
        },
        "glob": {
            "name": "glob",
            "description": "Find workspace files by glob pattern with ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "head_limit": {"type": "number"},
                    "offset": {"type": "number"},
                },
                "required": ["pattern"],
            },
            "is_read_only": True,
            "is_concurrency_safe": True,
        },
        "Bash": {
            "name": "Bash",
            "description": (
                "Execute an allow-listed command in the workspace. "
                + (
                    "argv is preferred; shell strings are enabled for pipes and globs."
                    if allow_shell
                    else "Shell-string mode is disabled; use argv."
                )
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "command": {"type": "string"},
                    "timeoutMs": {"type": "number"},
                    "cwd": {"type": "string"},
                    "run_in_background": {"type": "boolean"},
                },
            },
            "is_read_only": exec_is_read_only,
            "is_concurrency_safe": exec_is_read_only,
        },
        "web_fetch": {
            "name": "web_fetch",
            "description": (
                "Fetch an HTTPS URL and return readable content. HTTP is upgraded to HTTPS; "
                "results are cached by URL and prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "prompt": {"type": "string"},
                    "max_chars": {"type": "number"},
                },
                "required": ["url"],
            },
            "is_read_only": True,
            "is_concurrency_safe": True,
        },
    }
