"""Workspace-backed read, write, and edit built-ins."""

from __future__ import annotations

import base64
import json
from pathlib import PurePath
from typing import Any

from semora.workspace import ToolContext, WorkspaceViolation

from ._types import BuiltinToolState, ToolResult, error_result, require_workspace, text_result

MAX_LINES = 2_000
MAX_BYTES = 8 * 1024 * 1024
NOTEBOOK_OUTPUT_LIMIT = 10 * 1024
IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


async def read_tool(
    _call_id: str, arguments: object, context: ToolContext, state: BuiltinToolState
) -> ToolResult:
    """Read a text file, image, notebook, or directory through ``WorkspaceFS``.

    Ports ``createReadTool().execute`` and its ``numberLines`` expression.
    """
    params = arguments if isinstance(arguments, dict) else {}
    raw_path = params.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) else ""
    if not path:
        return error_result("path is required")
    workspace = require_workspace(context)
    if workspace is None:
        return error_result(
            "read requires an active workspace; configure AgentRuntime(workspace_provider=...) "
            "or pass a ToolContext to builtin_tools()"
        )

    try:
        stat = await workspace.fs.stat(path)
    except FileNotFoundError:
        return await _not_found(path, context)
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot read: {error}")

    if stat.is_directory:
        try:
            entries = await workspace.fs.readdir(path)
        except (OSError, ValueError, WorkspaceViolation) as error:
            return error_result(f"Cannot read directory: {error}")
        lines = sorted(entry.name + ("/" if entry.is_directory else "") for entry in entries)
        return text_result(f"Directory: {path}\n\n" + "\n".join(lines))
    if not stat.is_file:
        return error_result(f"Cannot read: {path} is not a regular file")

    suffix = PurePath(path).suffix.lower()
    try:
        data = await workspace.fs.read_file(path)
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot read: {error}")

    if suffix in IMAGE_MIME_BY_EXT:
        return {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": IMAGE_MIME_BY_EXT[suffix],
        }
    if suffix == ".pdf":
        return error_result(
            "Cannot read PDF: page rendering is not available in this runtime; "
            "provide a PDF-capable tool if the application needs it"
        )
    if len(data) > MAX_BYTES:
        return error_result(f"Cannot read: {path} exceeds {MAX_BYTES} byte cap ({len(data)} bytes)")
    if suffix == ".ipynb":
        return _read_notebook(path, data)

    offset = _positive_int(params.get("offset"))
    limit = _positive_int(params.get("limit"))
    key = f"{workspace.id}\0{path}"
    signature = (int(stat.mtime_ms), stat.size, offset, limit)
    if state.read_files.get(key) == signature:
        return text_result(
            f"<file unchanged since you last read it — its content is already in context: {path}>"
        )
    state.read_files[key] = signature
    return text_result(_number_lines(data.decode("utf-8", errors="replace"), offset, limit))


async def write_tool(
    _call_id: str, arguments: object, context: ToolContext, state: BuiltinToolState
) -> ToolResult:
    """Create or replace a workspace file, matching ``createWriteTool().execute``."""
    params = arguments if isinstance(arguments, dict) else {}
    raw_path = params.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) else ""
    if not path:
        return error_result("path is required")
    content = params.get("content")
    if not isinstance(content, str):
        return error_result("content is required")
    workspace = require_workspace(context)
    if workspace is None:
        return error_result("write requires an active workspace")
    try:
        resolved = await workspace.resolve(path, access="write")
        async with state.file_lock(f"{workspace.id}\0{resolved.relative_path}"):
            await workspace.fs.write_file(path, content.encode(), atomic=True)
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot write: {error}")
    return text_result(f"Wrote {len(content)} bytes to {path}")


async def edit_tool(
    _call_id: str, arguments: object, context: ToolContext, state: BuiltinToolState
) -> ToolResult:
    """Apply an exact string replacement, matching ``createEditTool().execute``."""
    params = arguments if isinstance(arguments, dict) else {}
    raw_path = params.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) else ""
    if not path:
        return error_result("path is required")
    old = params.get("old_string")
    new = params.get("new_string")
    if not isinstance(old, str):
        return error_result("old_string is required")
    if not isinstance(new, str):
        return error_result("new_string is required")
    if old == new:
        return error_result("old_string and new_string are identical")
    workspace = require_workspace(context)
    if workspace is None:
        return error_result("edit requires an active workspace")

    try:
        resolved = await workspace.resolve(path, access="readwrite")
        async with state.file_lock(f"{workspace.id}\0{resolved.relative_path}"):
            stat = await workspace.fs.stat(path)
            if stat.is_directory:
                return error_result(f"Cannot edit: {path} is a directory")
            content = (await workspace.fs.read_file(path)).decode("utf-8")
            occurrences = content.count(old)
            if occurrences == 0:
                return error_result("old_string not found in file")
            replace_all = params.get("replace_all") is True
            if occurrences > 1 and not replace_all:
                return error_result(
                    f"old_string appears {occurrences} times; provide a more specific string "
                    "or set replace_all=true"
                )
            updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
            await workspace.fs.write_file(
                path, updated.encode(), mode=stat.mode & 0o777, atomic=True
            )
    except FileNotFoundError:
        return error_result(f"Cannot edit: {path} not found")
    except UnicodeDecodeError:
        return error_result(f"Cannot edit: {path} is not UTF-8 text")
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot edit: {error}")
    count = occurrences if params.get("replace_all") is True else 1
    return text_result(f"Replaced {count} occurrence{'s' if count != 1 else ''} in {path}")


def _number_lines(content: str, offset: int | None, limit: int | None) -> str:
    lines = content.split("\n")
    start = offset - 1 if offset is not None else 0
    count = min(limit, MAX_LINES) if limit is not None else MAX_LINES
    end = min(start + count, len(lines))
    numbered = "\n".join(
        f"{line_number:>6}→{line}"
        for line_number, line in enumerate(lines[start:end], start=start + 1)
    )
    if end < len(lines):
        numbered += (
            f"\n\n[Showing lines {start + 1}-{end} of {len(lines)}. "
            f"Use offset={end + 1} to continue.]"
        )
    return numbered


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value)


async def _not_found(path: str, context: ToolContext) -> ToolResult:
    workspace = context.workspace
    assert workspace is not None
    pure = PurePath(path)
    try:
        entries = await workspace.fs.readdir(str(pure.parent))
    except (OSError, ValueError, WorkspaceViolation):
        return error_result(f"Cannot read: {path} not found")
    match = next(
        (
            entry.name
            for entry in entries
            if entry.name != pure.name and PurePath(entry.name).stem == pure.stem
        ),
        None,
    )
    suggestion = str(pure.parent / match) if match is not None else None
    message = f"Cannot read: {path} not found"
    if suggestion is not None:
        message += f". Did you mean {suggestion}?"
    return error_result(message)


def _read_notebook(path: str, data: bytes) -> ToolResult:
    try:
        notebook = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return error_result(f"Cannot read notebook {path}: {error}")
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        return error_result(f"Cannot read notebook {path}: cells is not a list")
    sections: list[str] = []
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            continue
        kind = cell.get("cell_type", "unknown")
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        sections.append(f"## Cell {index} ({kind})\n{text}")
        outputs = cell.get("outputs", [])
        if isinstance(outputs, list):
            rendered = "".join(_notebook_output(item) for item in outputs)
            if rendered:
                sections.append(rendered[:NOTEBOOK_OUTPUT_LIMIT])
    return text_result("\n\n".join(sections))


def _notebook_output(output: object) -> str:
    if not isinstance(output, dict):
        return ""
    for key in ("text", "traceback"):
        value = output.get(key)
        if isinstance(value, list):
            return "".join(str(item) for item in value)
        if isinstance(value, str):
            return value
    data = output.get("data")
    if isinstance(data, dict):
        text = data.get("text/plain")
        if isinstance(text, list):
            return "".join(str(item) for item in text)
        if isinstance(text, str):
            return text
    return ""
