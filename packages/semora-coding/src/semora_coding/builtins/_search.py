"""Workspace-backed glob and grep built-ins."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence

from semora.workspace import CommandResult, SandboxCommand, ToolContext, WorkspaceViolation

from ._types import (
    BuiltinToolState,
    ToolResult,
    error_result,
    require_workspace,
    text_result,
    tool_environment,
)

SEARCH_TIMEOUT_SECONDS = 120.0
DEFAULT_HEAD_LIMIT = 250
MAX_COLUMNS = 500


async def glob_tool(
    _call_id: str, arguments: object, context: ToolContext, state: BuiltinToolState
) -> ToolResult:
    """Find paths using ripgrep, porting ``createGlobTool().execute``."""
    params = arguments if isinstance(arguments, dict) else {}
    raw_pattern = params.get("pattern")
    pattern = raw_pattern.strip() if isinstance(raw_pattern, str) else ""
    if not pattern:
        return error_result("pattern is required")
    if ".." in pattern:
        return error_result('pattern must not contain ".."')
    workspace = require_workspace(context)
    if workspace is None:
        return error_result("glob requires an active workspace")
    offset = _clamp_int(params.get("offset"), 0, 0)
    head_limit = _clamp_int(params.get("head_limit"), DEFAULT_HEAD_LIMIT, 0)
    raw_path = params.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) and raw_path.strip() else "."
    try:
        resolved = await workspace.fs.real_path(path)
    except FileNotFoundError:
        return text_result("No files found.")
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot glob: {error}")
    if await _detect_engine(context, state) != "rg":
        return error_result("glob requires ripgrep (rg), which was not found on PATH")
    try:
        process = await _run_engine(
            context,
            "rg",
            ["--files", "--glob", pattern, str(resolved.path)],
            str(resolved.root),
        )
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot glob: {error}")
    classified = _classify(process, "glob")
    if isinstance(classified, str):
        return error_result(classified)
    stdout, warning = classified
    lines = [line for line in stdout.splitlines() if line]
    if not lines:
        return text_result("No files found." + warning)
    return await _format_files(
        lines, str(resolved.root), offset, head_limit, warning, context
    )


async def grep_tool(
    _call_id: str, arguments: object, context: ToolContext, state: BuiltinToolState
) -> ToolResult:
    """Search contents, porting ``createGrepTool().execute`` and argument builders."""
    params = arguments if isinstance(arguments, dict) else {}
    raw_pattern = params.get("pattern")
    pattern = raw_pattern.strip() if isinstance(raw_pattern, str) else ""
    if not pattern:
        return error_result("pattern is required")
    output_mode = params.get("output_mode", "content")
    if output_mode not in {"content", "files_with_matches", "count"}:
        return error_result("output_mode must be content, files_with_matches, or count")
    raw_glob = params.get("glob")
    glob = raw_glob.strip() if isinstance(raw_glob, str) and raw_glob.strip() else None
    if glob is not None and ".." in glob:
        return error_result('glob must not contain ".."')
    workspace = require_workspace(context)
    if workspace is None:
        return error_result("grep requires an active workspace")
    raw_path = params.get("path")
    path = raw_path.strip() if isinstance(raw_path, str) and raw_path.strip() else "."
    try:
        resolved = await workspace.fs.real_path(path)
    except FileNotFoundError:
        return text_result("No matches found.")
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot grep: {error}")

    offset = _clamp_int(params.get("offset"), 0, 0)
    head_limit = _clamp_int(params.get("head_limit"), DEFAULT_HEAD_LIMIT, 0)
    engine = await _detect_engine(context, state)
    target = str(resolved.path)
    args = (
        _rg_args(params, pattern, str(output_mode), glob, target)
        if engine == "rg"
        else _grep_args(params, pattern, str(output_mode), glob, target)
    )
    try:
        process = await _run_engine(context, engine, args, str(resolved.root))
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"Cannot grep: {error}")
    classified = _classify(process, "grep")
    if isinstance(classified, str):
        return error_result(classified)
    stdout, warning = classified
    degraded = _grep_degradation(params) if engine == "grep" else ""
    tail = warning + (f"\n\n{degraded}" if degraded else "")
    lines = [line for line in stdout.splitlines() if line]
    if not lines:
        return text_result("No matches found." + tail)
    root = str(resolved.root)
    if output_mode == "files_with_matches":
        return await _format_files(lines, root, offset, head_limit, tail, context)
    if output_mode == "count":
        return _format_count(lines, root, offset, head_limit, tail)
    return _format_content(lines, root, offset, head_limit, tail)


async def _detect_engine(context: ToolContext, state: BuiltinToolState) -> str:
    workspace = context.workspace
    assert workspace is not None
    cached = state.search_engines.get(workspace.id)
    if cached is not None:
        return cached
    try:
        result = await _run_engine(context, "rg", ["--version"], str(workspace.root), timeout=5)
        engine = "rg" if result.exit_code == 0 else "grep"
    except (OSError, ValueError, WorkspaceViolation):
        engine = "grep"
    state.search_engines[workspace.id] = engine
    return engine


async def _run_engine(
    context: ToolContext,
    engine: str,
    args: Sequence[str],
    cwd: str,
    *,
    timeout: float = SEARCH_TIMEOUT_SECONDS,
) -> CommandResult:
    workspace = context.workspace
    assert workspace is not None
    return await workspace.run(
        SandboxCommand(
            argv=[engine, *args],
            cwd=cwd,
            env=tool_environment(()),
            inherit_env=False,
            timeout_seconds=timeout,
            require_isolation=workspace.isolated,
            allowed_domains=() if workspace.isolated else None,
        )
    )


def _classify(result: CommandResult, noun: str) -> tuple[str, str] | str:
    has_stdout = bool(result.stdout.strip())
    if result.aborted:
        return (
            (result.stdout, f"\n\n[{noun} aborted: partial results returned]")
            if has_stdout
            else f"{noun} aborted"
        )
    if result.timed_out:
        return (
            (result.stdout, f"\n\n[{noun} timed out: partial results returned]")
            if has_stdout
            else f"{noun} timed out"
        )
    if result.signal:
        partial = f"\n\n[{noun} killed by signal {result.signal}: partial results returned]"
        return (
            (result.stdout, partial)
            if has_stdout
            else f"{noun} killed by signal {result.signal}"
        )
    if result.exit_code in {0, 1}:
        return result.stdout, ""
    detail = result.stderr.strip() or f"{noun} failed with exit {result.exit_code}"
    if has_stdout:
        warning = (
            f"\n\n[{noun} exited {result.exit_code}: {detail}; partial results returned]"
        )
        return result.stdout, warning
    return detail


def _rg_args(
    params: Mapping[str, object],
    pattern: str,
    output_mode: str,
    glob: str | None,
    target: str,
) -> list[str]:
    args = ["--max-columns", str(MAX_COLUMNS)]
    if params.get("multiline") is True:
        args.extend(["-U", "--multiline-dotall"])
    if params.get("-i") is True:
        args.append("-i")
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    else:
        if params.get("-n", True) is not False:
            args.append("-n")
        _push_context(args, params)
    _push_pattern(args, pattern)
    file_type = params.get("type")
    if isinstance(file_type, str) and file_type:
        args.extend(["--type", file_type])
    for item in _split_globs(glob):
        args.extend(["--glob", item])
    args.append(target)
    return args


def _grep_args(
    params: Mapping[str, object],
    pattern: str,
    output_mode: str,
    glob: str | None,
    target: str,
) -> list[str]:
    args = ["-rE", "--color=never"]
    if params.get("-i") is True:
        args.append("-i")
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    else:
        if params.get("-n", True) is not False:
            args.append("-n")
        _push_context(args, params)
    for item in _split_globs(glob):
        if "/" not in item:
            args.append(f"--include={item}")
    args.append("--")
    _push_pattern(args, pattern)
    args.append(target)
    return args


def _push_context(args: list[str], params: Mapping[str, object]) -> None:
    for key in ("context", "-C"):
        if key in params:
            args.extend(["-C", str(_clamp_int(params[key], 0, 0))])
            return
    if "-B" in params:
        args.extend(["-B", str(_clamp_int(params["-B"], 0, 0))])
    if "-A" in params:
        args.extend(["-A", str(_clamp_int(params["-A"], 0, 0))])


def _push_pattern(args: list[str], pattern: str) -> None:
    if pattern.startswith("-"):
        args.extend(["-e", pattern])
    else:
        args.append(pattern)


def _split_globs(glob: str | None) -> list[str]:
    if glob is None:
        return []
    output: list[str] = []
    for raw in glob.split():
        output.extend([raw] if "{" in raw and "}" in raw else filter(None, raw.split(",")))
    return output


def _grep_degradation(params: Mapping[str, object]) -> str:
    lost: list[str] = []
    if params.get("type"):
        lost.append("type")
    if params.get("multiline"):
        lost.append("multiline")
    lost.append(".gitignore not respected")
    return f"[grep fallback: ripgrep not found — {', '.join(lost)} unavailable]"


async def _format_files(
    lines: list[str],
    root: str,
    offset: int,
    head_limit: int,
    tail: str,
    context: ToolContext,
) -> ToolResult:
    workspace = context.workspace
    assert workspace is not None

    async def mtime(path: str) -> float:
        try:
            return (await workspace.fs.stat(path)).mtime_ms
        except (OSError, ValueError, WorkspaceViolation):
            return 0

    mtimes = await asyncio.gather(*(mtime(path) for path in lines))
    sorted_lines = [
        path
        for path, _ in sorted(
            zip(lines, mtimes, strict=True), key=lambda item: (-item[1], item[0])
        )
    ]
    items, limited = _page(sorted_lines, head_limit, offset)
    relative = [_strip_root(path, root) for path in items]
    info = _limit_info(head_limit if limited else None, offset)
    noun = "file" if len(relative) == 1 else "files"
    header = f"Found {len(relative)} {noun}" + (f" ({info})" if info else "")
    return text_result(f"{header}\n" + "\n".join(relative) + tail)


def _format_count(
    lines: list[str], root: str, offset: int, head_limit: int, tail: str
) -> ToolResult:
    items, limited = _page(lines, head_limit, offset)
    total = 0
    output: list[str] = []
    for line in items:
        path, separator, count = line.rpartition(":")
        if separator and count.isdigit():
            total += int(count)
            output.append(f"{_strip_root(path, root)}:{count}")
        else:
            output.append(_strip_root(line, root))
    info = _limit_info(head_limit if limited else None, offset)
    noun = "occurrence" if total == 1 else "occurrences"
    files = "file" if len(output) == 1 else "files"
    summary = f"Found {total} total {noun} across {len(output)} {files}."
    if info:
        summary += f" ({info})"
    return text_result("\n".join(output) + f"\n\n{summary}" + tail)


def _format_content(
    lines: list[str], root: str, offset: int, head_limit: int, tail: str
) -> ToolResult:
    items, limited = _page(lines, head_limit, offset)
    output = [_strip_content_root(line, root) for line in items]
    info = _limit_info(head_limit if limited else None, offset)
    footer = f"\n\n[pagination: {info}]" if info else ""
    return text_result("\n".join(output) + footer + tail)


def _strip_root(path: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def _strip_content_root(line: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    return line[len(prefix) :] if line.startswith(prefix) else line


def _page(items: list[str], limit: int, offset: int) -> tuple[list[str], bool]:
    if limit == 0:
        return items[offset:], False
    return items[offset : offset + limit], len(items) - offset > limit


def _limit_info(limit: int | None, offset: int) -> str:
    parts = ([f"limit {limit}"] if limit is not None else []) + (
        [f"offset {offset}"] if offset else []
    )
    return ", ".join(parts)


def _clamp_int(value: object, fallback: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return fallback
    return max(minimum, int(value))
