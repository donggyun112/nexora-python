"""Sandboxed command execution built-in."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping

from ..workspace import SandboxCommand, ToolContext, WorkspaceViolation
from ._types import (
    BuiltinToolState,
    ExecToolOptions,
    ToolResult,
    error_result,
    require_workspace,
    text_result,
    tool_environment,
)

MAX_OUTPUT_BYTES = 256 * 1024
MAX_PERSIST_BYTES = 16 * 1024 * 1024
MAX_TIMEOUT_MS = 600_000

SHELL_INTERPRETERS = frozenset(
    {
        "bash",
        "sh",
        "zsh",
        "dash",
        "ksh",
        "tcsh",
        "csh",
        "fish",
        "ash",
        "busybox",
        "python",
        "ruby",
        "perl",
        "node",
        "nodejs",
        "deno",
        "bun",
        "php",
        "lua",
        "awk",
        "gawk",
        "mawk",
        "nawk",
        "sed",
        "find",
        "xargs",
        "tar",
        "cpio",
        "zip",
        "unzip",
        "git",
        "hg",
        "svn",
        "wget",
        "curl",
        "scp",
        "rsync",
        "ssh",
        "sshpass",
        "telnet",
        "nc",
        "ncat",
        "socat",
        "env",
        "sudo",
        "doas",
        "su",
        "docker",
        "podman",
        "kubectl",
        "nsenter",
        "chroot",
        "unshare",
        "setsid",
    }
)


async def exec_tool(
    call_id: str,
    arguments: object,
    context: ToolContext,
    state: BuiltinToolState,
    options: ExecToolOptions,
) -> ToolResult:
    """Execute through ``WorkspaceSession.run``, porting ``createExecTool().execute``."""
    del state
    params = arguments if isinstance(arguments, dict) else {}
    if params.get("run_in_background") is True:
        return error_result(
            "run_in_background needs a background-task registry (not supported by this tool bundle)"
        )
    workspace = require_workspace(context)
    if workspace is None:
        return error_result("Bash requires an active workspace")

    resolved = _resolve_command(params, options)
    if isinstance(resolved, str):
        return error_result(resolved)
    argv, label = resolved

    raw_cwd = params.get("cwd")
    cwd = raw_cwd.strip() if isinstance(raw_cwd, str) and raw_cwd.strip() else "."
    try:
        resolved_cwd = await workspace.resolve(cwd, access="read")
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f'cwd "{cwd}" is not inside the workspace: {error}')

    timeout_ms = _timeout_ms(params.get("timeoutMs"), options.default_timeout_ms)
    try:
        result = await workspace.run(
            SandboxCommand(
                argv=argv,
                cwd=str(resolved_cwd.path),
                env=tool_environment(options.env_allow_list),
                inherit_env=False,
                timeout_seconds=timeout_ms / 1000,
                require_isolation=options.require_isolation,
                allowed_domains=options.allowed_domains,
            )
        )
    except (OSError, ValueError, WorkspaceViolation) as error:
        return error_result(f"sandboxed exec failed: {error}")

    combined = result.stdout + (f"\n[stderr]\n{result.stderr}" if result.stderr else "")
    encoded = combined.encode()
    persisted_path: str | None = None
    if len(encoded) > MAX_OUTPUT_BYTES:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", call_id)[:64] or "out"
        persisted_path = f".exec-output-{safe_id}.log"
        try:
            await workspace.fs.write_file(
                persisted_path, encoded[:MAX_PERSIST_BYTES], atomic=True
            )
        except (OSError, ValueError, WorkspaceViolation):
            persisted_path = None
    text = _utf8_prefix(encoded, MAX_OUTPUT_BYTES) or "(no output)"
    if persisted_path is not None:
        text += (
            f"\n\n[full output ({min(len(encoded), MAX_PERSIST_BYTES)} bytes) written to "
            f"{persisted_path} — read this file for the complete output]"
        )
    elif len(encoded) > MAX_OUTPUT_BYTES:
        text += f"\n\n[Output truncated at {MAX_OUTPUT_BYTES} bytes]"

    if result.timed_out:
        status = "timeout"
        text += f"\n\n[Killed: timeout after {timeout_ms}ms]"
    elif result.aborted:
        status = "aborted"
        text += "\n\n[Aborted by caller]"
    elif result.signal:
        status = f"signal {result.signal}"
        text += f"\n\n[Killed by signal: {result.signal}]"
    elif result.exit_code == 0:
        status = "ok"
    else:
        status = f"exit {result.exit_code if result.exit_code is not None else 'unknown'}"
    return text_result(text if status == "ok" else f"[{status}] {label}\n{text}")


def _resolve_command(
    params: Mapping[str, object], options: ExecToolOptions
) -> tuple[list[str], str] | str:
    argv_value = params.get("argv")
    allow = frozenset(options.allow_list)
    allow_all = "*" in allow
    if isinstance(argv_value, list) and argv_value:
        if not all(isinstance(item, str) for item in argv_value):
            return "argv must be an array of strings"
        argv = [str(item) for item in argv_value]
        error = _validate_program(argv[0])
        if error is not None:
            return error
        if not allow:
            return (
                "Bash is unconfigured: ExecToolOptions requires a non-empty allow_list. "
                "Pass explicit command names to enable them."
            )
        if not allow_all and argv[0] not in allow:
            return f'Executable "{argv[0]}" not in allow_list. Allowed: {", ".join(sorted(allow))}'
        if not options.allow_shell and _is_interpreter(argv[0]):
            return (
                f'Executable "{argv[0]}" is a shell/interpreter/exec-surface and is blocked. '
                "Set allow_shell=True only when an OS sandbox is the real boundary."
            )
        return argv, " ".join(argv)

    command = params.get("command")
    if isinstance(command, str) and command.strip():
        if not options.allow_shell:
            return 'Shell-string mode is disabled. Use { argv: ["program", "arg1", ...] } instead.'
        if not allow:
            return (
                "Bash is unconfigured: ExecToolOptions requires a non-empty allow_list. "
                "Pass explicit command names to enable them."
            )
        if not allow_all:
            programs = _shell_programs(command)
            if programs is None:
                return (
                    "Shell command could not be verified against the allow_list. "
                    "Use argv form or a simpler command."
                )
            offending = sorted(set(programs) - allow)
            if offending:
                return (
                    f"Command(s) not in allow_list: {', '.join(offending)}. "
                    f"Allowed: {', '.join(sorted(allow))}"
                )
        return ["bash", "-lc", command], command
    return "Either argv (preferred) or command must be provided"


def _validate_program(program: str) -> str | None:
    if not program:
        return "program is empty"
    if "/" in program or "\\" in program:
        return f'program "{program}" must be a bare command name (no path separators)'
    if ".." in program:
        return f'program "{program}" must not contain ".."'
    if program.startswith("-"):
        return f'program "{program}" must not start with "-"'
    return None


def _is_interpreter(program: str) -> bool:
    canonical = re.sub(
        r"^(python|ruby|perl|node|lua|php)(?:-|)(?:\d+(?:\.\d+)?)$", r"\1", program
    )
    if canonical == "nodejs":
        canonical = "node"
    return canonical in SHELL_INTERPRETERS


def _shell_programs(command: str) -> list[str] | None:
    """Conservatively parse simple shell pipelines for per-command allow-list checks."""
    if any(marker in command for marker in ("$", "`", "\n", "\r", "<", ">", "(", ")")):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    programs: list[str] = []
    expecting = True
    for token in tokens:
        if token in {"|", "||", "&&", ";", "&"}:
            if expecting:
                return None
            expecting = True
            continue
        if expecting and "=" in token and not token.startswith("="):
            continue
        if expecting:
            if _validate_program(token) is not None:
                return None
            programs.append(token)
            expecting = False
    return None if expecting or not programs else programs


def _timeout_ms(value: object, default: int) -> int:
    selected = (
        int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else default
    )
    return min(max(selected, 1_000), MAX_TIMEOUT_MS)


def _utf8_prefix(value: bytes, limit: int) -> str:
    return value[:limit].decode("utf-8", errors="ignore")


def exec_is_read_only(arguments: object) -> bool:
    """Fail-closed concurrency classification from TS ``classifyReadOnly``."""
    if not isinstance(arguments, dict) or arguments.get("run_in_background") is True:
        return False
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return False
    return argv[0] in {
        "cat",
        "head",
        "tail",
        "wc",
        "pwd",
        "ls",
        "rg",
        "grep",
        "find",
        "stat",
        "file",
        "diff",
    }
