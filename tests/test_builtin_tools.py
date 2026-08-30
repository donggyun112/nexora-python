from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from semora import AgentRuntime, HostWorkspaceProvider
from semora.builtins import (
    BuiltinTools,
    ExecToolOptions,
    WebFetchResponse,
    WebFetchToolOptions,
    builtin_tools,
)
from semora.builtins._types import BuiltinToolState
from semora.workspace import ToolContext

from .test_loop import a_call, says, scripted


class Fetches:
    def __init__(self, body: bytes, *, content_type: str = "text/plain; charset=utf-8") -> None:
        self.body = body
        self.content_type = content_type
        self.urls: list[str] = []

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        del headers, timeout_seconds
        self.urls.append(url)
        return WebFetchResponse(
            status=200,
            reason="OK",
            url=url,
            headers={"Content-Type": self.content_type},
            body=self.body[:max_bytes],
        )


async def in_workspace(
    root: Path,
    *,
    exec_options: ExecToolOptions | None = None,
    engine: str | None = None,
) -> BuiltinTools:
    """Bind the built-ins to a workspace, optionally pinning which search engine they resolve to.

    `engine` seeds the detection cache instead of probing, because whichever engine the machine
    has is the only one its tests would ever reach otherwise.
    """
    session = await HostWorkspaceProvider(root=root).acquire(run_id="builtins")
    state = BuiltinToolState()
    if engine is not None:
        state.search_engines[session.id] = engine
    return BuiltinTools(
        context=ToolContext(workdir=str(root), workspace=session),
        exec_options=exec_options,
        _state=state,
    )


def test_bundle_is_the_ts_core_set_without_web_search() -> None:
    names = {definition["name"] for definition in BuiltinTools().list()}

    assert names == {"read", "write", "edit", "grep", "glob", "Bash", "web_fetch"}


async def test_runtime_injects_its_workspace_into_the_builtin_bundle(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    runtime = AgentRuntime(workspace_provider=HostWorkspaceProvider(root=root))

    await runtime.run(
        "builtin-write",
        scripted(
            says(
                "",
                a_call(
                    "call-1",
                    "write",
                    {"path": "created.txt", "content": "from the agent"},
                ),
            ),
            says("done"),
        ),
        BuiltinTools(),
        "create the file",
    )

    assert (root / "created.txt").read_text() == "from the agent"


async def test_file_tools_fail_closed_without_a_workspace() -> None:
    result = await BuiltinTools().execute("read", "call-1", {"path": "README.md"})

    assert result["type"] == "error"
    assert "AgentRuntime(workspace_provider=...)" in result["message"]


async def test_builtin_factory_accepts_an_explicit_workspace_context(tmp_path: Path) -> None:
    """The public factory must preserve a caller-managed execution context."""
    (tmp_path / "notes.txt").write_text("factory context")
    session = await HostWorkspaceProvider(root=tmp_path).acquire(run_id="factory-context")
    context = ToolContext(workdir=str(tmp_path), workspace=session)

    result = await builtin_tools(context=context).execute(
        "read", "call-1", {"path": "notes.txt"}
    )

    assert result["text"] == "     1→factory context"


async def test_read_numbers_and_pages_text_like_ts_number_lines(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma")
    tools = await in_workspace(tmp_path)

    result = await tools.execute("read", "call-1", {"path": "notes.txt", "offset": 2, "limit": 1})

    assert result["text"] == (
        "     2→beta\n\n[Showing lines 2-2 of 3. Use offset=3 to continue.]"
    )


async def test_read_deduplicates_an_unchanged_window(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha")
    tools = await in_workspace(tmp_path)
    await tools.execute("read", "call-1", {"path": "notes.txt"})

    repeated = await tools.execute("read", "call-2", {"path": "notes.txt"})

    assert repeated["text"] == (
        "<file unchanged since you last read it — its content is already in context: notes.txt>"
    )


async def test_read_lists_directories(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    (tmp_path / "file.txt").write_text("x")
    tools = await in_workspace(tmp_path)

    result = await tools.execute("read", "call-1", {"path": "."})

    assert result["text"] == "Directory: .\n\nfile.txt\nfolder/"


async def test_write_creates_parent_directories(tmp_path: Path) -> None:
    tools = await in_workspace(tmp_path)

    await tools.execute("write", "call-1", {"path": "nested/file.txt", "content": "hello"})

    assert (tmp_path / "nested" / "file.txt").read_text() == "hello"


async def test_edit_refuses_an_ambiguous_replacement(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("same same")
    tools = await in_workspace(tmp_path)

    result = await tools.execute(
        "edit",
        "call-1",
        {"path": "file.txt", "old_string": "same", "new_string": "new"},
    )

    assert result == {
        "type": "error",
        "message": (
            "old_string appears 2 times; provide a more specific string or set replace_all=true"
        ),
    }


async def test_edit_replace_all_preserves_the_other_content(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("same middle same")
    tools = await in_workspace(tmp_path)

    await tools.execute(
        "edit",
        "call-1",
        {
            "path": "file.txt",
            "old_string": "same",
            "new_string": "new",
            "replace_all": True,
        },
    )

    assert (tmp_path / "file.txt").read_text() == "new middle new"


async def test_file_tools_cannot_escape_the_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    tools = await in_workspace(root)

    result = await tools.execute("write", "call-1", {"path": "../outside", "content": "no"})

    assert result["type"] == "error"
    assert not (tmp_path / "outside").exists()


@pytest.mark.skipif(shutil.which("rg") is None, reason="glob deliberately requires ripgrep")
async def test_glob_returns_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("one")
    (tmp_path / "src" / "two.txt").write_text("two")
    tools = await in_workspace(tmp_path)

    result = await tools.execute("glob", "call-1", {"pattern": "**/*.py"})

    assert result["text"] == "Found 1 file\nsrc/one.py"


async def test_grep_returns_matching_content(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("alpha\nneedle\n")
    (tmp_path / "two.txt").write_text("other\n")
    tools = await in_workspace(tmp_path)

    result = await tools.execute("grep", "call-1", {"pattern": "needle"})

    assert "one.txt:2:needle" in str(result["text"])
    assert "two.txt" not in str(result["text"])


async def test_grep_falls_back_to_system_grep_and_says_what_it_lost(tmp_path: Path) -> None:
    """The fallback is a supported path that no run reaches, because detection prefers ripgrep.

    `_detect_engine` picks whichever engine the machine has, so a developer or runner with ripgrep
    installed never executes `_grep_args` or the degradation notice — and one without it never
    executes the ripgrep half. Pinning the engine is what makes both reachable in the same run.
    """
    (tmp_path / "one.txt").write_text("alpha\nneedle\n")
    (tmp_path / "two.txt").write_text("other\n")
    tools = await in_workspace(tmp_path, engine="grep")

    result = await tools.execute("grep", "call-1", {"pattern": "needle", "type": "py"})

    # Found through POSIX grep arguments, which are built separately from ripgrep's.
    assert "one.txt:2:needle" in str(result["text"])
    assert "two.txt" not in str(result["text"])
    # And the answer says which requested behaviour grep could not honour, rather than pretending.
    assert "[grep fallback: ripgrep not found — type, .gitignore not respected unavailable]" in str(
        result["text"]
    )


async def test_bash_is_disabled_until_an_allow_list_is_supplied(tmp_path: Path) -> None:
    tools = await in_workspace(tmp_path)

    result = await tools.execute("Bash", "call-1", {"argv": ["pwd"]})

    assert result["type"] == "error"
    assert "unconfigured" in str(result["message"])


async def test_bash_runs_allowlisted_argv_in_the_workspace(tmp_path: Path) -> None:
    tools = await in_workspace(
        tmp_path,
        exec_options=ExecToolOptions(
            allow_list=("pwd",), require_isolation=False, allowed_domains=None
        ),
    )

    result = await tools.execute("Bash", "call-1", {"argv": ["pwd"]})

    assert result == {"type": "text", "text": f"{tmp_path}\n"}


async def test_bash_rejects_executable_paths_even_when_wildcarded(tmp_path: Path) -> None:
    tools = await in_workspace(
        tmp_path,
        exec_options=ExecToolOptions(
            allow_list=("*",), require_isolation=False, allowed_domains=None
        ),
    )

    result = await tools.execute("Bash", "call-1", {"argv": ["/bin/pwd"]})

    assert result["type"] == "error"
    assert "bare command name" in str(result["message"])


async def test_web_fetch_upgrades_http_and_cleans_html() -> None:
    transport = Fetches(
        b"<style>no</style><h1>Hello &amp; welcome</h1><p>Body</p>",
        content_type="text/html",
    )
    tools = BuiltinTools(web_fetch_options=WebFetchToolOptions(transport=transport))

    result = await tools.execute("web_fetch", "call-1", {"url": "http://example.test/page"})

    assert transport.urls == ["https://example.test/page"]
    assert result["text"] == (
        "URL: https://example.test/page\nContent-Type: text/html\n\nHello & welcome\n\nBody"
    )


async def test_web_fetch_caches_by_url_and_prompt() -> None:
    transport = Fetches(b"body")
    tools = BuiltinTools(web_fetch_options=WebFetchToolOptions(transport=transport))
    request: dict[str, Any] = {"url": "https://example.test", "prompt": "extract"}
    await tools.execute("web_fetch", "call-1", request)

    await tools.execute("web_fetch", "call-2", request)

    assert len(transport.urls) == 1


async def test_web_fetch_rejects_non_http_protocols_without_fetching() -> None:
    transport = Fetches(b"secret")
    tools = BuiltinTools(web_fetch_options=WebFetchToolOptions(transport=transport))

    result = await tools.execute("web_fetch", "call-1", {"url": "file:///etc/passwd"})

    assert result == {"type": "error", "message": "Unsupported URL scheme: file:"}
    assert transport.urls == []
