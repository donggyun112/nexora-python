import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from nexora.builtins import BuiltinTools, ExecToolOptions
from nexora.sandbox_remote import HTTPResponse, RemoteSandboxClient, RemoteSandboxError
from nexora.workspace import (
    ContinuousWorkspaceProvider,
    MemoryWorkspaceStateStore,
    SandboxCommand,
    SandboxSessionState,
    ToolContext,
    WorkspaceSeed,
    WorkspaceSnapshot,
)


@dataclass(frozen=True)
class Request:
    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None


class FakeTransport:
    def __init__(self, *responses: HTTPResponse) -> None:
        self.responses = list(responses)
        self.requests: list[Request] = []

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> HTTPResponse:
        self.requests.append(Request(method, url, dict(headers), body))
        return self.responses.pop(0)


def response(body: object, status: int = 200) -> HTTPResponse:
    return HTTPResponse(status, {"content-type": "application/json"}, json.dumps(body).encode())


async def test_remote_command_executes_over_the_sandbox_wire() -> None:
    transport = FakeTransport(
        response({"sessionId": "session-1", "root": "/workspace"}),
        response({"exitCode": 0, "signal": None, "stdout": "ok", "stderr": ""}),
    )
    session = await RemoteSandboxClient(
        "https://sandbox.example", token="secret", transport=transport
    ).acquire(run_id="run-1")

    result = await session.run(SandboxCommand(["pwd"], cwd="src"))

    request = transport.requests[1]
    assert request.url == "https://sandbox.example/sessions/session-1/exec"
    assert json.loads(request.body or b"") == {"argv": ["pwd"], "cwd": "src"}
    assert request.headers["authorization"] == "Bearer secret"
    assert result.stdout == "ok"


async def test_builtin_bash_executes_through_the_remote_workspace_wire() -> None:
    transport = FakeTransport(
        response({"sessionId": "session-1", "root": "/workspace"}),
        response({"exitCode": 0, "signal": None, "stdout": "remote", "stderr": ""}),
    )
    session = await RemoteSandboxClient(
        "https://sandbox.example", transport=transport
    ).acquire(run_id="run-1")
    tools = BuiltinTools(
        context=ToolContext(workdir="/workspace", workspace=session),
        exec_options=ExecToolOptions(allow_list=("pwd",)),
    )

    result = await tools.execute("Bash", "call-1", {"argv": ["pwd"]})

    request = transport.requests[1]
    assert request.url.endswith("/sessions/session-1/exec")
    assert json.loads(request.body or b"")["argv"] == ["pwd"]
    assert result == {"type": "text", "text": "remote"}


async def test_remote_files_use_the_workspace_filesystem_wire() -> None:
    transport = FakeTransport(
        response({"sessionId": "session-1", "root": "/workspace"}),
        HTTPResponse(200, {"content-type": "application/octet-stream"}, b"remote"),
        response({"ok": True}),
    )
    session = await RemoteSandboxClient(
        "https://sandbox.example", transport=transport
    ).acquire()

    content = await session.fs.read_file("dir/a file.txt")
    await session.fs.write_file("result.txt", b"written")

    assert content == b"remote"
    assert transport.requests[1].url.endswith("/fs?path=dir%2Fa+file.txt")
    assert transport.requests[2].body == b"written"


async def test_remote_acquire_forwards_manifest_and_seed_files(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "SKILL.md").write_text("instructions")
    transport = FakeTransport(
        response({"sessionId": "session-1", "root": "/workspace"}),
        response({"ok": True}),
    )
    client = RemoteSandboxClient("https://sandbox.example", transport=transport)

    await client.acquire(
        manifest={"mounts": [{"name": "scratch", "target": "/scratch"}]},
        seed_dirs=[WorkspaceSeed(str(seed), ".agents/skills/example")],
    )

    assert json.loads(transport.requests[0].body or b"")["manifest"]["mounts"][0][
        "name"
    ] == "scratch"
    assert transport.requests[1].url.endswith(
        "/fs?path=.agents%2Fskills%2Fexample%2FSKILL.md"
    )
    assert transport.requests[1].body == b"instructions"


async def test_continuous_remote_workspace_reattaches_its_live_session() -> None:
    transport = FakeTransport(
        response({"sessionId": "session-1", "root": "/workspace"}),
        response({"alive": True, "root": "/workspace"}),
    )
    provider = ContinuousWorkspaceProvider(
        RemoteSandboxClient("https://sandbox.example", transport=transport),
        MemoryWorkspaceStateStore(),
        "conversation-1",
    )
    first = await provider.acquire(run_id="turn-1")
    await first.cleanup()

    second = await provider.acquire(run_id="turn-2")

    assert second.id == "session-1"
    assert transport.requests[1].url.endswith("/sessions/session-1/reattach")
    await second.cleanup()


async def test_remote_resume_rehydrates_snapshot_when_session_is_gone(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.tar"
    archive.write_bytes(b"archive bytes")
    transport = FakeTransport(
        response({"alive": False}),
        response({"sessionId": "session-2", "root": "/workspace"}),
        response({"ok": True}),
    )
    client = RemoteSandboxClient("https://sandbox.example", transport=transport)
    state = SandboxSessionState(
        backend="remote",
        ref="session-1",
        snapshot=WorkspaceSnapshot("snapshot-1", "remote-tar", ref=str(archive)),
    )

    restored = await client.resume(state, run_id="turn-2")

    assert restored.id == "session-2"
    assert transport.requests[2].url.endswith("/sessions/session-2/hydrate")
    assert transport.requests[2].body == b"archive bytes"


async def test_remote_error_preserves_server_classification_without_token() -> None:
    transport = FakeTransport(
        response(
            {"code": "session_not_found", "message": "gone", "retryable": False},
            status=404,
        )
    )
    client = RemoteSandboxClient(
        "https://sandbox.example", token="do-not-leak", transport=transport
    )

    with pytest.raises(RemoteSandboxError) as raised:
        await client.acquire()

    assert (raised.value.code, raised.value.status, raised.value.retryable) == (
        "session_not_found",
        404,
        False,
    )
    assert "do-not-leak" not in str(raised.value)
