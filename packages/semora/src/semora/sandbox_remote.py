"""Remote workspace client for Semora's provider-neutral sandbox HTTP protocol."""

import asyncio
import json
import os
import posixpath
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from .workspace import (
    CommandResult,
    PathAccess,
    ResolvedWorkspacePath,
    SandboxCommand,
    SandboxSessionState,
    WorkspaceAccessMode,
    WorkspaceDirEntry,
    WorkspaceFileStat,
    WorkspaceFS,
    WorkspaceSeed,
    WorkspaceSession,
    WorkspaceSnapshot,
    WorkspaceViolation,
)

__all__ = [
    "HTTPResponse",
    "HTTPTransport",
    "RemoteSandboxClient",
    "RemoteSandboxError",
    "UrllibHTTPTransport",
]


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    """Transport-neutral HTTP response."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class HTTPTransport(Protocol):
    """Injectable asynchronous HTTP transport used by the remote client."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> HTTPResponse:
        """Perform one request without interpreting its body."""
        ...


class UrllibHTTPTransport:
    """Dependency-free transport that moves blocking urllib work off the event loop."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> HTTPResponse:
        """Perform one HTTP request and retain error response bodies."""

        def send() -> HTTPResponse:
            request = Request(url, data=body, headers=dict(headers), method=method)
            try:
                with urlopen(request) as response:
                    return HTTPResponse(
                        response.status,
                        {name.lower(): value for name, value in response.headers.items()},
                        response.read(),
                    )
            except HTTPError as error:
                return HTTPResponse(
                    error.code,
                    {name.lower(): value for name, value in error.headers.items()},
                    error.read(),
                )

        return await asyncio.to_thread(send)


class RemoteSandboxError(RuntimeError):
    """Structured error returned by a sandbox server."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int,
        retryable: bool | None,
    ) -> None:
        """Retain stable wire metadata without including credentials."""
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable


class RemoteSandboxClient:
    """Provision, execute, persist, and reconnect remote workspace sessions.

    This matches the TS ``RemoteSandboxClient`` wire. The remote server owns OS and network
    isolation; no command ever falls back to an unjailed local subprocess.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        spool_dir: str | Path | None = None,
        transport: HTTPTransport | None = None,
        end_of_turn: Literal["release", "delete"] = "release",
        root_dir: str | None = None,
        enforced_domains: Sequence[str] = (),
    ) -> None:
        """Configure transport credentials, lifecycle, and declared server egress policy."""
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RemoteSandboxClient endpoint must be an absolute HTTP(S) URL")
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._spool_dir = (
            Path(spool_dir).expanduser()
            if spool_dir is not None
            else Path(tempfile.gettempdir()) / "semora-remote-snapshots"
        )
        self._transport = transport or UrllibHTTPTransport()
        self._end_of_turn = end_of_turn
        self._root_dir = root_dir
        self._enforced_domains = frozenset(enforced_domains)

    async def acquire(
        self,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        seed_dirs: Sequence[WorkspaceSeed] = (),
    ) -> WorkspaceSession:
        """Provision a fresh remote session."""
        del base_workdir
        selected_root = root_dir or self._root_dir
        payload = _without_none(
            {"runId": run_id, "rootDir": selected_root, "manifest": manifest}
        )
        created = await self._json("POST", "/sessions", payload=payload)
        await self._seed_remote(str(created["sessionId"]), seed_dirs)
        return self._session(str(created["sessionId"]), str(created["root"]))

    async def resume(
        self,
        state: SandboxSessionState,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        seed_dirs: Sequence[WorkspaceSeed] = (),
    ) -> WorkspaceSession:
        """Reattach a live session, otherwise recreate and hydrate its archived bytes."""
        del base_workdir
        if state.ref:
            try:
                attached = await self._json(
                    "POST", f"/sessions/{_encode(state.ref)}/reattach", payload={}
                )
                if attached.get("alive") and attached.get("root"):
                    return self._session(state.ref, str(attached["root"]))
            except Exception:
                pass

        selected_root = root_dir or self._root_dir
        created = await self._json(
            "POST",
            "/sessions",
            payload=_without_none(
                {"runId": run_id, "rootDir": selected_root, "manifest": manifest}
            ),
        )
        session_id = str(created["sessionId"])
        snapshot = state.snapshot
        if snapshot is not None and snapshot.ref is not None and selected_root is None:
            archive = Path(snapshot.ref)
            if archive.is_file():
                await self._bytes(
                    "POST",
                    f"/sessions/{_encode(session_id)}/hydrate",
                    body=await asyncio.to_thread(archive.read_bytes),
                    content_type="application/octet-stream",
                )
        await self._seed_remote(session_id, seed_dirs)
        return self._session(session_id, str(created["root"]))

    async def _seed_remote(
        self, session_id: str, seeds: Sequence[WorkspaceSeed]
    ) -> None:
        """Copy local seed files over the remote FS wire, best effort and no symlinks."""
        for seed in seeds:
            source = Path(seed.source).expanduser()
            if not source.is_dir() or source.is_symlink():
                continue
            for item in source.rglob("*"):
                if item.is_symlink() or not item.is_file():
                    continue
                relative = item.relative_to(source).as_posix()
                destination = str(PurePosixPath(seed.destination) / relative)
                try:
                    body = await asyncio.to_thread(item.read_bytes)
                    query = urlencode({"path": destination})
                    await self._bytes(
                        "PUT",
                        f"/sessions/{_encode(session_id)}/fs?{query}",
                        body=body,
                        content_type="application/octet-stream",
                    )
                except Exception:
                    continue

    def _session(self, session_id: str, root: str) -> "RemoteSandboxSession":
        return RemoteSandboxSession(
            session_id,
            root,
            self,
            spool_dir=self._spool_dir,
            end_of_turn=self._end_of_turn,
            enforced_domains=self._enforced_domains,
        )

    async def _json(
        self, method: str, path: str, *, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        raw = await self._request(
            method,
            path,
            body=body,
            content_type="application/json" if body is not None else None,
        )
        if not raw:
            return {}
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise TypeError("sandbox response must be a JSON object")
        return decoded

    async def _json_list(self, method: str, path: str) -> list[dict[str, Any]]:
        raw = await self._request(method, path)
        decoded = json.loads(raw)
        if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
            raise TypeError("sandbox response must be a JSON object list")
        return decoded

    async def _bytes(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        return await self._request(method, path, body=body, content_type=content_type)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        headers: dict[str, str] = {}
        if self._token is not None:
            headers["authorization"] = f"Bearer {self._token}"
        if content_type is not None:
            headers["content-type"] = content_type
        try:
            response = await self._transport.request(
                method,
                f"{self._endpoint}{path}",
                headers=headers,
                body=body,
            )
        except Exception as error:
            raise RemoteSandboxError(
                str(error), code="transport_error", status=0, retryable=True
            ) from error
        if 200 <= response.status < 300:
            return response.body
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        retryable = payload.get("retryable")
        raise RemoteSandboxError(
            str(payload.get("message") or f"HTTP {response.status}"),
            code=str(payload.get("code") or "http_error"),
            status=response.status,
            retryable=retryable if isinstance(retryable, bool) else None,
        )


class RemoteSandboxSession:
    """One isolated remote workspace and its wire-backed filesystem."""

    mode: WorkspaceAccessMode = "workspace-write"
    isolated = True

    def __init__(
        self,
        session_id: str,
        root: str,
        client: RemoteSandboxClient,
        *,
        spool_dir: Path,
        end_of_turn: Literal["release", "delete"],
        enforced_domains: frozenset[str],
    ) -> None:
        """Bind one remote session id to the shared client transport."""
        self.id = session_id
        self.root = PurePosixPath(posixpath.normpath(root))
        if not self.root.is_absolute():
            raise ValueError("remote workspace root must be absolute")
        self._client = client
        self._spool_dir = spool_dir
        self._end_of_turn = end_of_turn
        self._enforced_domains = enforced_domains
        self._cleaned = False
        self.fs: WorkspaceFS = _RemoteWorkspaceFS(self)

    async def resolve(
        self, path: str, *, access: PathAccess = "read"
    ) -> ResolvedWorkspacePath:
        """Validate paths lexically; the remote server validates again on use."""
        del access
        raw = PurePosixPath(path)
        candidate = raw if raw.is_absolute() else self.root / raw
        normalized = PurePosixPath(posixpath.normpath(str(candidate)))
        try:
            relative = normalized.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceViolation(f"path escapes remote workspace root: {path!r}") from error
        return ResolvedWorkspacePath(normalized, self.root, str(relative or "."), True)

    async def run(self, command: SandboxCommand) -> CommandResult:
        """Execute only over the remote wire, never through a local fallback."""
        if not command.argv or any(not isinstance(part, str) or not part for part in command.argv):
            raise ValueError("command argv must contain non-empty strings")
        if command.allowed_domains is not None and not set(command.allowed_domains).issubset(
            self._enforced_domains
        ):
            raise WorkspaceViolation("command requests network domains outside provider policy")
        await self.resolve(command.cwd, access="read")
        payload = _without_none(
            {
                "argv": list(command.argv),
                "cwd": command.cwd,
                "env": dict(command.env) or None,
                "timeoutMs": (
                    int(command.timeout_seconds * 1000)
                    if command.timeout_seconds is not None
                    else None
                ),
            }
        )
        result = await self._client._json(
            "POST", f"/sessions/{_encode(self.id)}/exec", payload=payload
        )
        return CommandResult(
            exit_code=result.get("exitCode"),
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            timed_out=bool(result.get("timedOut")),
            signal=result.get("signal"),
            aborted=bool(result.get("aborted")),
        )

    async def snapshot(self) -> WorkspaceSnapshot:
        """Spool a remote tar archive locally for cold recovery."""
        archive = await self._client._bytes(
            "POST", f"/sessions/{_encode(self.id)}/persist"
        )
        self._spool_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        ref = self._spool_dir / f"{uuid4()}.tar"

        def save() -> None:
            descriptor = os.open(ref, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(archive)
            finally:
                os.close(descriptor)

        await asyncio.to_thread(save)
        return WorkspaceSnapshot(
            self.id,
            "remote-tar",
            ref=str(ref),
            created_at=datetime.now(UTC).isoformat(),
            metadata={"remote_session_id": self.id},
        )

    async def session_state(self) -> SandboxSessionState:
        """Prefer the live remote reference; callers may explicitly request a snapshot."""
        return SandboxSessionState(
            backend="remote",
            ref=self.id,
            created_at=datetime.now(UTC).isoformat(),
        )

    async def cleanup(self) -> None:
        """Release by default, or request eager remote deletion when configured."""
        if self._cleaned:
            return
        self._cleaned = True
        if self._end_of_turn == "release":
            return
        with suppress(Exception):
            await self._client._json("DELETE", f"/sessions/{_encode(self.id)}")


class _RemoteWorkspaceFS:
    """Workspace filesystem routed entirely through the remote server."""

    def __init__(self, session: RemoteSandboxSession) -> None:
        self._session = session

    async def read_file(self, path: str) -> bytes:
        """Read file bytes over the wire."""
        await self._session.resolve(path, access="read")
        return await self._session._client._bytes("GET", self._path("fs", path))

    async def write_file(
        self, path: str, data: bytes, *, mode: int = 0o644, atomic: bool = True
    ) -> None:
        """Write file bytes; the server owns no-follow and atomic policy."""
        del mode, atomic
        await self._session.resolve(path, access="write")
        await self._session._client._bytes(
            "PUT", self._path("fs", path), body=data, content_type="application/octet-stream"
        )

    async def stat(self, path: str) -> WorkspaceFileStat:
        """Read remote file metadata."""
        await self._session.resolve(path, access="read")
        result = await self._session._client._json("GET", self._path("stat", path))
        return WorkspaceFileStat(
            size=int(result["size"]),
            mtime_ms=float(result["mtimeMs"]),
            is_file=bool(result["isFile"]),
            is_directory=bool(result["isDirectory"]),
            mode=int(result["mode"]),
        )

    async def readdir(self, path: str) -> list[WorkspaceDirEntry]:
        """List remote directory children."""
        await self._session.resolve(path, access="list")
        rows = await self._session._client._json_list("GET", self._path("readdir", path))
        return [
            WorkspaceDirEntry(str(row["name"]), bool(row["isDirectory"])) for row in rows
        ]

    async def real_path(self, path: str) -> ResolvedWorkspacePath:
        """Return a lexically resolved remote path; the server validates on use."""
        return await self._session.resolve(path, access="read")

    def _path(self, operation: str, path: str) -> str:
        query = urlencode({"path": path})
        return f"/sessions/{_encode(self._session.id)}/{operation}?{query}"


def _encode(value: str) -> str:
    return quote(value, safe="")


def _without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: item for name, item in value.items() if item is not None}
