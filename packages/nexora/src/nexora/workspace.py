"""Workspace lifecycle, path boundary, command execution, and durable snapshots.

``HostWorkspaceProvider`` is a concrete host-directory implementation, not an OS sandbox. Its
``isolated`` flag is deliberately false so untrusted callers can fail closed and substitute a
container, mount-namespace, or remote implementation of the same protocols.
"""

import asyncio
import hashlib
import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

WorkspaceAccessMode = Literal["read-only", "workspace-write", "danger-full-access"]
PathAccess = Literal["read", "write", "readwrite", "list"]
CleanupMode = Literal["keep", "delete"]

__all__ = [
    "CommandResult",
    "HostWorkspaceProvider",
    "ResolvedWorkspacePath",
    "SandboxCommand",
    "SnapshotBackend",
    "TarSnapshotBackend",
    "WorkspaceAccessMode",
    "WorkspaceProvider",
    "WorkspaceSession",
    "WorkspaceSnapshot",
    "WorkspaceViolation",
]


class WorkspaceViolation(PermissionError):
    """A path, write, or isolation request crossed the workspace policy."""


@dataclass(frozen=True, slots=True)
class ResolvedWorkspacePath:
    """A validated host path and its root-relative spelling."""

    path: Path
    root: Path
    relative_path: str
    writable: bool


@dataclass(frozen=True, slots=True)
class SandboxCommand:
    """Shell-free process request.

    ``require_isolation`` defaults true. Host sessions therefore reject it unless a caller marks
    the command trusted or supplies an isolated WorkspaceSession implementation.
    """

    argv: Sequence[str]
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    require_isolation: bool = True
    allowed_domains: Sequence[str] | None = field(default_factory=tuple)
    """Allowed egress domains; empty denies all, while ``None`` explicitly inherits host access."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured process termination state."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """Portable or live-root workspace restore coordinate."""

    id: str
    backend: str
    ref: str | None = None
    root: str | None = None
    fingerprint: str | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SnapshotBackend(Protocol):
    """Persist and restore complete workspace bytes independently of process isolation."""

    kind: str

    async def persist(self, snapshot_id: str, root: Path) -> str:
        """Archive ``root`` and return an opaque restore reference."""
        ...

    async def restore(self, ref: str, destination: Path) -> None:
        """Restore ``ref`` into an empty destination."""
        ...

    async def restorable(self, ref: str) -> bool:
        """Return whether the reference is currently available."""
        ...


@runtime_checkable
class WorkspaceSession(Protocol):
    """Directory-level boundary used by file and process tools."""

    id: str
    root: Path
    mode: WorkspaceAccessMode
    isolated: bool

    async def resolve(self, path: str, *, access: PathAccess = "read") -> ResolvedWorkspacePath:
        """Resolve a path under this session's policy."""
        ...

    async def run(self, command: SandboxCommand) -> CommandResult:
        """Run a shell-free command according to the session isolation contract."""
        ...

    async def snapshot(self) -> WorkspaceSnapshot:
        """Persist or point at the current workspace state."""
        ...

    async def cleanup(self) -> None:
        """Release session resources idempotently."""
        ...


@runtime_checkable
class WorkspaceProvider(Protocol):
    """Acquire and resume workspace sessions."""

    async def acquire(
        self,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
    ) -> WorkspaceSession:
        """Acquire a new or fixed-root workspace session."""
        ...

    async def resume(
        self, snapshot: WorkspaceSnapshot, *, run_id: str | None = None
    ) -> WorkspaceSession:
        """Reconnect a live root or restore a durable snapshot."""
        ...


class TarSnapshotBackend:
    """Store portable gzip tar snapshots in a caller-selected durable directory."""

    kind = "tar"

    def __init__(self, directory: str | Path) -> None:
        """Store archives below ``directory``."""
        self._directory = Path(directory).expanduser().resolve()

    async def persist(self, snapshot_id: str, root: Path) -> str:
        """Atomically archive the complete root under a safe snapshot name."""
        safe_id = _safe_segment(snapshot_id)
        try:
            self._directory.relative_to(root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("snapshot directory cannot be inside the workspace root")
        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = self._directory / f"{safe_id}.tar.gz"
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

        def archive() -> None:
            try:
                with tarfile.open(temporary, "w:gz", dereference=False) as bundle:
                    for child in sorted(root.iterdir(), key=lambda item: item.name):
                        bundle.add(child, arcname=child.name, recursive=True)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

        await asyncio.to_thread(archive)
        return str(target)

    async def restore(self, ref: str, destination: Path) -> None:
        """Safely extract an owned archive into an empty directory."""
        archive = Path(ref).resolve()
        if not await self.restorable(str(archive)):
            raise LookupError(f"workspace snapshot {ref!r} is not restorable")

        def extract() -> None:
            destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            if any(destination.iterdir()):
                raise FileExistsError("snapshot destination must be empty")
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(destination, filter="data")

        await asyncio.to_thread(extract)

    async def restorable(self, ref: str) -> bool:
        """Accept only existing archive references owned by this backend."""
        candidate = Path(ref).resolve()
        try:
            candidate.relative_to(self._directory)
        except ValueError:
            return False
        return candidate.is_file()


class HostWorkspaceProvider:
    """Create best-effort host-directory sessions matching ``WorkspaceProvider``.

    This ports the TS ``HostWorkspaceProvider.acquire`` lifecycle and
    ``HostWorkspaceSession.resolve`` boundary. It intentionally does not claim network, syscall,
    or host-filesystem isolation.
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        base_dir: str | Path | None = None,
        per_run: bool = False,
        mode: WorkspaceAccessMode = "workspace-write",
        cleanup: CleanupMode | None = None,
        snapshot_backend: SnapshotBackend | None = None,
    ) -> None:
        """Configure fixed or per-run roots and optional portable snapshots."""
        self._root = Path(root).expanduser() if root is not None else None
        self._base_dir = (
            Path(base_dir).expanduser()
            if base_dir is not None
            else Path(tempfile.gettempdir()) / "nexora-workspaces"
        )
        self._per_run = per_run
        self._mode = mode
        self._cleanup = cleanup or ("delete" if per_run else "keep")
        self._snapshot_backend = snapshot_backend

    async def acquire(
        self,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
    ) -> "HostWorkspaceSession":
        """Create a host-directory session, respecting externally owned roots."""
        session_id = run_id or str(uuid4())
        externally_owned = root_dir is not None
        if root_dir is not None:
            root = Path(root_dir).expanduser()
        elif self._per_run:
            self._base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            root = Path(
                tempfile.mkdtemp(
                    prefix=f"{_safe_segment(session_id)}-", dir=self._base_dir
                )
            )
        else:
            selected = self._root or (Path(base_workdir).expanduser() if base_workdir else None)
            if selected is None:
                raise ValueError(
                    "HostWorkspaceProvider requires root, base_workdir, root_dir, or per_run=True"
                )
            root = selected
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
        return HostWorkspaceSession(
            session_id,
            resolved,
            mode=self._mode,
            cleanup="keep" if externally_owned else self._cleanup,
            snapshot_backend=self._snapshot_backend,
        )

    async def resume(
        self, snapshot: WorkspaceSnapshot, *, run_id: str | None = None
    ) -> "HostWorkspaceSession":
        """Reuse an unchanged live root or restore into a fresh per-run root."""
        if snapshot.root is not None:
            live = Path(snapshot.root).expanduser().resolve()
            if live.is_dir() and (
                snapshot.fingerprint is None
                or snapshot.fingerprint == await asyncio.to_thread(_fingerprint, live)
            ):
                return await self.acquire(run_id=run_id or snapshot.id, root_dir=str(live))
        if (
            self._snapshot_backend is None
            or snapshot.backend != self._snapshot_backend.kind
            or snapshot.ref is None
            or not await self._snapshot_backend.restorable(snapshot.ref)
        ):
            raise LookupError(f"workspace snapshot {snapshot.id!r} is not restorable")
        session = await self.acquire(run_id=run_id or snapshot.id)
        if any(session.root.iterdir()):
            await session.cleanup()
            raise FileExistsError("snapshot destination must be empty")
        await self._snapshot_backend.restore(snapshot.ref, session.root)
        return session


class HostWorkspaceSession:
    """Concrete host workspace. File paths are confined; process isolation is not provided."""

    isolated = False

    def __init__(
        self,
        session_id: str,
        root: Path,
        *,
        mode: WorkspaceAccessMode,
        cleanup: CleanupMode,
        snapshot_backend: SnapshotBackend | None,
    ) -> None:
        self.id = session_id
        self.root = root
        self.mode = mode
        self._cleanup_mode = cleanup
        self._snapshot_backend = snapshot_backend
        self._cleaned = False

    async def resolve(
        self, path: str, *, access: PathAccess = "read"
    ) -> ResolvedWorkspacePath:
        writing = access in {"write", "readwrite"}
        if writing and self.mode == "read-only":
            raise WorkspaceViolation("workspace is read-only")
        raw = Path(path).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        if self.mode != "danger-full-access":
            try:
                relative = resolved.relative_to(self.root)
            except ValueError as error:
                raise WorkspaceViolation(f"path escapes workspace root: {path!r}") from error
        else:
            relative = Path(os.path.relpath(resolved, self.root))
        return ResolvedWorkspacePath(resolved, self.root, str(relative), self.mode != "read-only")

    async def run(self, command: SandboxCommand) -> CommandResult:
        if command.require_isolation:
            raise WorkspaceViolation(
                "host workspace is not an OS sandbox; use an isolated WorkspaceProvider "
                "or set require_isolation=False for trusted commands"
            )
        if command.allowed_domains is not None:
            raise WorkspaceViolation(
                "host workspace cannot enforce network egress policy; use an isolated "
                "WorkspaceProvider or set allowed_domains=None for trusted commands"
            )
        if not command.argv or any(not isinstance(part, str) or not part for part in command.argv):
            raise ValueError("command argv must contain non-empty strings")
        cwd = await self.resolve(command.cwd, access="read")
        environment = {**os.environ, **dict(command.env)}
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=cwd.path,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=command.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return CommandResult(
                process.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
                timed_out=True,
            )
        return CommandResult(
            process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def snapshot(self) -> WorkspaceSnapshot:
        fingerprint = await asyncio.to_thread(_fingerprint, self.root)
        if self._snapshot_backend is None:
            return WorkspaceSnapshot(
                self.id,
                "inline-root",
                root=str(self.root),
                fingerprint=fingerprint,
                created_at=datetime.now(UTC).isoformat(),
                metadata={"mode": self.mode},
            )
        ref = await self._snapshot_backend.persist(self.id, self.root)
        return WorkspaceSnapshot(
            self.id,
            self._snapshot_backend.kind,
            ref=ref,
            root=str(self.root),
            fingerprint=fingerprint,
            created_at=datetime.now(UTC).isoformat(),
            metadata={"mode": self.mode},
        )

    async def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if self._cleanup_mode == "delete":
            await asyncio.to_thread(shutil.rmtree, self.root, True)


def _safe_segment(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    return safe.strip(".-") or "workspace"


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        if path.is_symlink():
            digest.update(b"L")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"D")
    return digest.hexdigest()
