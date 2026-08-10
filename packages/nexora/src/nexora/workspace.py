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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from .contracts.types import Tools

WorkspaceAccessMode = Literal["read-only", "workspace-write", "danger-full-access"]
PathAccess = Literal["read", "write", "readwrite", "list"]
CleanupMode = Literal["keep", "delete"]

__all__ = [
    "CommandResult",
    "ContextualTools",
    "ContinuousWorkspaceProvider",
    "HostWorkspaceProvider",
    "MemoryWorkspaceStateStore",
    "ResolvedWorkspacePath",
    "ResumableWorkspaceProvider",
    "SandboxCommand",
    "SandboxSessionState",
    "SnapshotBackend",
    "TarSnapshotBackend",
    "ToolContext",
    "WorkspaceAccessMode",
    "WorkspaceDirEntry",
    "WorkspaceFS",
    "WorkspaceFileStat",
    "WorkspaceProvider",
    "WorkspaceSeed",
    "WorkspaceSession",
    "WorkspaceSnapshot",
    "WorkspaceStateStore",
    "WorkspaceViolation",
]


class WorkspaceViolation(PermissionError):
    """A path, write, or isolation request crossed the workspace policy."""


@dataclass(frozen=True, slots=True)
class ResolvedWorkspacePath:
    """A validated host path and its root-relative spelling."""

    path: PurePath
    root: PurePath
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
    signal: str | None = None
    aborted: bool = False


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

    def to_dict(self) -> dict[str, Any]:
        """Encode a JSON-safe store value."""
        return {
            name: value
            for name, value in {
                "id": self.id,
                "backend": self.backend,
                "ref": self.ref,
                "root": self.root,
                "fingerprint": self.fingerprint,
                "created_at": self.created_at,
                "metadata": dict(self.metadata),
            }.items()
            if value is not None
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkspaceSnapshot":
        """Decode a snapshot persisted as JSON."""
        return cls(
            id=str(value["id"]),
            backend=str(value["backend"]),
            ref=str(value["ref"]) if value.get("ref") is not None else None,
            root=str(value["root"]) if value.get("root") is not None else None,
            fingerprint=(
                str(value["fingerprint"]) if value.get("fingerprint") is not None else None
            ),
            created_at=(
                str(value["created_at"]) if value.get("created_at") is not None else None
            ),
            metadata=(
                dict(value["metadata"])
                if isinstance(value.get("metadata"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceSeed:
    """Host directory copied into a fresh workspace without following symlinks."""

    source: str
    destination: str


@dataclass(frozen=True, slots=True)
class SandboxSessionState:
    """JSON-safe reconnect state with no provider credentials."""

    backend: str
    ref: str | None = None
    snapshot: WorkspaceSnapshot | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Encode reconnect state without provider credentials."""
        return {
            name: value
            for name, value in {
                "backend": self.backend,
                "ref": self.ref,
                "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
                "created_at": self.created_at,
                "metadata": dict(self.metadata),
            }.items()
            if value is not None
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SandboxSessionState":
        """Decode reconnect state persisted as JSON."""
        snapshot = value.get("snapshot")
        return cls(
            backend=str(value["backend"]),
            ref=str(value["ref"]) if value.get("ref") is not None else None,
            snapshot=(
                WorkspaceSnapshot.from_dict(snapshot)
                if isinstance(snapshot, Mapping)
                else None
            ),
            created_at=(
                str(value["created_at"]) if value.get("created_at") is not None else None
            ),
            metadata=(
                dict(value["metadata"])
                if isinstance(value.get("metadata"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceFileStat:
    """Provider-neutral file metadata."""

    size: int
    mtime_ms: float
    is_file: bool
    is_directory: bool
    mode: int


@dataclass(frozen=True, slots=True)
class WorkspaceDirEntry:
    """One immediate directory child."""

    name: str
    is_directory: bool


@runtime_checkable
class WorkspaceFS(Protocol):
    """File operations shared by host and remote workspaces."""

    async def read_file(self, path: str) -> bytes:
        """Read one file without escaping the workspace."""
        ...

    async def write_file(
        self, path: str, data: bytes, *, mode: int = 0o644, atomic: bool = True
    ) -> None:
        """Create or replace one file, creating parents as needed."""
        ...

    async def stat(self, path: str) -> WorkspaceFileStat:
        """Return metadata without following a final-component symlink."""
        ...

    async def readdir(self, path: str) -> list[WorkspaceDirEntry]:
        """Return immediate children."""
        ...

    async def real_path(self, path: str) -> ResolvedWorkspacePath:
        """Return the canonical provider-side path."""
        ...


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

    @property
    def id(self) -> str:
        """Backend session identity."""
        ...

    @property
    def root(self) -> PurePath:
        """Logical workspace root visible to tools."""
        ...

    @property
    def mode(self) -> WorkspaceAccessMode:
        """Filesystem access mode."""
        ...

    @property
    def isolated(self) -> bool:
        """Whether process execution is enforced outside the host boundary."""
        ...

    @property
    def fs(self) -> WorkspaceFS:
        """Provider-specific filesystem operations."""
        ...

    async def resolve(self, path: str, *, access: PathAccess = "read") -> ResolvedWorkspacePath:
        """Resolve a path under this session's policy."""
        ...

    async def run(self, command: SandboxCommand) -> CommandResult:
        """Run a shell-free command according to the session isolation contract."""
        ...

    async def snapshot(self) -> WorkspaceSnapshot:
        """Persist or point at the current workspace state."""
        ...

    async def session_state(self) -> SandboxSessionState:
        """Return reconnect state, preferring a live backend reference when available."""
        ...

    async def cleanup(self) -> None:
        """Release session resources idempotently."""
        ...


@runtime_checkable
class WorkspaceProvider(Protocol):
    """Acquire workspace sessions."""

    async def acquire(
        self,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        seed_dirs: Sequence[WorkspaceSeed] = (),
    ) -> WorkspaceSession:
        """Acquire a new or fixed-root workspace session."""
        ...


@runtime_checkable
class ResumableWorkspaceProvider(WorkspaceProvider, Protocol):
    """Workspace provider that can reconnect or rehydrate prior state."""

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
        """Reconnect live state or restore its snapshot into a fresh session."""
        ...


@runtime_checkable
class WorkspaceStateStore(Protocol):
    """Persist the latest reconnect state for each conversation."""

    async def load(self, conversation_id: str) -> SandboxSessionState | None:
        """Load the latest state, or ``None``."""
        ...

    async def save(self, conversation_id: str, state: SandboxSessionState) -> None:
        """Insert or replace the latest state."""
        ...

    async def delete(self, conversation_id: str) -> None:
        """Forget the conversation's state."""
        ...


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Execution resources injected into a contextual tool collection."""

    workdir: str
    workspace: WorkspaceSession | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextualTools(Tools, Protocol):
    """Tools that can be rebound to one attempt's workspace context."""

    def get_context(self) -> ToolContext:
        """Return the current immutable context."""
        ...

    def with_context(self, context: ToolContext) -> Tools:
        """Return a tool collection bound to ``context``."""
        ...


class MemoryWorkspaceStateStore:
    """Process-local ``WorkspaceStateStore`` for tests and ephemeral runtimes."""

    def __init__(self) -> None:
        """Initialize an empty conversation-state map."""
        self._states: dict[str, SandboxSessionState] = {}

    async def load(self, conversation_id: str) -> SandboxSessionState | None:
        """Load the latest state."""
        state = self._states.get(conversation_id)
        return SandboxSessionState.from_dict(state.to_dict()) if state is not None else None

    async def save(self, conversation_id: str, state: SandboxSessionState) -> None:
        """Replace the latest state."""
        self._states[conversation_id] = SandboxSessionState.from_dict(state.to_dict())

    async def delete(self, conversation_id: str) -> None:
        """Forget the latest state."""
        self._states.pop(conversation_id, None)


class ContinuousWorkspaceProvider:
    """Keep one workspace continuous across serialized turns of a conversation.

    This ports ``ContinuousWorkspaceProvider.acquire`` and its cleanup wrapper: load prior state,
    resume when possible, acquire fresh on failure, then save new reconnect state before cleanup.
    State-store failures are best effort so observability does not become runtime availability.
    """

    def __init__(
        self,
        inner: ResumableWorkspaceProvider,
        store: WorkspaceStateStore,
        conversation_id: str,
        *,
        on_warning: Callable[[str, Exception], None] | None = None,
    ) -> None:
        """Bind a resumable provider and state store to one conversation."""
        self._inner = inner
        self._store = store
        self._conversation_id = conversation_id
        self._on_warning = on_warning

    async def acquire(
        self,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        seed_dirs: Sequence[WorkspaceSeed] = (),
    ) -> WorkspaceSession:
        """Resume the prior session or acquire fresh, then persist state on cleanup."""
        try:
            prior = await self._store.load(self._conversation_id)
        except Exception as error:
            self._warn("workspace-state load failed; acquiring fresh", error)
            prior = None
        if prior is not None:
            try:
                session = await self._inner.resume(
                    prior,
                    run_id=run_id,
                    base_workdir=base_workdir,
                    root_dir=root_dir,
                    manifest=manifest,
                    seed_dirs=seed_dirs,
                )
            except Exception as error:
                self._warn("workspace resume failed; acquiring fresh", error)
                session = await self._inner.acquire(
                    run_id=run_id,
                    base_workdir=base_workdir,
                    root_dir=root_dir,
                    manifest=manifest,
                    seed_dirs=seed_dirs,
                )
        else:
            session = await self._inner.acquire(
                run_id=run_id,
                base_workdir=base_workdir,
                root_dir=root_dir,
                manifest=manifest,
                seed_dirs=seed_dirs,
            )
        return _ContinuousWorkspaceSession(
            session, self._store, self._conversation_id, self._warn
        )

    def _warn(self, message: str, error: Exception) -> None:
        if self._on_warning is not None:
            self._on_warning(message, error)


class _ContinuousWorkspaceSession:
    """Delegate a session while adding state persistence to idempotent cleanup."""

    def __init__(
        self,
        inner: WorkspaceSession,
        store: WorkspaceStateStore,
        conversation_id: str,
        warn: Callable[[str, Exception], None],
    ) -> None:
        self._inner = inner
        self._store = store
        self._conversation_id = conversation_id
        self._warn = warn
        self._cleaned = False
        self.id = inner.id
        self.root = inner.root
        self.mode = inner.mode
        self.isolated = inner.isolated
        self.fs = inner.fs

    async def resolve(
        self, path: str, *, access: PathAccess = "read"
    ) -> ResolvedWorkspacePath:
        """Delegate path resolution."""
        return await self._inner.resolve(path, access=access)

    async def run(self, command: SandboxCommand) -> CommandResult:
        """Delegate process execution."""
        return await self._inner.run(command)

    async def snapshot(self) -> WorkspaceSnapshot:
        """Delegate snapshot creation."""
        return await self._inner.snapshot()

    async def session_state(self) -> SandboxSessionState:
        """Delegate reconnect state creation."""
        return await self._inner.session_state()

    async def cleanup(self) -> None:
        """Persist reconnect state before releasing the underlying session."""
        if self._cleaned:
            return
        self._cleaned = True
        try:
            state = await self._inner.session_state()
            await self._store.save(self._conversation_id, state)
        except Exception as error:
            self._warn("workspace snapshot/persist failed on cleanup", error)
        await self._inner.cleanup()


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
        manifest: Mapping[str, Any] | None = None,
        seed_dirs: Sequence[WorkspaceSeed] = (),
    ) -> "HostWorkspaceSession":
        """Create a host-directory session, respecting externally owned roots."""
        del manifest
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
        await _materialize_seed_dirs(resolved, seed_dirs)
        return HostWorkspaceSession(
            session_id,
            resolved,
            mode=self._mode,
            cleanup="keep" if externally_owned else self._cleanup,
            snapshot_backend=self._snapshot_backend,
        )

    async def resume(
        self,
        state: SandboxSessionState,
        *,
        run_id: str | None = None,
        base_workdir: str | None = None,
        root_dir: str | None = None,
        manifest: Mapping[str, Any] | None = None,
        seed_dirs: Sequence[WorkspaceSeed] = (),
    ) -> "HostWorkspaceSession":
        """Reuse an unchanged live root or restore into a fresh per-run root."""
        snapshot = state.snapshot
        if snapshot is None:
            raise LookupError("host workspace state has no snapshot")
        if snapshot.root is not None:
            live = Path(snapshot.root).expanduser().resolve()
            if live.is_dir() and (
                snapshot.fingerprint is None
                or snapshot.fingerprint == await asyncio.to_thread(_fingerprint, live)
            ):
                return await self.acquire(
                    run_id=run_id or snapshot.id,
                    root_dir=str(live),
                    manifest=manifest,
                    seed_dirs=seed_dirs,
                )
        if (
            self._snapshot_backend is None
            or snapshot.backend != self._snapshot_backend.kind
            or snapshot.ref is None
            or not await self._snapshot_backend.restorable(snapshot.ref)
        ):
            raise LookupError(f"workspace snapshot {snapshot.id!r} is not restorable")
        session = await self.acquire(
            run_id=run_id or snapshot.id,
            base_workdir=base_workdir,
            root_dir=root_dir,
            manifest=manifest,
        )
        if any(session.root.iterdir()):
            await session.cleanup()
            raise FileExistsError("snapshot destination must be empty")
        await self._snapshot_backend.restore(snapshot.ref, session.root)
        await _materialize_seed_dirs(session.root, seed_dirs)
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
        self.fs = _HostWorkspaceFS(self)

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

    async def session_state(self) -> SandboxSessionState:
        """Wrap the latest host snapshot as reconnect state."""
        snapshot = await self.snapshot()
        return SandboxSessionState(
            backend=snapshot.backend,
            snapshot=snapshot,
            created_at=datetime.now(UTC).isoformat(),
        )

    async def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if self._cleanup_mode == "delete":
            await asyncio.to_thread(shutil.rmtree, self.root, True)


class _HostWorkspaceFS:
    """Filesystem operations enforced through a ``HostWorkspaceSession``."""

    def __init__(self, session: HostWorkspaceSession) -> None:
        self._session = session

    async def read_file(self, path: str) -> bytes:
        """Read a regular file without following a final symlink."""
        resolved = await self._session.resolve(path, access="read")

        def read() -> bytes:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(resolved.path, flags)
            try:
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    return stream.read()
            finally:
                os.close(descriptor)

        return await asyncio.to_thread(read)

    async def write_file(
        self, path: str, data: bytes, *, mode: int = 0o644, atomic: bool = True
    ) -> None:
        """Write without following the final path component, atomically by default."""
        resolved = await self._session.resolve(path, access="write")
        destination = Path(resolved.path)

        def write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if atomic:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{destination.name}.", dir=destination.parent
                )
                try:
                    os.fchmod(descriptor, mode)
                    stream = os.fdopen(descriptor, "wb", closefd=True)
                    descriptor = -1
                    with stream:
                        stream.write(data)
                    os.replace(temporary, destination)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                    Path(temporary).unlink(missing_ok=True)
                return
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_TRUNC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(destination, flags, mode)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(data)
            finally:
                os.close(descriptor)

        await asyncio.to_thread(write)

    async def stat(self, path: str) -> WorkspaceFileStat:
        """Return ``lstat`` metadata for a confined path."""
        resolved = await self._session.resolve(path, access="read")
        result = await asyncio.to_thread(os.lstat, resolved.path)
        return WorkspaceFileStat(
            size=result.st_size,
            mtime_ms=result.st_mtime * 1000,
            is_file=os.path.isfile(resolved.path) and not os.path.islink(resolved.path),
            is_directory=os.path.isdir(resolved.path) and not os.path.islink(resolved.path),
            mode=result.st_mode & 0o7777,
        )

    async def readdir(self, path: str) -> list[WorkspaceDirEntry]:
        """List immediate children of a confined directory."""
        resolved = await self._session.resolve(path, access="list")

        def list_entries() -> list[WorkspaceDirEntry]:
            with os.scandir(resolved.path) as entries:
                return sorted(
                    (
                        WorkspaceDirEntry(entry.name, entry.is_dir(follow_symlinks=False))
                        for entry in entries
                    ),
                    key=lambda entry: entry.name,
                )

        return await asyncio.to_thread(list_entries)

    async def real_path(self, path: str) -> ResolvedWorkspacePath:
        """Resolve an existing path and reject missing entries."""
        resolved = await self._session.resolve(path, access="read")
        if not await asyncio.to_thread(os.path.exists, resolved.path):
            raise FileNotFoundError(path)
        return resolved


def _safe_segment(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    return safe.strip(".-") or "workspace"


async def _materialize_seed_dirs(root: Path, seeds: Sequence[WorkspaceSeed]) -> None:
    """Copy declared seed files best-effort, skipping all symbolic links."""

    def materialize() -> None:
        for seed in seeds:
            try:
                source = Path(seed.source).expanduser()
                destination = (root / seed.destination).resolve(strict=False)
                destination.relative_to(root)
                if not source.is_dir() or source.is_symlink():
                    continue
                for item in source.rglob("*"):
                    if item.is_symlink() or not item.is_file():
                        continue
                    relative = item.relative_to(source)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item, target, follow_symlinks=False)
            except (OSError, ValueError):
                continue

    await asyncio.to_thread(materialize)


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
