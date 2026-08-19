from pathlib import Path

import pytest
from nexora import HostWorkspaceProvider
from nexora.workspace import (
    ContinuousWorkspaceProvider,
    MemoryWorkspaceStateStore,
    SandboxCommand,
    SandboxSessionState,
    TarSnapshotBackend,
    WorkspaceSeed,
    WorkspaceViolation,
)


async def test_host_workspace_rejects_paths_outside_its_root(tmp_path: Path) -> None:
    session = await HostWorkspaceProvider(root=tmp_path / "workspace").acquire(run_id="run-1")

    with pytest.raises(WorkspaceViolation):
        await session.resolve("../secret", access="read")


async def test_read_only_workspace_rejects_write_resolution(tmp_path: Path) -> None:
    session = await HostWorkspaceProvider(
        root=tmp_path / "workspace", mode="read-only"
    ).acquire(run_id="run-1")

    with pytest.raises(WorkspaceViolation, match="read-only"):
        await session.resolve("result.txt", access="write")


async def test_host_workspace_fails_closed_for_untrusted_execution(tmp_path: Path) -> None:
    session = await HostWorkspaceProvider(root=tmp_path / "workspace").acquire(run_id="run-1")

    with pytest.raises(WorkspaceViolation, match="not an OS sandbox"):
        await session.run(SandboxCommand(["python", "-c", "print('unsafe')"]))


async def test_trusted_host_command_uses_workspace_cwd(tmp_path: Path) -> None:
    session = await HostWorkspaceProvider(root=tmp_path / "workspace").acquire(run_id="run-1")

    result = await session.run(
        SandboxCommand(
            ["python", "-c", "import pathlib; print(pathlib.Path.cwd().name)"],
            require_isolation=False,
            allowed_domains=None,
        )
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "workspace"


async def test_tar_snapshot_restores_into_a_fresh_workspace(tmp_path: Path) -> None:
    backend = TarSnapshotBackend(tmp_path / "snapshots")
    provider = HostWorkspaceProvider(
        base_dir=tmp_path / "runs",
        per_run=True,
        cleanup="delete",
        snapshot_backend=backend,
    )
    original = await provider.acquire(run_id="original")
    (original.root / "state.txt").write_text("durable")
    snapshot = await original.snapshot()
    await original.cleanup()

    restored = await provider.resume(
        SandboxSessionState(backend=snapshot.backend, snapshot=snapshot),
        run_id="restored",
    )

    assert (restored.root / "state.txt").read_text() == "durable"
    await restored.cleanup()


async def test_host_workspace_never_claims_to_enforce_egress_policy(tmp_path: Path) -> None:
    session = await HostWorkspaceProvider(root=tmp_path / "workspace").acquire(run_id="run-1")

    with pytest.raises(WorkspaceViolation, match="egress"):
        await session.run(
            SandboxCommand(["python", "-V"], require_isolation=False, allowed_domains=[])
        )


async def test_workspace_filesystem_seam_reads_what_it_writes(tmp_path: Path) -> None:
    session = await HostWorkspaceProvider(root=tmp_path / "workspace").acquire(run_id="run-1")

    await session.fs.write_file("nested/state.txt", b"durable")

    assert await session.fs.read_file("nested/state.txt") == b"durable"


async def test_continuous_provider_restores_the_previous_turn(tmp_path: Path) -> None:
    states = MemoryWorkspaceStateStore()
    provider = ContinuousWorkspaceProvider(
        HostWorkspaceProvider(
            base_dir=tmp_path / "runs",
            per_run=True,
            snapshot_backend=TarSnapshotBackend(tmp_path / "snapshots"),
        ),
        states,
        "conversation-1",
    )
    first = await provider.acquire(run_id="turn-1")
    await first.fs.write_file("state.txt", b"from turn one")
    await first.cleanup()

    second = await provider.acquire(run_id="turn-2")

    assert await second.fs.read_file("state.txt") == b"from turn one"
    await second.cleanup()


async def test_host_workspace_materializes_declared_seed_directories(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "instructions.md").write_text("use this")
    provider = HostWorkspaceProvider(root=tmp_path / "workspace")

    session = await provider.acquire(
        seed_dirs=[WorkspaceSeed(str(seed), ".agents/skills/example")]
    )

    assert (
        session.root / ".agents" / "skills" / "example" / "instructions.md"
    ).read_text() == "use this"
