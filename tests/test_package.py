"""The top-level package is the every-run vocabulary. Equality, not a subset."""

import nexora

STABLE_CONTRACT = frozenset(
    {
        "Agent",
        "AgentDefinition",
        "AgentRuntime",
        "ChatModel",
        "Continue",
        "ControlPlane",
        "Controls",
        "Ctx",
        "Deny",
        "ExecutionContext",
        "FinishPolicy",
        "Halt",
        "HostWorkspaceProvider",
        "Ingress",
        "MemorySteps",
        "PendingInput",
        "Permissions",
        "Proceed",
        "ResumeInput",
        "Suspend",
        "ToolCall",
        "Tools",
        "__version__",
        "gate",
        "run",
    }
)


def test_package_exposes_version() -> None:
    """That there is one, not which one — pinning the literal only breaks on the next bump."""
    assert nexora.__version__


def test_the_public_package_is_the_stable_contract() -> None:
    """A name on the top-level package is a breaking export; a missing one is a missing product."""
    assert set(nexora.__all__) == STABLE_CONTRACT
    for name in STABLE_CONTRACT:
        assert getattr(nexora, name) is not None


def test_workspace_internals_are_not_the_public_surface() -> None:
    """HostWorkspaceProvider is the host seam; directory entries and tar backends are not."""
    assert "HostWorkspaceProvider" in nexora.__all__
    assert "WorkspaceDirEntry" not in nexora.__all__
    assert "TarSnapshotBackend" not in nexora.__all__
    assert "ToolContext" not in nexora.__all__
