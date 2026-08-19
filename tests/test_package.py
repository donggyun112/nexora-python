import nexora


def test_package_exposes_version() -> None:
    """That there is one, not which one — pinning the literal only breaks on the next bump."""
    assert nexora.__version__


def test_package_exports_the_control_plane() -> None:
    """The decision types are the product. A missing name here is a missing public API."""
    names = (
        "ChatModel",
        "Controls",
        "ControlPlane",
        "FinishPolicy",
        "Suspend",
        "MemorySteps",
        "gate",
    )
    assert set(names) <= set(nexora.__all__)
    for name in names:
        assert getattr(nexora, name) is not None


def test_workspace_internals_are_not_the_public_surface() -> None:
    """HostWorkspaceProvider is public; directory entries and tar backends are not."""
    assert "HostWorkspaceProvider" in nexora.__all__
    assert "WorkspaceDirEntry" not in nexora.__all__
    assert "TarSnapshotBackend" not in nexora.__all__
    assert "ToolContext" not in nexora.__all__
