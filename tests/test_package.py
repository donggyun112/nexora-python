import nexora


def test_package_exposes_version() -> None:
    """That there is one, not which one — pinning the literal only breaks on the next bump."""
    assert nexora.__version__
