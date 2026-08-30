"""Identifiers generated at Nexora host boundaries."""

from langchain_core.utils.uuid import uuid7


def new_run_id(prefix: str = "run") -> str:
    """Return a full, time-ordered UUIDv7 run identifier."""
    return f"{prefix}-{uuid7().hex}"


__all__ = ["new_run_id"]
