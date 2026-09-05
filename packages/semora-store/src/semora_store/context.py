"""Trusted, storage-neutral scope for one execution."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

__all__ = ["ExecutionContext", "ScopedStore"]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Carry host-authenticated execution identity without interpreting tenant metadata.

    ``branch_id`` names one branch of a conversation: the durable unit that a first loop, its
    resumes and its recoveries all share, and that a fork starts anew. It is Semora's idempotency
    coordinate. ``conversation_id`` is the conversation the branch belongs to, as Pydantic AI
    uses the word; the ledger is scoped by it. Every other field is opaque to the framework and
    must originate at the host trust boundary, never in model output or tool input.
    """

    branch_id: str
    conversation_id: str | None = None
    namespace: str | None = None
    actor: str | None = None
    subject: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze a copy so caller mutation cannot change authority mid-run."""
        if not self.branch_id:
            raise ValueError("execution context requires a non-empty branch_id")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


class ScopedStore(Protocol):
    """Bind a trusted execution scope without prescribing an adapter's physical layout."""

    def for_execution(self, context: ExecutionContext) -> "ScopedStore":
        """Return an adapter view bound to ``context``: itself, or a narrower view of itself."""
        ...
