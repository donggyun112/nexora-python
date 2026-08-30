"""Encode LangChain messages as append-only transcript entries.

Entries use the TypeScript transcript envelope and a Python-specific LangChain message body.
The backing transcript store treats the resulting mappings as opaque values.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import messages_from_dict, messages_to_dict
from semora_store import Transcript

from .contracts.types import BaseMessage

__all__ = [
    "SCHEMA_VERSION",
    "TranscriptWriter",
    "active_branch",
    "entry_id",
    "marker_entry",
    "message_entry",
    "messages_at",
    "messages_of",
]

SCHEMA_VERSION = "py-v1"

_VOLATILE = frozenset({"id", "usage_metadata", "response_metadata"})
"""Message fields excluded from deterministic entry identifiers."""


def entry_id(parent_uuid: str | None, body: dict[str, Any]) -> str:
    """Derive an idempotent entry identifier from its parent and stable body fields."""
    canonical = json.dumps(_stripped(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{parent_uuid or ''}\x1f{canonical}".encode()).hexdigest()[:32]


def _stripped(value: Any) -> Any:
    """Remove fields that vary across equivalent conversation replays."""
    if isinstance(value, dict):
        return {k: _stripped(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stripped(item) for item in value]
    return value


def message_entry(
    message: BaseMessage,
    *,
    conversation_id: str,
    parent_uuid: str | None = None,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Encode one message as a transcript entry."""
    body = messages_to_dict([message])[0]
    entry: dict[str, Any] = {
        "uuid": entry_id(parent_uuid, body),
        "parent_uuid": parent_uuid,
        "conversation_id": conversation_id,
        "type": str(body.get("type", "")),
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "message": body,
    }
    if metadata:
        entry["metadata"] = metadata
    return entry


def marker_entry(kind: str, conversation_id: str, **fields: Any) -> dict[str, Any]:
    """Create an unchained transcript marker.

    ``leaf`` markers receive positional identities so repeated rewinds remain observable. Other
    marker kinds are content-addressed and therefore idempotent.
    """
    body = {"type": kind, **fields}
    timestamp = datetime.now(UTC).isoformat()
    return {
        "uuid": entry_id(timestamp if kind in _POSITIONAL else None, body),
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "schema_version": SCHEMA_VERSION,
        **body,
    }


_POSITIONAL = frozenset({"leaf"})
"""Marker kinds whose effect depends on when they were made, not only on what they say."""


def active_branch(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve the active transcript branch in chronological order.

    The latest ``leaf`` marker selects a prior tip unless newer chained entries exist. Tombstoned
    entries are omitted without changing parent links.

    Args:
        entries: Transcript entries in append order.

    Returns:
        Entries reachable from the active tip, oldest first.
    """
    chained = {entry["uuid"]: entry for entry in entries if "parent_uuid" in entry}
    buried = {entry["deleted_uuid"] for entry in entries if entry.get("type") == "tombstone"}
    marks = [position for position, entry in enumerate(entries) if entry.get("type") == "leaf"]
    grown = [e["uuid"] for e in entries[marks[-1] + 1 if marks else 0 :] if "parent_uuid" in e]
    tip = grown[-1] if grown else (entries[marks[-1]].get("leaf_uuid") if marks else None)
    branch: list[dict[str, Any]] = []
    seen: set[str] = set()
    while tip is not None and tip in chained and tip not in seen:
        seen.add(tip)
        entry = chained[tip]
        if tip not in buried:
            branch.append(entry)
        tip = entry.get("parent_uuid")
    branch.reverse()
    return branch


def messages_of(entries: list[dict[str, Any]]) -> list[BaseMessage]:
    """Decode model messages from the active transcript branch."""
    bodies = [
        entry["message"]
        for entry in active_branch(entries)
        if isinstance(entry.get("message"), dict)
    ]
    return list(messages_from_dict(bodies))


def messages_at(entries: list[dict[str, Any]], leaf_uuid: str | None) -> list[BaseMessage]:
    """Decode the branch ending at an explicit cursor, independent of the active leaf."""
    chained = {entry["uuid"]: entry for entry in entries if "parent_uuid" in entry}
    buried = {entry["deleted_uuid"] for entry in entries if entry.get("type") == "tombstone"}
    branch: list[dict[str, Any]] = []
    seen: set[str] = set()
    tip = leaf_uuid
    while tip is not None and tip in chained and tip not in seen:
        seen.add(tip)
        entry = chained[tip]
        if tip not in buried and isinstance(entry.get("message"), dict):
            branch.append(entry["message"])
        tip = entry.get("parent_uuid")
    if leaf_uuid is not None and leaf_uuid not in chained:
        raise LookupError(f"transcript cursor {leaf_uuid!r} does not exist")
    branch.reverse()
    return list(messages_from_dict(branch))


class TranscriptWriter:
    """Append one run's conversation and cost metadata to a transcript."""

    def __init__(
        self,
        store: Transcript,
        *,
        conversation_id: str,
        run_id: str,
        parent_uuid: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a writer for one run and conversation chain."""
        self._store = store
        self._conversation_id = conversation_id
        self._run_id = run_id
        self._parent_uuid = parent_uuid
        self._context = dict(context or {})

    @property
    def parent_uuid(self) -> str | None:
        """Return the last written entry identifier for continuing the chain."""
        return self._parent_uuid

    async def opened(self) -> None:
        """Register the run and its start timestamp."""
        await self._store.record_run(
            self._run_id,
            {"conversation_id": self._conversation_id, "started_at": datetime.now(UTC)},
        )

    async def record(self, message: BaseMessage, **metadata: Any) -> bool:
        """Append one message idempotently and report whether it was new."""
        entry = message_entry(
            message,
            conversation_id=self._conversation_id,
            parent_uuid=self._parent_uuid,
            metadata={"run_id": self._run_id, **self._context, **metadata},
        )
        appended = await self._store.append(entry)
        # Advanced even when the entry was a duplicate: the chain position is the same either way,
        # and stopping would re-parent the next entry onto an older link and fork the conversation.
        self._parent_uuid = entry["uuid"]
        return appended

    async def rewind(self, leaf_uuid: str | None) -> None:
        """Move the active branch tip without deleting transcript entries."""
        await self._append_marker("leaf", leaf_uuid=leaf_uuid)
        self._parent_uuid = leaf_uuid

    async def forget(self, deleted_uuid: str) -> None:
        """Hide an entry from the active branch using a tombstone marker."""
        await self._append_marker("tombstone", deleted_uuid=deleted_uuid)

    async def _append_marker(self, kind: str, **fields: Any) -> None:
        entry = marker_entry(kind, self._conversation_id, **fields)
        await self._store.append(entry)

    async def closed(
        self, done: dict[str, Any], *, cost_usd: dict[str, float] | None = None
    ) -> None:
        """Record terminal run metadata and per-model usage.

        Args:
            done: Terminal runtime event containing usage and stop metadata.
            cost_usd: Optional precomputed cost keyed by model name.
        """
        await self._store.record_run(
            self._run_id,
            {
                "conversation_id": self._conversation_id,
                "stop_reason": done.get("stop_reason"),
                "tool_calls": len(done.get("tool_calls") or []),
                "interrupted_mid_turn": bool(done.get("interrupted_mid_turn")),
                "ended_at": datetime.now(UTC),
            },
        )
        for model, counts in _per_model(done).items():
            usage = {name: count for name, count in counts.items() if name in _MODEL_TOKEN_FIELDS}
            if cost_usd and model in cost_usd:
                usage["cost_usd"] = cost_usd[model]
            if usage:
                await self._store.record_model_usage(self._run_id, model, usage)


def _per_model(done: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize terminal usage into a mapping keyed by model name."""
    breakdown = done.get("usage_by_model")
    if isinstance(breakdown, dict) and breakdown:
        return breakdown
    usage = done.get("usage") or {}
    return {str(done.get("model") or ""): usage} if usage else {}


_MODEL_TOKEN_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_write_tokens",
    }
)
"""Usage keys `semora_run_model` has columns for."""
