"""Encode Pydantic AI messages as append-only transcript entries.

Entries chain through `parent_uuid`; `leaf` markers move the active tip without deleting anything,
`tombstone` markers hide one entry. The backing transcript store treats entries as opaque values.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_core import to_jsonable_python
from semora_store import ExecutionContext, Transcript

__all__ = [
    "SCHEMA_VERSION",
    "Branch",
    "TranscriptWriter",
    "active_branch",
    "encode_message",
    "entry_id",
    "marker_entry",
    "message_entry",
    "messages_at",
    "messages_of",
    "stripped",
]

SCHEMA_VERSION = "pai-v2"

_VOLATILE = frozenset(
    {
        "timestamp",
        "run_id",  # Pydantic AI's: a new one per loop, so a resumed round must not hash differently
        "branch_id",  # ours, in entry metadata
        "conversation_id",
        "usage",
        "provider_details",
        "provider_response_id",
        "provider_name",
        "provider_url",
        "finish_reason",
        "model_name",
    }
)
"""Message fields excluded from deterministic entry identifiers."""


def stripped(value: Any) -> Any:
    """Remove fields that vary across equivalent conversation replays."""
    if isinstance(value, dict):
        return {k: stripped(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [stripped(item) for item in value]
    return value


def entry_id(parent_uuid: str | None, body: dict[str, Any]) -> str:
    """Derive an idempotent entry identifier from its parent and stable body fields."""
    canonical = json.dumps(stripped(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{parent_uuid or ''}\x1f{canonical}".encode()).hexdigest()[:32]


def encode_message(message: ModelMessage) -> dict[str, Any]:
    """One message as a JSON-compatible body."""
    body: dict[str, Any] = to_jsonable_python(message)
    return body


def decode_messages(bodies: list[dict[str, Any]]) -> list[ModelMessage]:
    """The other half."""
    return list(ModelMessagesTypeAdapter.validate_python(bodies))


def message_entry(
    message: ModelMessage,
    *,
    conversation_id: str,
    parent_uuid: str | None = None,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Encode one message as a transcript entry."""
    body = encode_message(message)
    entry: dict[str, Any] = {
        "uuid": entry_id(parent_uuid, body),
        "parent_uuid": parent_uuid,
        "conversation_id": conversation_id,
        "type": str(body.get("kind", "")),
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "message": body,
    }
    if metadata:
        entry["metadata"] = metadata
    return entry


_POSITIONAL = frozenset({"leaf"})
"""Marker kinds whose effect depends on when they were made, not only on what they say."""


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


def active_branch(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve the active transcript branch in chronological order.

    The latest ``leaf`` marker selects a prior tip unless newer chained entries exist. Tombstoned
    entries are omitted without changing parent links.
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


def messages_of(entries: list[dict[str, Any]]) -> list[ModelMessage]:
    """Decode model messages from the active transcript branch."""
    return decode_messages(
        [e["message"] for e in active_branch(entries) if isinstance(e.get("message"), dict)]
    )


def messages_at(entries: list[dict[str, Any]], leaf_uuid: str | None) -> list[ModelMessage]:
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
    return decode_messages(branch)


class TranscriptWriter:
    """Append one run's conversation and lifecycle metadata to a transcript."""

    def __init__(
        self,
        store: Transcript,
        *,
        conversation_id: str,
        branch_id: str,
        parent_uuid: str | None = None,
    ) -> None:
        """Initialize a writer for one run and conversation chain."""
        self._store = store
        self._conversation_id = conversation_id
        self._branch_id = branch_id
        self._parent_uuid = parent_uuid

    @property
    def parent_uuid(self) -> str | None:
        """The last written entry identifier, for continuing the chain."""
        return self._parent_uuid

    async def opened(self) -> None:
        """Register the run and its start timestamp."""
        await self._store.record_branch(
            self._branch_id,
            {"conversation_id": self._conversation_id, "started_at": datetime.now(UTC)},
        )

    async def record(self, message: ModelMessage) -> bool:
        """Append one message idempotently and report whether it was new."""
        entry = message_entry(
            message,
            conversation_id=self._conversation_id,
            parent_uuid=self._parent_uuid,
            metadata={"branch_id": self._branch_id},
        )
        appended = await self._store.append(entry)
        # Advanced even when the entry was a duplicate: the chain position is the same either way,
        # and stopping would re-parent the next entry onto an older link and fork the conversation.
        self._parent_uuid = entry["uuid"]
        if not appended:
            # A replayed round re-records messages the conversation already holds, so nothing is
            # appended and `active_branch` would still report the leaf the replay started from.
            await self._append_marker("leaf", leaf_uuid=entry["uuid"])
        return appended

    async def rewind(self, leaf_uuid: str | None) -> None:
        """Move the active branch tip without deleting transcript entries."""
        await self._append_marker("leaf", leaf_uuid=leaf_uuid)
        self._parent_uuid = leaf_uuid

    async def forget(self, deleted_uuid: str) -> None:
        """Hide an entry from the active branch using a tombstone marker."""
        await self._append_marker("tombstone", deleted_uuid=deleted_uuid)

    async def closed(
        self,
        stop_reason: str,
        *,
        tool_calls: int = 0,
        usage: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Record terminal run metadata and per-model usage."""
        await self._store.record_branch(
            self._branch_id,
            {
                "conversation_id": self._conversation_id,
                "stop_reason": stop_reason,
                "tool_calls": tool_calls,
                "interrupted_mid_turn": False,
                "ended_at": datetime.now(UTC),
            },
        )
        for model, counts in (usage or {}).items():
            if counts:
                await self._store.record_model_usage(self._branch_id, model, counts)

    async def _append_marker(self, kind: str, **fields: Any) -> None:
        await self._store.append(marker_entry(kind, self._conversation_id, **fields))


class Branch:
    """Keep one runtime attempt aligned with an append-only conversation branch."""

    def __init__(
        self, writer: TranscriptWriter, messages: list[ModelMessage], uuids: list[str]
    ) -> None:
        """Bind a writer to the branch it continues."""
        self.writer = writer
        self.messages = messages
        self.uuids = uuids

    @classmethod
    async def open(
        cls, store: Transcript, conversation_id: str, branch_id: str, *, context: ExecutionContext
    ) -> "Branch":
        """Restore a run's active branch and continue writing from its tip."""
        store = store.for_execution(context)
        entries = await store.read(conversation_id)
        branch = active_branch(entries)
        message_entries = [e for e in branch if isinstance(e.get("message"), dict)]
        writer = TranscriptWriter(
            store,
            conversation_id=conversation_id,
            branch_id=branch_id,
            parent_uuid=branch[-1]["uuid"] if branch else None,
        )
        await writer.opened()
        return cls(writer, messages_of(entries), [str(e["uuid"]) for e in message_entries])

    async def replace(self, desired: list[ModelMessage]) -> None:
        """Move to the common prefix and append the desired model-visible history."""
        common = 0
        current = [stripped(encode_message(m)) for m in self.messages]
        wanted = [stripped(encode_message(m)) for m in desired]
        while common < min(len(current), len(wanted)) and current[common] == wanted[common]:
            common += 1
        if common < len(current):
            await self.writer.rewind(self.uuids[common - 1] if common else None)
            del self.messages[common:]
            del self.uuids[common:]
        await self.append(desired[common:])

    async def append(self, messages: list[ModelMessage]) -> None:
        """Append exact messages and retain their resulting branch coordinates."""
        for message in messages:
            await self.writer.record(message)
            self.messages.append(message)
            assert self.writer.parent_uuid is not None
            self.uuids.append(self.writer.parent_uuid)
