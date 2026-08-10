"""Encode LangChain messages as append-only transcript entries.

Entries use the TypeScript transcript envelope and a Python-specific LangChain message body.
The backing transcript store treats the resulting mappings as opaque values.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import messages_from_dict, messages_to_dict
from nexora_store import Transcript

from .contracts.types import BaseMessage

__all__ = [
    "SCHEMA_VERSION",
    "TranscriptWriter",
    "active_branch",
    "entry_id",
    "marker_entry",
    "message_entry",
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
    """A non-message entry: a `leaf` pointer, a `tombstone`, a `summary`.

    Derived from its contents like any other entry, so appending the same marker twice is absorbed
    by the store rather than doubling — **except a `leaf`, whose identity carries its timestamp.**
    A tombstone states set membership and says the same thing however often it is made; a leaf
    states where the branch ends *now*, and `active_branch` reads the last one appended. Content-
    addressing it meant a rewind to a tip that had held the branch before was absorbed as a
    duplicate: the branch stayed wherever the previous leaf pointed while the writer believed it
    had moved, and the next entry chained onto a tip no reader would walk to.

    Markers are deliberately *not* chained — `parent_uuid` is absent — because they are statements
    about the chain, not links in it, and a chain that ran through them would move when a leaf
    pointer was rewritten.
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
    """The entries on the conversation's live branch, oldest first.

    Three rules, in the order they apply:

    * the branch ends at the **last chained entry appended after the last `leaf`**, and at that
      leaf's own target when nothing followed it. A leaf discards what came before — those entries
      stay on disk as a sibling branch, so the rewind keeps its history and a later leaf can return
      to them — and says nothing about what comes after, which is the conversation continuing from
      where it now ends. A leaf naming nothing empties the branch and the next entry starts it over.
      With no leaf at all this is the last chained entry, the linear behaviour every conversation
      had before leaves existed.

      A leaf that stayed the tip was the bug: `TranscriptWriter.record` appends no leaf of its own,
      so every message recorded after the first rewind reached the store and never appeared in the
      conversation again.
    * a **`tombstone` removes its target** from the branch without rewriting anything, which is the
      only deletion an append-only log can offer.
    * the branch itself is walked **backwards along `parent_uuid`** from that tip. Arrival order
      cannot do this job: a rewound conversation has later entries that are not on the branch, and
      `seq` order includes them.

    A tombstoned entry in the middle of the chain is skipped but not spliced — its children keep
    pointing at it, so the walk continues through it. Removing a message must not silently re-parent
    the conversation around the hole.
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
    """The conversation those entries recorded, ready to hand back to a model.

    Reads the active branch, so a rewound or partly deleted conversation decodes to what it is now
    rather than to everything that was ever written. Entries without a `message` are skipped rather
    than raising — a store that accepts unknown kinds (it does, on purpose) will hand one back.
    """
    bodies = [
        entry["message"]
        for entry in active_branch(entries)
        if isinstance(entry.get("message"), dict)
    ]
    return list(messages_from_dict(bodies))


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
        """Initialize a writer for one run and conversation chain.

        `context` is stamped onto every entry's metadata: where the run happened and on what build.
        `cwd`, `git_branch` and `version` are the three the Claude Code transcript carries, and the
        reason is reproducibility — a conversation you cannot locate or match to a build is a
        conversation you cannot re-run or debug. Passed in rather than discovered here, because
        reading a git branch is an effect and this is a codec.
        """
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
        """Register the run before it performs work.

        `started_at` is written here rather than left to a column default, because a default exists
        on one implementation and cannot on the other: the same run then reads back with a field the
        memory store never had. The vantage point that knows the run began is this one.
        """
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
        """Point the active branch at `leaf_uuid`, or empty it with `None`.

        Nothing is deleted. Entries after the new tip stay on disk as a sibling branch, and
        appending a later leaf that names one of them returns to it — the property a rewrite-based
        rewind cannot offer.
        """
        await self._append_marker("leaf", leaf_uuid=leaf_uuid)
        self._parent_uuid = leaf_uuid

    async def forget(self, deleted_uuid: str) -> None:
        """Remove one entry from the active branch without rewriting the log.

        The only deletion an append-only log can offer, and the one a "delete that" request needs.
        The entry's bytes remain; a reader skips it. **That is not enough for a secret** — for
        content that must actually stop existing, the row has to be redacted in the store.
        """
        await self._append_marker("tombstone", deleted_uuid=deleted_uuid)

    async def _append_marker(self, kind: str, **fields: Any) -> None:
        entry = marker_entry(kind, self._conversation_id, **fields)
        await self._store.append(entry)

    async def closed(
        self, done: dict[str, Any], *, cost_usd: dict[str, float] | None = None
    ) -> None:
        """Record how the run ended, and what each model that answered it cost.

        Per model rather than per run: a provider-side fallback answers on a model nobody asked
        for, and two models at one blended rate is a number that prices neither. The loop already
        reports the breakdown when there was one, so this reads `usage_by_model` and otherwise
        attributes the flat total to the single model named — never re-accumulating, which would be
        a second answer to a question the loop already answered.

        `cost_usd` is supplied by the caller because rates are not this layer's knowledge. It is
        stored rather than derived on read for the same reason a rate table would have to exist to
        derive it: without one, yesterday's cost is unrecoverable once prices move.
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
    """The run's token counts keyed by the model that spent them.

    `usage_by_model` is present only when more than one model answered — the loop omits it otherwise
    rather than repeating what `usage` and `model` already say. So the single-model case is rebuilt
    here from those two, and a run whose provider named no model keys on the empty string: that the
    tokens are unattributed is itself a fact, and picking a model for them would be a guess.
    """
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
"""Usage keys `nexora_run_model` has columns for."""
