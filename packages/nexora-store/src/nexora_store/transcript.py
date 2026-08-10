"""Append-only transcript and per-model run-usage store contracts.

Transcript entries are opaque observation records and do not participate in effect execution or
recovery decisions.
"""

import json
from typing import Any, NamedTuple, Protocol, runtime_checkable

__all__ = [
    "MODEL_USAGE_FIELDS",
    "RUN_FIELDS",
    "MemoryTranscript",
    "Transcript",
]

RUN_FIELDS = frozenset(
    {
        "conversation_id",
        "stop_reason",
        "tool_calls",
        "interrupted_mid_turn",
        "started_at",
        "ended_at",
    }
)
"""Fields accepted by run metadata records."""

MODEL_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "cost_usd",
    }
)
"""Fields accepted by per-model usage records."""


class _Row(NamedTuple):
    """Store one transcript entry and its arrival order."""

    seq: int
    entry: dict[str, Any]


@runtime_checkable
class Transcript(Protocol):
    """Persist conversation entries and per-model run cost."""

    async def append(self, entry: dict[str, Any]) -> bool:
        """Append an entry idempotently.

        Args:
            entry: Transcript entry containing ``conversation_id`` and ``uuid``.

        Returns:
            ``True`` when inserted, or ``False`` when the identity already exists.
        """
        ...

    async def read(self, conversation_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return entries in arrival order, optionally limited to the newest tail."""
        ...

    async def record_run(self, run_id: str, fields: dict[str, Any]) -> None:
        """Merge validated fields into a run record.

        Raises:
            ValueError: If ``fields`` contains a name outside ``RUN_FIELDS``.
        """
        ...

    async def record_model_usage(self, run_id: str, model: str, counts: dict[str, Any]) -> None:
        """Merge validated usage fields for one model in a run.

        Raises:
            ValueError: If ``counts`` contains a name outside ``MODEL_USAGE_FIELDS``.
        """
        ...

    async def read_run(self, run_id: str) -> dict[str, Any] | None:
        """The run record, or `None` if this run was never opened."""
        ...

    async def read_model_usage(self, run_id: str) -> dict[str, dict[str, Any]]:
        """This run's token counts keyed by the model that spent them."""
        ...


def check_fields(table: str, fields: dict[str, Any], allowed: frozenset[str]) -> None:
    """Raise ``ValueError`` when a record contains unsupported fields."""
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"{table} has no field {sorted(unknown)}")


class MemoryTranscript:
    """Implement ``Transcript`` with process-local collections."""

    def __init__(self) -> None:
        """Initialize empty transcript, run, and per-model stores."""
        self._rows: dict[str, list[_Row]] = {}
        self._seen: set[tuple[str, str]] = set()
        self._runs: dict[str, dict[str, Any]] = {}
        self._model_usage: dict[tuple[str, str], dict[str, Any]] = {}
        self._sequence = 0

    async def append(self, entry: dict[str, Any]) -> bool:
        """Append an entry unless its conversation already holds that `uuid`."""
        conversation_id = str(entry.get("conversation_id", ""))
        key = (conversation_id, str(entry.get("uuid", "")))
        if key in self._seen:
            return False
        self._seen.add(key)
        self._sequence += 1
        self._rows.setdefault(conversation_id, []).append(_Row(self._sequence, _frozen(entry)))
        return True

    async def read(self, conversation_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return entries in arrival order, optionally limited to the newest tail."""
        rows = self._rows.get(conversation_id, [])
        window = rows if limit is None else rows[len(rows) - limit :] if limit > 0 else []
        return [_frozen(row.entry) for row in window]

    async def record_run(self, run_id: str, fields: dict[str, Any]) -> None:
        """Merge validated fields into a process-local run record."""
        check_fields("run", fields, RUN_FIELDS)
        self._runs.setdefault(run_id, {}).update(fields)

    async def record_model_usage(self, run_id: str, model: str, counts: dict[str, Any]) -> None:
        """Merge validated token counts for one model of one run."""
        check_fields("model usage", counts, MODEL_USAGE_FIELDS)
        self._model_usage.setdefault((run_id, model), {}).update(counts)

    async def read_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a copied run record."""
        row = self._runs.get(run_id)
        return dict(row) if row is not None else None

    async def read_model_usage(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Return this run's token counts per model."""
        return {
            model: dict(counts)
            for (recorded, model), counts in self._model_usage.items()
            if recorded == run_id
        }


def _frozen(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible deep copy of an entry."""
    copied: dict[str, Any] = json.loads(json.dumps(entry))
    return copied
