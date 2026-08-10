"""Define an append-only conversation transcript and per-model run cost records.

Transcript entries are opaque mappings stored verbatim. This observation record is separate from
the load-bearing step ledger and is never consulted to decide whether an effect ran.

`RUN_FIELDS` and `MODEL_USAGE_FIELDS` are declared here rather than in each backing store, so the
in-memory and durable implementations cannot drift into accepting different field sets. A caller
that works against one works against the other; `tests/test_transcript_conformance.py` runs the same
suite over both.
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
"""What a run record holds. Tokens are not here — they belong per model."""

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
"""What one model of one run cost. `total_tokens` is kept so a reader can tell which convention
the provider used for `prompt_tokens` rather than assuming one."""


class _Row(NamedTuple):
    """Store one transcript entry and its arrival order."""

    seq: int
    entry: dict[str, Any]


@runtime_checkable
class Transcript(Protocol):
    """Persist conversation entries and per-model run cost."""

    async def append(self, entry: dict[str, Any]) -> bool:
        """Append an entry verbatim and report whether it was new.

        Identity is `(entry["conversation_id"], entry["uuid"])`, read out of the document.
        Neither is a parameter, because a second way to name the same row is a way for the two to
        disagree — and they did: this store filed by the parameter while the durable one filed by
        the generated column, so one call could land in two different conversations.

        Idempotent on that pair, because retries happen — a crash after the model answered and
        before the transcript committed leaves a resumed run about to write the same entry again.
        """
        ...

    async def read(self, conversation_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return entries in arrival order, optionally limited to the newest tail."""
        ...

    async def record_run(self, run_id: str, fields: dict[str, Any]) -> None:
        """Merge `RUN_FIELDS` into a run record, creating it when absent.

        Merge and not replace: the row is written twice from different vantage points, once when the
        run opens and once when it ends. Raises `ValueError` on a field outside `RUN_FIELDS`, so a
        caller finds a typo here rather than discovering the cost was never written.
        """
        ...

    async def record_model_usage(self, run_id: str, model: str, counts: dict[str, Any]) -> None:
        """Merge `MODEL_USAGE_FIELDS` for one model of one run.

        Keyed by model rather than folded into the run record because one run is not always
        answered by one model — a provider-side fallback answers on a model nobody asked for, and
        rates differ. A single total plus the last model named prices the whole run at that
        model's rate. Tokens reported before any model was named key on the empty string:
        unattributed is a fact, and charging them to whichever model came later is a guess.
        """
        ...

    async def read_run(self, run_id: str) -> dict[str, Any] | None:
        """The run record, or `None` if this run was never opened."""
        ...

    async def read_model_usage(self, run_id: str) -> dict[str, dict[str, Any]]:
        """This run's token counts keyed by the model that spent them."""
        ...


def check_fields(table: str, fields: dict[str, Any], allowed: frozenset[str]) -> None:
    """Reject a field no implementation has a place for.

    Shared so that the in-memory store refuses exactly what the durable one refuses. A store that
    quietly accepted an unknown field would let a test pass and the production write fail.
    """
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
    """A copy that shares nothing with the caller's mapping.

    A shallow `dict()` would leave nested lists and dicts aliased, so mutating a message's content
    after appending it would rewrite history here and not in a database — the divergence that makes
    an in-memory store lie about a bug. Round-tripping through JSON also refuses an entry a JSONB
    column would refuse, at the point the test can see it.
    """
    copied: dict[str, Any] = json.loads(json.dumps(entry))
    return copied
