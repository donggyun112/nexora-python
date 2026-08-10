"""Implement the append-only transcript and run records with PostgreSQL.

Transcript entries are stored verbatim as JSON documents: `select entry order by seq` is that
conversation's JSONL, so a file and a table hold the same thing and neither has to be converted into
the other. Shredding a message into relational rows would mean reassembling it on every read, and a
wrong reassembly is a corrupted conversation rather than a failed query.

Every column but `seq` and `ts` is **generated from `entry`**, not copied into place by the caller,
so a column and the document it came from cannot drift — `insert` passes two values. `ts` is the one
exception: `(entry ->> 'timestamp')::timestamptz` reads the session `TimeZone` when the string
carries no offset, which makes the cast stable rather than immutable, and Postgres refuses it in a
generation expression (verified: `ERROR: generation expression is not immutable`).

`type` carries no `check` constraint on purpose. An entry kind nobody has taught this table about
must still store, so that a reader can skip it; a constraint would turn "unknown type" from a reader
concern into an insert failure, and the next entry kind into a migration.
"""

from typing import Any

from nexora_store import MODEL_USAGE_FIELDS, RUN_FIELDS, check_fields
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

__all__ = ["TRANSCRIPT_SCHEMA", "PostgresTranscript"]

TRANSCRIPT_SCHEMA = """
create table if not exists nexora_transcript (
    entry           jsonb       not null,
    seq             bigserial   primary key,
    ts              timestamptz not null,
    conversation_id text        generated always as (entry ->> 'conversation_id') stored,
    uuid            text        generated always as (entry ->> 'uuid') stored,
    parent_uuid     text        generated always as (entry ->> 'parent_uuid') stored,
    type            text        generated always as (entry ->> 'type') stored,
    schema_version  text        generated always as (entry ->> 'schema_version') stored,
    run_id          text        generated always as (entry -> 'metadata' ->> 'run_id') stored,
    unique (conversation_id, uuid)
);

create index if not exists nexora_transcript_replay
    on nexora_transcript (conversation_id, seq);

create index if not exists nexora_transcript_run
    on nexora_transcript (run_id) where run_id is not null;

create table if not exists nexora_run (
    run_id               text primary key,
    conversation_id      text,
    stop_reason          text,
    tool_calls           integer,
    interrupted_mid_turn boolean,
    started_at           timestamptz,
    ended_at             timestamptz
);

create index if not exists nexora_run_conversation
    on nexora_run (conversation_id, started_at desc);

create table if not exists nexora_run_model (
    run_id             text   not null,
    model              text   not null,
    prompt_tokens      bigint,
    completion_tokens  bigint,
    total_tokens       bigint,
    cached_tokens      bigint,
    cache_write_tokens bigint,
    cost_usd           numeric(12, 6),
    primary key (run_id, model)
);

create index if not exists nexora_run_model_by_model
    on nexora_run_model (model);
"""
"""Three tables: the conversation, the run, and what each model of that run cost.

Tokens live only in `nexora_run_model`, never on `nexora_run`. A run answered by one model is one
child row and a run that fell back to another is two, so the same query prices both — and there is
no run-level total to disagree with the per-model rows.

`total_tokens` is stored rather than derived because whether `prompt_tokens` already contains the
cache counts is a per-provider convention (see `_usage_of` in the loop). With the reported total in
hand a reader can tell which convention produced a row instead of assuming one.

`cost_usd` is stored rather than computed on read: rates change, and without a historical rate table
yesterday's cost cannot be recovered. `numeric` and not `double precision` — money that rounds
differently on two machines is money nobody can reconcile.

**No column outside a key is defaulted**, `started_at` included. A `not null default now()` reads
back as a value `MemoryTranscript` cannot produce, so a run opened the same way reported a different
field set depending on which store answered — the one divergence `tests/test_store_conformance.py`
exists to catch, and it is the same rule the token columns follow — nullable rather than
`default 0`, because a provider that reported no usage must not read back as a free run. A field
nobody wrote is absent, not invented. `TranscriptWriter.opened` writes `started_at`.
"""


class PostgresTranscript:
    """Persist conversation entries and per-run cost in PostgreSQL."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[Any]]) -> None:
        """Initialize the transcript with a pool to borrow a connection from per operation.

        A pool and not a connection, for the reason `PostgresSteps.__init__` spells out: two runs
        sharing one connection share its transaction boundary, so one run's commit publishes the
        other's half-written work. Here that would append an entry the conversation had not reached
        yet. The caller owns the pool's lifecycle; this borrows.
        """
        self._pool = pool

    async def append(self, entry: dict[str, Any]) -> bool:
        """Append an entry verbatim and report whether it was new."""
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into nexora_transcript (entry, ts)
                values (%s, coalesce((%s)::timestamptz, now()))
                on conflict (conversation_id, uuid) do nothing
                returning seq
                """,
                (Jsonb(entry), entry.get("timestamp")),
            )
            inserted = await cursor.fetchone()
        return inserted is not None

    async def read(self, conversation_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return entries in arrival order, optionally limited to the newest tail."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            if limit is None:
                await cursor.execute(
                    """
                    select entry from nexora_transcript
                    where conversation_id = %s
                    order by seq
                    """,
                    (conversation_id,),
                )
            else:
                await cursor.execute(
                    """
                    select entry from (
                        select entry, seq from nexora_transcript
                        where conversation_id = %s
                        order by seq desc
                        limit %s
                    ) tail
                    order by seq
                    """,
                    (conversation_id, max(limit, 0)),
                )
            rows = await cursor.fetchall()
        return [row["entry"] for row in rows]

    async def record_run(self, run_id: str, fields: dict[str, Any]) -> None:
        """Merge validated fields into a run record, creating it when absent."""
        await self._upsert("nexora_run", ("run_id",), (run_id,), fields, RUN_FIELDS)

    async def record_model_usage(self, run_id: str, model: str, counts: dict[str, Any]) -> None:
        """Merge validated token counts for one model of one run."""
        await self._upsert(
            "nexora_run_model", ("run_id", "model"), (run_id, model), counts, MODEL_USAGE_FIELDS
        )

    async def read_run(self, run_id: str) -> dict[str, Any] | None:
        """The run record, or `None` if this run was never opened.

        Null columns are dropped rather than returned as `None`, because the in-memory store has no
        columns and can only report a field it was given. Reporting `{"stop_reason": None}` on one
        side and `{}` on the other would make the same run read as two different facts.
        """
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                f"select {', '.join(sorted(RUN_FIELDS))} from nexora_run where run_id = %s",
                (run_id,),
            )
            row = await cursor.fetchone()
        return None if row is None else {k: v for k, v in row.items() if v is not None}

    async def read_model_usage(self, run_id: str) -> dict[str, dict[str, Any]]:
        """This run's token counts keyed by the model that spent them."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                f"""
                select model, {", ".join(sorted(MODEL_USAGE_FIELDS))}
                from nexora_run_model where run_id = %s
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return {
            row["model"]: {k: v for k, v in row.items() if k != "model" and v is not None}
            for row in rows
        }

    async def _upsert(
        self,
        table: str,
        keys: tuple[str, ...],
        key_values: tuple[Any, ...],
        fields: dict[str, Any],
        allowed: frozenset[str],
    ) -> None:
        """Insert or merge `fields` into `table` under `keys`.

        Column names are checked against an allow-list rather than interpolated as given: they
        cannot be parameterized, so an unchecked key would be a place where caller-supplied text
        reaches the statement text. An unknown key is a caller bug, and a silent no-op would hide
        it until someone queried for a cost that was never written.

        Merge and not replace, because these rows are written twice from different vantage points —
        once when the run opens knowing only where it belongs, once when it ends knowing the cost.
        A replace would erase `started_at`, which is the field that says the run began at all.
        """
        check_fields(table, fields, allowed)
        columns = sorted(fields)
        named = ", ".join((*keys, *columns))
        placeholders = ", ".join(["%s"] * (len(keys) + len(columns)))
        # No column to merge means the row only has to exist — an insert that keeps what is there.
        merge = (
            ", ".join(f"{column} = excluded.{column}" for column in columns)
            if columns
            else f"{keys[0]} = {table}.{keys[0]}"
        )
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                f"""
                insert into {table} ({named})
                values ({placeholders})
                on conflict ({", ".join(keys)}) do update set {merge}
                """,
                (*key_values, *(fields[column] for column in columns)),
            )
