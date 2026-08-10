"""PostgreSQL implementation of the append-only transcript store.

Entries are persisted verbatim as JSONB. Generated columns provide indexed transcript metadata,
while run and model-usage tables store operational accounting data.
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
"""Schema for transcript entries, run metadata, and per-model usage records."""


class PostgresTranscript:
    """Persist conversation entries and per-run cost in PostgreSQL."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[Any]]) -> None:
        """Initialize the store with a caller-owned connection pool."""
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
        """Return a run record with unwritten null fields omitted."""
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
        """Merge allow-listed fields into a keyed record."""
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
