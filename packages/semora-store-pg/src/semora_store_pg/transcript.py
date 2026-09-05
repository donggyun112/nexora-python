"""PostgreSQL implementation of the append-only transcript store.

Entries are persisted verbatim as JSONB. Generated columns provide indexed transcript metadata,
while run and model-usage tables store operational accounting data.
"""

from typing import Any, Self

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from semora_store import BRANCH_FIELDS, MODEL_USAGE_FIELDS, ExecutionContext, check_fields

__all__ = ["TRANSCRIPT_SCHEMA", "PostgresTranscript"]

TRANSCRIPT_SCHEMA = """
create table if not exists ledger_transcript (
    entry           jsonb       not null,
    seq             bigserial   primary key,
    ts              timestamptz not null,
    conversation_id text        generated always as (entry ->> 'conversation_id') stored,
    uuid            text        generated always as (entry ->> 'uuid') stored,
    parent_uuid     text        generated always as (entry ->> 'parent_uuid') stored,
    type            text        generated always as (entry ->> 'type') stored,
    schema_version  text        generated always as (entry ->> 'schema_version') stored,
    branch_id          text        generated always as (entry -> 'metadata' ->> 'branch_id') stored,
    unique (conversation_id, uuid)
);

create index if not exists ledger_transcript_replay
    on ledger_transcript (conversation_id, seq);

create index if not exists ledger_transcript_run
    on ledger_transcript (branch_id) where branch_id is not null;

create table if not exists ledger_branch (
    branch_id               text primary key,
    conversation_id      text,
    stop_reason          text,
    tool_calls           integer,
    interrupted_mid_turn boolean,
    started_at           timestamptz,
    ended_at             timestamptz
);

create index if not exists ledger_run_conversation
    on ledger_branch (conversation_id, started_at desc);

create table if not exists ledger_branch_model (
    branch_id             text   not null,
    model              text   not null,
    prompt_tokens      bigint,
    completion_tokens  bigint,
    total_tokens       bigint,
    cached_tokens      bigint,
    cache_write_tokens bigint,
    cost_usd           numeric(12, 6),
    primary key (branch_id, model)
);

create index if not exists ledger_branch_model_by_model
    on ledger_branch_model (model);
"""
"""Schema for transcript entries, run metadata, and per-model usage records."""


class PostgresTranscript:
    """Persist conversation entries and per-run cost in PostgreSQL."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[Any]]) -> None:
        """Initialize the store with a caller-owned connection pool."""
        self._pool = pool

    def for_execution(self, context: ExecutionContext) -> Self:
        """Return this adapter because its default schema is scope-neutral."""
        del context
        return self

    async def append(self, entry: dict[str, Any]) -> bool:
        """Append an entry verbatim and report whether it was new."""
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into ledger_transcript (entry, ts)
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
                    select entry from ledger_transcript
                    where conversation_id = %s
                    order by seq
                    """,
                    (conversation_id,),
                )
            else:
                await cursor.execute(
                    """
                    select entry from (
                        select entry, seq from ledger_transcript
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

    async def record_branch(self, branch_id: str, fields: dict[str, Any]) -> None:
        """Merge validated fields into a run record, creating it when absent."""
        await self._upsert("ledger_branch", ("branch_id",), (branch_id,), fields, BRANCH_FIELDS)

    async def record_model_usage(self, branch_id: str, model: str, counts: dict[str, Any]) -> None:
        """Merge validated token counts for one model of one run."""
        await self._upsert(
            "ledger_branch_model",
            ("branch_id", "model"),
            (branch_id, model),
            counts,
            MODEL_USAGE_FIELDS,
        )

    async def read_branch(self, branch_id: str) -> dict[str, Any] | None:
        """Return a run record with unwritten null fields omitted."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                f"select {', '.join(sorted(BRANCH_FIELDS))} from ledger_branch"
                " where branch_id = %s",
                (branch_id,),
            )
            row = await cursor.fetchone()
        return None if row is None else {k: v for k, v in row.items() if v is not None}

    async def read_model_usage(self, branch_id: str) -> dict[str, dict[str, Any]]:
        """This run's token counts keyed by the model that spent them."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                f"""
                select model, {", ".join(sorted(MODEL_USAGE_FIELDS))}
                from ledger_branch_model where branch_id = %s
                """,
                (branch_id,),
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
