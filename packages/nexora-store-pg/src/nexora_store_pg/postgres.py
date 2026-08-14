"""Implement the durable step ledger with PostgreSQL."""

from typing import Any

from nexora_store import Fenced, InputRecord, Step
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

__all__ = ["SCHEMA", "PostgresSteps"]

SCHEMA = """
create table if not exists nexora_step (
    run_id      text        not null,
    key         text        not null,
    status      text        not null check (status in ('running', 'done')),
    value       jsonb,
    attempt     integer     not null default 1,
    started_at  timestamptz not null default now(),
    finished_at timestamptz,
    primary key (run_id, key)
);

create table if not exists nexora_run_lease (
    run_id     text        primary key,
    owner      text        not null,
    token      bigint      not null default 1,
    expires_at timestamptz not null
);

create table if not exists nexora_input (
    sequence     bigserial   primary key,
    run_id       text        not null,
    input_id     text        not null,
    status       text        not null,
    value        jsonb       not null,
    submitted_at timestamptz not null default now(),
    admitted_at  timestamptz,
    unique (run_id, input_id)
);

alter table nexora_input drop constraint if exists nexora_input_status_check;
alter table nexora_input add constraint nexora_input_status_check
    check (status in ('pending', 'claimed', 'admitted', 'discarded'));

drop index if exists nexora_input_pending;
create index if not exists nexora_input_pending
    on nexora_input (run_id, sequence) where status in ('pending', 'claimed');

create index if not exists nexora_step_running
    on nexora_step (started_at) where status = 'running';
"""
"""The three tables. The partial indexes serve recovery and operator stuck-work queries."""


class PostgresSteps:
    """Persist steps, leases, and queued inputs in PostgreSQL."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[Any]]) -> None:
        """Initialize the ledger with a caller-owned connection pool."""
        self._pool = pool

    async def read(self, run_id: str, key: str) -> Step:
        """Return the persisted state of a step."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "select status, value from nexora_step where run_id = %s and key = %s",
                (run_id, key),
            )
            row = await cursor.fetchone()
        if row is None:
            return Step("absent")
        if row["status"] == "running":
            return Step("running")
        return Step("done", row["value"])

    async def start(self, run_id: str, key: str, token: int = 0) -> bool:
        """Atomically record new running intent after validating the fencing token."""
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                insert into nexora_step (run_id, key, status)
                values (%s, %s, 'running')
                on conflict (run_id, key) do nothing
                returning 1
                """,
                    (run_id, key),
                )
                return await cursor.fetchone() is not None

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        """Record a completed step after validating the fencing token."""
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                insert into nexora_step (run_id, key, status, value, finished_at)
                values (%s, %s, 'done', %s, now())
                on conflict (run_id, key) do update
                    set status = 'done', value = excluded.value, finished_at = now()
                """,
                    (run_id, key, Jsonb(value)),
                )

    async def _fence(self, connection: AsyncConnection[Any], run_id: str, token: int) -> None:
        """Reject stale lease holders within the caller's transaction."""
        if not token:
            return
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute("select token from nexora_run_lease where run_id = %s", (run_id,))
            row = await cursor.fetchone()
        issued = row["token"] if row else 0
        if token < issued:
            raise Fenced(run_id, token, issued)

    async def forget(self, run_id: str, key: str, token: int = 0) -> None:
        """Remove an unfinished step after validating the fencing token."""
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "delete from nexora_step where run_id = %s and key = %s and status = 'running'",
                    (run_id, key),
                )

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        """Acquire or renew a run lease and return its fencing token, or zero on contention."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                """
                insert into nexora_run_lease (run_id, owner, token, expires_at)
                values (%s, %s, 1, now() + make_interval(secs => %s))
                on conflict (run_id) do update
                    set owner = excluded.owner,
                        expires_at = excluded.expires_at,
                        token = nexora_run_lease.token
                                + (nexora_run_lease.owner <> excluded.owner)::int
                    where nexora_run_lease.owner = excluded.owner
                       or nexora_run_lease.expires_at < now()
                returning token
                """,
                (run_id, owner, ttl_seconds),
            )
            row = await cursor.fetchone()
        return int(row["token"]) if row else 0

    async def release(self, run_id: str, owner: str) -> None:
        """Expire an owned lease without resetting its fencing token.

        Args:
            run_id: Durable run identifier.
            owner: Current lease owner.
        """
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                """
                update nexora_run_lease
                set owner = '', expires_at = to_timestamp(0)
                where run_id = %s and owner = %s
                """,
                (run_id, owner),
            )

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently and report whether it was inserted."""
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into nexora_input (run_id, input_id, status, value)
                values (%s, %s, 'pending', %s)
                on conflict (run_id, input_id) do nothing
                """,
                (run_id, input_id, Jsonb(value)),
            )
            inserted = cursor.rowcount > 0
        return inserted

    async def list_inputs(self, run_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                """
                select input_id, status, value, sequence
                from nexora_input
                where run_id = %s
                order by sequence
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [
            InputRecord(row["input_id"], row["status"], row["value"], int(row["sequence"]))
            for row in rows
        ]

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                update nexora_input
                set status = 'claimed'
                where run_id = %s and input_id = %s
                  and status not in ('admitted', 'discarded')
                """,
                    (run_id, input_id),
                )

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted."""
        if not input_ids:
            return
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                update nexora_input
                set status = 'admitted', admitted_at = now()
                where run_id = %s and input_id = any(%s) and status <> 'discarded'
                """,
                    (run_id, input_ids),
                )

    async def discard_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        """Make screened-out inputs terminal without deleting their idempotency keys."""
        if not input_ids:
            return
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                update nexora_input
                set status = 'discarded'
                where run_id = %s and input_id = any(%s)
                """,
                    (run_id, input_ids),
                )

    async def commit_transition(
        self,
        run_id: str,
        steps: dict[str, Any],
        inputs: list[tuple[str, dict[str, Any]]],
        token: int = 0,
    ) -> set[str]:
        """Atomically finish metadata steps and append idempotent inbox inputs."""
        inserted: set[str] = set()
        async with self._pool.connection() as connection:
            await self._fence(connection, run_id, token)
            async with connection.cursor() as cursor:
                for key, value in steps.items():
                    await cursor.execute(
                        """
                        insert into nexora_step (run_id, key, status, value, finished_at)
                        values (%s, %s, 'done', %s, now())
                        on conflict (run_id, key) do update
                            set status = 'done', value = excluded.value, finished_at = now()
                        """,
                        (run_id, key, Jsonb(value)),
                    )
                for input_id, value in inputs:
                    await cursor.execute(
                        """
                        insert into nexora_input (run_id, input_id, status, value)
                        values (%s, %s, 'pending', %s)
                        on conflict (run_id, input_id) do nothing
                        returning input_id
                        """,
                        (run_id, input_id, Jsonb(value)),
                    )
                    if await cursor.fetchone() is not None:
                        inserted.add(input_id)
        return inserted
