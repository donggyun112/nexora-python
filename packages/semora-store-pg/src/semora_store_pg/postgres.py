"""Implement the durable step ledger with PostgreSQL."""

from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from semora_store import (
    ConversationScopedSteps,
    EffectConflict,
    ExecutionContext,
    ExecutionStore,
    ExecutionTransition,
    Fenced,
    InputRecord,
    Step,
)

__all__ = ["SCHEMA", "PostgresSteps"]

SCHEMA = """
create table if not exists ledger_step (
    branch_id      text        not null,
    key         text        not null,
    status      text        not null check (status in ('running', 'done')),
    value       jsonb,
    attempt     integer     not null default 1,
    started_at  timestamptz not null default now(),
    finished_at timestamptz,
    primary key (branch_id, key)
);

create table if not exists ledger_branch_lease (
    branch_id     text        primary key,
    owner      text        not null,
    token      bigint      not null default 1,
    expires_at timestamptz not null
);

create table if not exists ledger_input (
    sequence     bigserial   primary key,
    branch_id       text        not null,
    input_id     text        not null,
    status       text        not null,
    value        jsonb       not null,
    submitted_at timestamptz not null default now(),
    admitted_at  timestamptz,
    unique (branch_id, input_id)
);

alter table ledger_input drop constraint if exists ledger_input_status_check;
alter table ledger_input add constraint ledger_input_status_check
    check (status in ('pending', 'claimed', 'admitted', 'discarded'));

drop index if exists ledger_input_pending;
create index if not exists ledger_input_pending
    on ledger_input (branch_id, sequence) where status in ('pending', 'claimed');

create index if not exists ledger_step_running
    on ledger_step (started_at) where status = 'running';
"""
"""The three tables. The partial indexes serve recovery and operator stuck-work queries."""


class PostgresSteps:
    """Persist steps, leases, and queued inputs in PostgreSQL."""

    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[Any]]) -> None:
        """Initialize the ledger with a caller-owned connection pool."""
        self._pool = pool

    def for_execution(self, context: ExecutionContext) -> ExecutionStore:
        """The whole ledger without a conversation; one conversation's view of it with.

        The schema stays scope-neutral: a scoped branch is filed under ``"<conversation>/<branch>"``
        in the same tables, so an operator query by conversation is a prefix match on ``branch_id``.
        """
        if context.conversation_id is None:
            return self
        return ConversationScopedSteps(self, context.conversation_id)

    async def read(self, branch_id: str, key: str) -> Step:
        """Return the persisted state of a step."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "select status, value from ledger_step where branch_id = %s and key = %s",
                (branch_id, key),
            )
            row = await cursor.fetchone()
        if row is None:
            return Step("absent")
        if row["status"] == "running":
            return Step("running")
        return Step("done", row["value"])

    async def start(self, branch_id: str, key: str, token: int = 0) -> bool:
        """Atomically record new running intent after validating the fencing token."""
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                insert into ledger_step (branch_id, key, status)
                values (%s, %s, 'running')
                on conflict (branch_id, key) do nothing
                returning 1
                """,
                    (branch_id, key),
                )
                return await cursor.fetchone() is not None

    async def finish_effect(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Complete a running effect without replacing a committed result."""
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            await self._complete_effect(connection, branch_id, key, value, "running")

    async def _complete_effect(
        self,
        connection: AsyncConnection[Any],
        branch_id: str,
        key: str,
        value: Any,
        expected: str,
    ) -> None:
        """Apply one conditional effect completion inside the caller's transaction."""
        async with connection.cursor(row_factory=dict_row) as cursor:
            if expected == "absent":
                await cursor.execute(
                    """
                    insert into ledger_step (branch_id, key, status, value, finished_at)
                    values (%s, %s, 'done', %s, now())
                    on conflict (branch_id, key) do nothing
                    returning 1
                    """,
                    (branch_id, key, Jsonb(value)),
                )
            else:
                await cursor.execute(
                    """
                    update ledger_step
                    set status = 'done', value = %s, finished_at = now()
                    where branch_id = %s and key = %s and status = 'running'
                    returning 1
                    """,
                    (Jsonb(value), branch_id, key),
                )
            if await cursor.fetchone() is not None:
                return
            await cursor.execute(
                "select status, value from ledger_step where branch_id = %s and key = %s",
                (branch_id, key),
            )
            row = await cursor.fetchone()
        if row is not None and row["status"] == "done":
            if row["value"] == value:
                return
            raise EffectConflict(branch_id, key, "a different result is already committed")
        status = "absent" if row is None else str(row["status"])
        raise EffectConflict(branch_id, key, f"expected {expected!r}, found {status!r}")

    async def write_control(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Upsert mutable framework control state."""
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                insert into ledger_step (branch_id, key, status, value, finished_at)
                values (%s, %s, 'done', %s, now())
                on conflict (branch_id, key) do update
                    set status = 'done', value = excluded.value, finished_at = now()
                """,
                    (branch_id, key, Jsonb(value)),
                )

    async def finish(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Compatibility alias for ``write_control``."""
        await self.write_control(branch_id, key, value, token)

    async def _fence(self, connection: AsyncConnection[Any], branch_id: str, token: int) -> None:
        """Reject stale lease holders within the caller's transaction."""
        if not token:
            return
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                "select token from ledger_branch_lease where branch_id = %s", (branch_id,)
            )
            row = await cursor.fetchone()
        issued = row["token"] if row else 0
        if token < issued:
            raise Fenced(branch_id, token, issued)

    async def forget(self, branch_id: str, key: str, token: int = 0) -> None:
        """Remove an unfinished step after validating the fencing token."""
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "delete from ledger_step"
                    " where branch_id = %s and key = %s and status = 'running'",
                    (branch_id, key),
                )

    async def acquire(self, branch_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        """Acquire or renew a run lease and return its fencing token, or zero on contention."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                """
                insert into ledger_branch_lease (branch_id, owner, token, expires_at)
                values (%s, %s, 1, now() + make_interval(secs => %s))
                on conflict (branch_id) do update
                    set owner = excluded.owner,
                        expires_at = excluded.expires_at,
                        token = ledger_branch_lease.token
                                + (ledger_branch_lease.owner <> excluded.owner)::int
                    where ledger_branch_lease.owner = excluded.owner
                       or ledger_branch_lease.expires_at < now()
                returning token
                """,
                (branch_id, owner, ttl_seconds),
            )
            row = await cursor.fetchone()
        return int(row["token"]) if row else 0

    async def release(self, branch_id: str, owner: str) -> None:
        """Expire an owned lease without resetting its fencing token.

        Args:
            branch_id: Durable run identifier.
            owner: Current lease owner.
        """
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                """
                update ledger_branch_lease
                set owner = '', expires_at = to_timestamp(0)
                where branch_id = %s and owner = %s
                """,
                (branch_id, owner),
            )

    async def enqueue_input(self, branch_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently and report whether it was inserted."""
        async with self._pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into ledger_input (branch_id, input_id, status, value)
                values (%s, %s, 'pending', %s)
                on conflict (branch_id, input_id) do nothing
                """,
                (branch_id, input_id, Jsonb(value)),
            )
            inserted = cursor.rowcount > 0
        return inserted

    async def list_inputs(self, branch_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                """
                select input_id, status, value, sequence
                from ledger_input
                where branch_id = %s
                order by sequence
                """,
                (branch_id,),
            )
            rows = await cursor.fetchall()
        return [
            InputRecord(row["input_id"], row["status"], row["value"], int(row["sequence"]))
            for row in rows
        ]

    async def claim_input(self, branch_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                update ledger_input
                set status = 'claimed'
                where branch_id = %s and input_id = %s
                  and status not in ('admitted', 'discarded')
                """,
                    (branch_id, input_id),
                )

    async def admit_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted."""
        if not input_ids:
            return
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                update ledger_input
                set status = 'admitted', admitted_at = now()
                where branch_id = %s and input_id = any(%s) and status <> 'discarded'
                """,
                    (branch_id, input_ids),
                )

    async def discard_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Make screened-out inputs terminal without deleting their idempotency keys."""
        if not input_ids:
            return
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                update ledger_input
                set status = 'discarded'
                where branch_id = %s and input_id = any(%s)
                """,
                    (branch_id, input_ids),
                )

    async def commit_transition(
        self,
        branch_id: str,
        transition: ExecutionTransition,
        token: int = 0,
    ) -> set[str]:
        """Atomically apply typed effect, control, and input changes."""
        inserted: set[str] = set()
        async with self._pool.connection() as connection:
            await self._fence(connection, branch_id, token)
            seen: set[str] = set()
            for effect in transition.effects:
                if effect.key in seen:
                    raise ValueError(f"effect {effect.key!r} appears twice in one transition")
                seen.add(effect.key)
            overlap = seen.intersection(transition.controls)
            if overlap:
                raise ValueError(
                    f"transition classifies keys as both effect and control: {sorted(overlap)}"
                )
            for effect in transition.effects:
                await self._complete_effect(
                    connection, branch_id, effect.key, effect.value, effect.expected
                )
            async with connection.cursor() as cursor:
                for key, value in transition.controls.items():
                    await cursor.execute(
                        """
                        insert into ledger_step (branch_id, key, status, value, finished_at)
                        values (%s, %s, 'done', %s, now())
                        on conflict (branch_id, key) do update
                            set status = 'done', value = excluded.value, finished_at = now()
                        """,
                        (branch_id, key, Jsonb(value)),
                    )
                for input_id, value in transition.inputs:
                    await cursor.execute(
                        """
                        insert into ledger_input (branch_id, input_id, status, value)
                        values (%s, %s, 'pending', %s)
                        on conflict (branch_id, input_id) do nothing
                        returning input_id
                        """,
                        (branch_id, input_id, Jsonb(value)),
                    )
                    if await cursor.fetchone() is not None:
                        inserted.add(input_id)
        return inserted
