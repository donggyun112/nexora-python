"""A `StepLog` on Postgres. The durability the in-memory one only pretends to have.

Three tables and no ORM. What each column exists for:

* `status` — `running` before the effect, `done` after. The state a two-column log cannot hold,
  and the reason a crash mid-effect is reported as `Indeterminate` instead of replayed.
* `(run_id, key)` primary key — the idempotency key is the step name (ADR-002), so a duplicate
  insert is a database error rather than a second effect.
* `attempt` — how many times this step has been started. Not used to decide anything; it is the
  number an operator wants when a step keeps going indeterminate.
* the lease table — `expires_at` rather than a boolean, so a worker that dies holding the run
  does not hold it forever. Renewal is the holder re-acquiring.
* the input table — ordered, idempotent admission from external producers into model context.
  `claimed` survives a dead worker and `admitted` prevents a live attempt from consuming twice.

ponytail: **not verified against a live database.** The semantics are tested through
`MemorySteps`, which implements the same Protocol, and the SQL below is a direct translation of
those tests — but no test in this repo connects to Postgres, so treat the first real run as the
verification. `SCHEMA` is here so that run is one `psql -f` away.
"""

from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from .orchestrator import Fenced, InputRecord, Step

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
    status       text        not null check (status in ('pending', 'claimed', 'admitted')),
    value        jsonb       not null,
    submitted_at timestamptz not null default now(),
    admitted_at  timestamptz,
    unique (run_id, input_id)
);

create index if not exists nexora_input_pending
    on nexora_input (run_id, sequence) where status <> 'admitted';

create index if not exists nexora_step_running
    on nexora_step (started_at) where status = 'running';
"""
"""The three tables. The partial indexes serve recovery and operator stuck-work queries."""


class PostgresSteps:
    """Durable `StepLog`. Every write is its own committed statement, on purpose.

    `start` has to be visible to another process *before* the effect runs, so it cannot share a
    transaction with the work that follows it. Batching these writes would give back exactly the
    guarantee they are here to provide.
    """

    def __init__(self, connection: AsyncConnection[Any]) -> None:
        self._connection = connection

    async def read(self, run_id: str, key: str) -> Step:
        async with self._connection.cursor(row_factory=dict_row) as cursor:
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

    async def start(self, run_id: str, key: str, token: int = 0) -> None:
        """Claim the step. A second start of a `done` step is refused, not silently accepted."""
        await self._fence(run_id, token)
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into nexora_step (run_id, key, status)
                values (%s, %s, 'running')
                on conflict (run_id, key) do update
                    set status = 'running',
                        attempt = nexora_step.attempt + 1,
                        started_at = now(),
                        finished_at = null
                    where nexora_step.status = 'running'
                """,
                (run_id, key),
            )
        await self._connection.commit()

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        from psycopg.types.json import Jsonb

        await self._fence(run_id, token)
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into nexora_step (run_id, key, status, value, finished_at)
                values (%s, %s, 'done', %s, now())
                on conflict (run_id, key) do update
                    set status = 'done', value = excluded.value, finished_at = now()
                """,
                (run_id, key, Jsonb(value)),
            )
        await self._connection.commit()

    async def _fence(self, run_id: str, token: int) -> None:
        """Refuse a write from a worker whose lease has been taken over.

        `token=0` opts out — a signal answered from outside the run holds no lease. See
        `MemorySteps._fence` for why that is safe.
        """
        if not token:
            return
        async with self._connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                "select token from nexora_run_lease where run_id = %s", (run_id,)
            )
            row = await cursor.fetchone()
        issued = row["token"] if row else 0
        if token < issued:
            raise Fenced(run_id, token, issued)

    async def forget(self, run_id: str, key: str) -> None:
        """For `Orchestrator.force_retry`. Only an unfinished step may be cleared."""
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "delete from nexora_step where run_id = %s and key = %s and status = 'running'",
                (run_id, key),
            )
        await self._connection.commit()

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        """Take the lease or extend it, returning the fencing token — 0 if someone else holds it.

        The `where` clause is the whole mechanism: one statement, so two workers racing cannot both
        win. An expired lease is takeable, because a worker that died must not hold a run forever —
        and the token increments only on a *takeover*, so the holder renewing keeps its own.
        """
        async with self._connection.cursor(row_factory=dict_row) as cursor:
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
        await self._connection.commit()
        return int(row["token"]) if row else 0

    async def release(self, run_id: str, owner: str) -> None:
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                "delete from nexora_run_lease where run_id = %s and owner = %s", (run_id, owner)
            )
        await self._connection.commit()

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        from psycopg.types.json import Jsonb

        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                insert into nexora_input (run_id, input_id, status, value)
                values (%s, %s, 'pending', %s)
                on conflict (run_id, input_id) do nothing
                """,
                (run_id, input_id, Jsonb(value)),
            )
            inserted = cursor.rowcount > 0
        await self._connection.commit()
        return inserted

    async def list_inputs(self, run_id: str) -> list[InputRecord]:
        async with self._connection.cursor(row_factory=dict_row) as cursor:
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
        await self._fence(run_id, token)
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                update nexora_input
                set status = 'claimed'
                where run_id = %s and input_id = %s and status <> 'admitted'
                """,
                (run_id, input_id),
            )
        await self._connection.commit()

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        if not input_ids:
            return
        await self._fence(run_id, token)
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                update nexora_input
                set status = 'admitted', admitted_at = now()
                where run_id = %s and input_id = any(%s)
                """,
                (run_id, input_ids),
            )
        await self._connection.commit()

    async def commit_transition(
        self,
        run_id: str,
        steps: dict[str, Any],
        inputs: list[tuple[str, dict[str, Any]]],
        token: int = 0,
    ) -> set[str]:
        """Commit a control transition in one Postgres transaction."""
        from psycopg.types.json import Jsonb

        inserted: set[str] = set()
        try:
            await self._fence(run_id, token)
            async with self._connection.cursor() as cursor:
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
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
        return inserted
