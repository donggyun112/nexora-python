"""One suite, both stores. The only thing that keeps the in-memory and durable halves the same.

The memory implementations exist so a caller can find out whether its step boundaries and entry
shapes are right before paying for a database. That promise is worth nothing if the two answer
differently — a suite that passes in memory and fails on Postgres is worse than having no memory
store at all, because it moves the discovery from the test to production. `postgres.py` said as much
in its own docstring: *the semantics are tested through `MemorySteps` … but no test in this repo
connects to Postgres.* This file is what that sentence was waiting for.

So every property here runs over both `StepLog` implementations and both `Transcript` ones. The
Postgres half is skipped unless `NEXORA_TEST_DSN` is set, so **it is still unverified in an ordinary
run** — but the skip is visible, which is the point. Point it at a scratch database to close it:

    NEXORA_TEST_DSN=postgresql://localhost/nexora_test uv run pytest tests/test_store_conformance.py

Divergences it was written to catch, all now fixed:

* the in-memory transcript accepted run fields Postgres has no column for;
* the two disagreed about what identifies an entry, and about which conversation it belonged to;
* `read_run` / `read_model_usage` existed on one side only;
* `MemorySteps.start` reopened a `done` step, so the in-memory ledger alone could lose a committed
  result on replay;
* `MemorySteps` ignored `ttl_seconds`, making lease takeover unreachable there;
* `MemorySteps.claim_input` raised `KeyError` where the durable store no-ops;
* `PostgresSteps.release` deleted the lease row, so the next token restarted at 1 and a stale worker
  passed the fence.

The last four are why the parity test alone is not enough: every one of them was a method both
stores had, doing different things.
"""

import inspect
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from nexora_store import Fenced, MemorySteps, MemoryTranscript, StepLog, Transcript
from nexora_store_pg import SCHEMA, TRANSCRIPT_SCHEMA, PostgresSteps, PostgresTranscript
from psycopg_pool import AsyncConnectionPool

pytestmark = pytest.mark.anyio

DSN = os.environ.get("NEXORA_TEST_DSN")


async def _postgres(schema: str, tables: str, cls: Any) -> AsyncIterator[Any]:
    """One store on a freshly truncated schema, for the span of one test.

    The pool is opened here and closed on the way out, because that lifecycle is the caller's — the
    stores borrow and never own.

    **A pool of exactly one, with a short timeout, on purpose.** Every store method must borrow, use
    and return a connection before it needs another; a method that borrows a second one while still
    holding the first cannot be served and fails with `PoolTimeout`. A larger pool would serve that
    nesting and let the bug through — which is the trap `_fence` was one edit away from, since it
    reads the lease and the write it guards must land in the same transaction. Two seconds rather
    than the 30-second default so the failure is a fast red, not a stalled suite.
    """
    async with AsyncConnectionPool(
        str(DSN), min_size=1, max_size=1, timeout=2.0, open=False
    ) as pool:
        await pool.open(wait=True)
        async with pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(schema)
            await cursor.execute(f"truncate {tables} restart identity")
        yield cls(pool)


@pytest.fixture(params=["memory", "postgres"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[Transcript]:
    """Each test body runs once per implementation."""
    if request.param == "memory":
        yield MemoryTranscript()
        return
    if not DSN:
        pytest.skip("set NEXORA_TEST_DSN to run the durable half of this suite")
    async for durable in _postgres(
        TRANSCRIPT_SCHEMA,
        "nexora_transcript, nexora_run, nexora_run_model",
        PostgresTranscript,
    ):
        yield durable


@pytest.fixture(params=["memory", "postgres"])
async def steps(request: pytest.FixtureRequest) -> AsyncIterator[StepLog]:
    """The same arrangement for the ledger — its durable half was equally unverified."""
    if request.param == "memory":
        yield MemorySteps()
        return
    if not DSN:
        pytest.skip("set NEXORA_TEST_DSN to run the durable half of this suite")
    async for durable in _postgres(
        SCHEMA, "nexora_step, nexora_run_lease, nexora_input", PostgresSteps
    ):
        yield durable


# ── surface parity ───────────────────────────────────────────────────────────


def _api(cls: Any) -> set[str]:
    return {name for name, value in inspect.getmembers(cls, callable) if not name.startswith("_")}


@pytest.mark.parametrize(
    ("protocol", "memory", "postgres"),
    [(StepLog, MemorySteps, PostgresSteps), (Transcript, MemoryTranscript, PostgresTranscript)],
)
def test_both_implementations_expose_the_same_surface(
    protocol: Any, memory: Any, postgres: Any
) -> None:
    """A method on one store and not the other is a caller that works until it is deployed.

    Structural typing will not catch it: a `MemoryTranscript` passed where a `Transcript` is wanted
    satisfies the Protocol whatever extra it carries, and the extra is exactly what a test reaches
    for. `forget` used to be the standing example — declared on neither Protocol while both stores
    had it and the orchestrator reached for it behind a capability check.
    """
    assert _api(memory) == _api(postgres)
    declared = {name for name in dir(protocol) if not name.startswith("_")}
    assert declared <= _api(memory)


# ── ledger ───────────────────────────────────────────────────────────────────


async def test_an_unrecorded_step_is_absent(steps: StepLog) -> None:
    assert (await steps.read("run-1", "meds")).status == "absent"


async def test_a_started_step_is_running_until_it_finishes(steps: StepLog) -> None:
    """The state a two-state log cannot hold, and the one a crash mid-effect leaves behind."""
    await steps.start("run-1", "meds")

    assert (await steps.read("run-1", "meds")).status == "running"


async def test_only_the_first_caller_records_step_intent(steps: StepLog) -> None:
    """The start result is the atomic right to execute the effect."""
    assert await steps.start("run-1", "meds") is True

    assert await steps.start("run-1", "meds") is False


async def test_a_finished_step_reports_its_value(steps: StepLog) -> None:
    await steps.start("run-1", "meds")

    await steps.finish("run-1", "meds", {"dispensed": True})

    record = await steps.read("run-1", "meds")
    assert record.status == "done"
    assert record.value == {"dispensed": True}


async def test_a_second_worker_is_refused_the_lease(steps: StepLog) -> None:
    first = await steps.acquire("run-1", "worker-a", 60.0)

    assert first > 0
    assert await steps.acquire("run-1", "worker-b", 60.0) == 0


async def test_the_holder_renewing_keeps_its_token(steps: StepLog) -> None:
    """A token that moved on renewal would fence the holder's own in-flight writes."""
    first = await steps.acquire("run-1", "worker-a", 60.0)

    assert await steps.acquire("run-1", "worker-a", 60.0) == first


async def test_a_write_from_a_replaced_worker_is_fenced(steps: StepLog) -> None:
    """Renewal cannot make a lease safe — only the token can, which is why it is carried."""
    stale = await steps.acquire("run-1", "worker-a", 60.0)
    await steps.release("run-1", "worker-a")
    current = await steps.acquire("run-1", "worker-b", 60.0)
    assert current > stale

    with pytest.raises(Fenced):
        await steps.finish("run-1", "meds", {"late": True}, stale)


async def test_a_released_lease_does_not_hand_out_an_earlier_token(steps: StepLog) -> None:
    """The token must never go backwards.

    Deleting the lease row on release sent the next `acquire` down the insert path and restarted the
    token at 1, so a worker still holding token 1 passed the fence — the exact write `Fenced` exists
    to refuse. Both stores now expire the row and keep the counter.
    """
    first = await steps.acquire("run-1", "worker-a", 60.0)
    await steps.release("run-1", "worker-a")

    assert await steps.acquire("run-1", "worker-b", 60.0) > first


async def test_an_expired_lease_can_be_taken_over(steps: StepLog) -> None:
    """The scenario the lease exists for: a worker stalls past its TTL and another takes the run.

    Unreachable in memory until `ttl_seconds` was honoured there, which is why no test covered it —
    a store that never expires a lease can never hand one over.
    """
    stalled = await steps.acquire("run-1", "worker-a", 0.0)

    taken = await steps.acquire("run-1", "worker-b", 60.0)

    assert taken > stalled


async def test_a_step_that_already_finished_is_not_reopened(steps: StepLog) -> None:
    """Replay must not throw away a committed result.

    `start` on a `done` step used to reset it to `running` in memory, so a replayed run re-ran the
    effect and discarded what the first attempt had recorded — the one thing the ledger exists to
    prevent, broken only in the store used to check the ledger.
    """
    await steps.start("run-1", "meds")
    await steps.finish("run-1", "meds", {"dispensed": True})

    inserted = await steps.start("run-1", "meds")

    assert inserted is False
    record = await steps.read("run-1", "meds")
    assert record.status == "done"
    assert record.value == {"dispensed": True}


async def test_claiming_an_input_that_is_not_there_does_nothing(steps: StepLog) -> None:
    """A no-op, not a `KeyError`. The durable store expresses it as an update matching no row."""
    await steps.claim_input("run-1", "never-enqueued")

    assert await steps.list_inputs("run-1") == []


async def test_an_unleased_write_is_not_fenced(steps: StepLog) -> None:
    """Token zero means "I hold no lease" — a webhook answering a signal is not a stale worker."""
    await steps.acquire("run-1", "worker-a", 60.0)

    await steps.finish("run-1", "meds", {"ok": True}, 0)

    assert (await steps.read("run-1", "meds")).status == "done"


async def test_inputs_are_appended_idempotently_and_ordered(steps: StepLog) -> None:
    assert await steps.enqueue_input("run-1", "in-1", {"text": "one"}) is True
    assert await steps.enqueue_input("run-1", "in-1", {"text": "one"}) is False
    await steps.enqueue_input("run-1", "in-2", {"text": "two"})

    assert [record.input_id for record in await steps.list_inputs("run-1")] == ["in-1", "in-2"]


async def test_an_admitted_input_is_not_reclaimed(steps: StepLog) -> None:
    """`admitted` is what stops a live attempt consuming an input twice."""
    await steps.enqueue_input("run-1", "in-1", {"text": "one"})
    await steps.admit_inputs("run-1", ["in-1"])

    await steps.claim_input("run-1", "in-1")

    assert [r.status for r in await steps.list_inputs("run-1")] == ["admitted"]


async def test_a_discarded_input_is_terminal(steps: StepLog) -> None:
    """An ingress screen's removal must survive a fresh attempt."""
    await steps.enqueue_input("run-1", "in-1", {"text": "one"})
    await steps.claim_input("run-1", "in-1")

    await steps.discard_inputs("run-1", ["in-1"])
    await steps.claim_input("run-1", "in-1")
    await steps.admit_inputs("run-1", ["in-1"])

    assert [record.status for record in await steps.list_inputs("run-1")] == ["discarded"]


async def test_a_transition_commits_steps_and_inputs_together(steps: StepLog) -> None:
    """The atomicity that keeps a cancellation ahead of its replacement across a crash."""
    inserted = await steps.commit_transition(
        "run-1", {"turn": {"n": 1}}, [("in-1", {"text": "one"})]
    )

    assert inserted == {"in-1"}
    assert (await steps.read("run-1", "turn")).value == {"n": 1}
    assert [r.input_id for r in await steps.list_inputs("run-1")] == ["in-1"]


async def test_a_transition_does_not_re_insert_an_input_it_already_holds(steps: StepLog) -> None:
    await steps.enqueue_input("run-1", "in-1", {"text": "one"})

    inserted = await steps.commit_transition("run-1", {}, [("in-1", {"text": "one"})])

    assert inserted == set()


async def test_forgetting_a_step_clears_the_intent(steps: StepLog) -> None:
    """What every retryable failure needs, not just `force_retry`.

    A step that raised, a model request that failed before its first chunk, and an aborted stream
    all end by clearing their intent. That is why `StepLog` declares `forget` rather than treating
    it as a capability to check for: a ledger that could skip it would turn all three into a
    permanent `Indeterminate` without saying so.
    """
    await steps.start("run-1", "meds")

    await steps.forget("run-1", "meds")

    assert (await steps.read("run-1", "meds")).status == "absent"


async def test_forgetting_a_finished_step_leaves_its_result_alone(steps: StepLog) -> None:
    """The invariant that keeps clearing from becoming a way to replay a recorded effect.

    Every retryable failure clears, so a ledger that deleted by key alone would erase results the
    run is entitled to replay instead of re-executing. The two implementations word the guard
    differently — one skips a `done` entry, the other deletes only `running` rows — and this is
    what says they must agree.
    """
    await steps.start("run-1", "meds")
    await steps.finish("run-1", "meds", {"dispensed": True})

    await steps.forget("run-1", "meds")

    record = await steps.read("run-1", "meds")
    assert record.status == "done"
    assert record.value == {"dispensed": True}


async def test_forgetting_from_a_replaced_worker_is_fenced(steps: StepLog) -> None:
    """The one write where a missing fence loses an effect instead of just failing.

    A stalled worker does not know it was replaced — `_renew` keeps its old token — so on its way
    out it clears the step it thinks is its own. Unfenced, that deletes the running intent of the
    worker now executing that step, and the next attempt reads `absent` and charges twice.
    """
    stale = await steps.acquire("run-1", "worker-a", 60.0)
    await steps.release("run-1", "worker-a")
    current = await steps.acquire("run-1", "worker-b", 60.0)
    await steps.start("run-1", "meds", current)

    with pytest.raises(Fenced):
        await steps.forget("run-1", "meds", stale)

    assert (await steps.read("run-1", "meds")).status == "running"


# ── transcript ───────────────────────────────────────────────────────────────


def an_entry(uuid: str, **fields: Any) -> dict[str, Any]:
    return {"uuid": uuid, "conversation_id": "conv-1", "type": "human", **fields}


# ── append ───────────────────────────────────────────────────────────────────


async def test_an_entry_is_returned_exactly_as_it_was_appended(store: Transcript) -> None:
    """Verbatim is the whole claim.

    A store that normalized the document would break the round trip, and one that only *mostly*
    preserves it would break it later and quietly.
    """
    entry = an_entry("a", message={"type": "human", "data": {"content": "hi", "tags": ["x"]}})

    await store.append(entry)

    assert await store.read("conv-1") == [entry]


async def test_appending_the_same_uuid_twice_is_absorbed(store: Transcript) -> None:
    assert await store.append(an_entry("a")) is True
    assert await store.append(an_entry("a")) is False
    assert len(await store.read("conv-1")) == 1


async def test_the_same_uuid_in_another_conversation_is_a_different_entry(
    store: Transcript,
) -> None:
    """Identity is `(conversation, uuid)`.

    Two conversations replaying the same words derive the same chain hash, and a store keyed on uuid
    alone would swallow the second one.
    """
    await store.append(an_entry("a"))

    assert await store.append({**an_entry("a"), "conversation_id": "conv-2"}) is True


async def test_a_stored_entry_does_not_change_when_the_caller_mutates_its_input(
    store: Transcript,
) -> None:
    """A shallow copy would leave nested content aliased in memory and not in a database."""
    nested: dict[str, Any] = {"blocks": ["one"]}
    entry = an_entry("a", message=nested)
    await store.append(entry)

    nested["blocks"].append("two")

    assert (await store.read("conv-1"))[0]["message"] == {"blocks": ["one"]}


# ── read ─────────────────────────────────────────────────────────────────────


async def test_entries_come_back_in_arrival_order(store: Transcript) -> None:
    for uuid in "abc":
        await store.append(an_entry(uuid))

    assert [e["uuid"] for e in await store.read("conv-1")] == ["a", "b", "c"]


async def test_a_limit_takes_the_newest_tail_in_arrival_order(store: Transcript) -> None:
    for uuid in "abc":
        await store.append(an_entry(uuid))

    assert [e["uuid"] for e in await store.read("conv-1", limit=2)] == ["b", "c"]


@pytest.mark.parametrize("limit", [0, -1])
async def test_a_non_positive_limit_returns_nothing(store: Transcript, limit: int) -> None:
    await store.append(an_entry("a"))

    assert await store.read("conv-1", limit=limit) == []


async def test_a_limit_larger_than_the_conversation_returns_all_of_it(store: Transcript) -> None:
    await store.append(an_entry("a"))

    assert len(await store.read("conv-1", limit=99)) == 1


async def test_an_unknown_conversation_reads_as_empty_rather_than_raising(
    store: Transcript,
) -> None:
    assert await store.read("never-existed") == []


# ── run records ──────────────────────────────────────────────────────────────


async def test_an_unopened_run_reads_as_none(store: Transcript) -> None:
    assert await store.read_run("never-ran") is None


async def test_run_fields_merge_across_writes(store: Transcript) -> None:
    """The property `opened`/`closed` depends on: two writes from different vantage points."""
    await store.record_run("run-1", {"conversation_id": "conv-1"})

    await store.record_run("run-1", {"stop_reason": "completed"})

    row = await store.read_run("run-1")
    assert row is not None
    assert row["conversation_id"] == "conv-1"
    assert row["stop_reason"] == "completed"


async def test_a_run_reads_back_exactly_the_fields_that_were_recorded(store: Transcript) -> None:
    """Field sets, not keys one at a time.

    A column default invents a field the memory store has no way to produce, so the same run read as
    two different facts — and every assertion here checked keys individually, which cannot see it.
    """
    await store.record_run("run-1", {"conversation_id": "conv-1"})

    assert await store.read_run("run-1") == {"conversation_id": "conv-1"}


async def test_a_field_no_table_has_a_column_for_is_refused(store: Transcript) -> None:
    """The divergence this suite was written for.

    The in-memory store used to accept anything, so a typo passed every test and failed the first
    real write.
    """
    with pytest.raises(ValueError, match="prompt_tokens"):
        await store.record_run("run-1", {"prompt_tokens": 10})


async def test_a_field_outside_the_per_model_set_is_refused(store: Transcript) -> None:
    with pytest.raises(ValueError, match="stop_reason"):
        await store.record_model_usage("run-1", "m", {"stop_reason": "completed"})


async def test_an_empty_write_still_creates_the_row(store: Transcript) -> None:
    """`opened()` has to register a run before it knows anything else about it."""
    await store.record_run("run-1", {})

    assert await store.read_run("run-1") is not None


# ── per-model usage ──────────────────────────────────────────────────────────


async def test_a_run_with_no_usage_has_no_model_rows(store: Transcript) -> None:
    await store.record_run("run-1", {})

    assert await store.read_model_usage("run-1") == {}


async def test_usage_is_kept_per_model(store: Transcript) -> None:
    await store.record_run("run-1", {})

    await store.record_model_usage("run-1", "opus", {"prompt_tokens": 10})
    await store.record_model_usage("run-1", "haiku", {"prompt_tokens": 20})

    assert await store.read_model_usage("run-1") == {
        "opus": {"prompt_tokens": 10},
        "haiku": {"prompt_tokens": 20},
    }


async def test_unattributed_tokens_key_on_the_empty_model(store: Transcript) -> None:
    """A provider that never named a model still spent tokens.

    Charging them to a guess is worse than leaving them unattributed.
    """
    await store.record_run("run-1", {})

    await store.record_model_usage("run-1", "", {"prompt_tokens": 7})

    assert await store.read_model_usage("run-1") == {"": {"prompt_tokens": 7}}


async def test_per_model_counts_merge_across_writes(store: Transcript) -> None:
    await store.record_run("run-1", {})
    await store.record_model_usage("run-1", "opus", {"prompt_tokens": 10})

    await store.record_model_usage("run-1", "opus", {"completion_tokens": 2})

    assert await store.read_model_usage("run-1") == {
        "opus": {"prompt_tokens": 10, "completion_tokens": 2}
    }


async def test_an_unwritten_count_is_absent_rather_than_zero(store: Transcript) -> None:
    """Absent and zero are different facts all the way down.

    A `0` here prices an unmeasured run as free, and the two stores must agree on which of the two
    they are reporting.
    """
    await store.record_run("run-1", {})

    await store.record_model_usage("run-1", "opus", {"prompt_tokens": 10})

    assert "cached_tokens" not in (await store.read_model_usage("run-1"))["opus"]
