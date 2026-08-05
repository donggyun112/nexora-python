"""The step ledger's two jobs: never run an effect twice, and never guess when it is unknown."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from nexora.contracts import PendingInput
from nexora.orchestrator import (
    Contended,
    Fenced,
    Indeterminate,
    MemorySteps,
    Orchestrator,
    Step,
    StepLog,
)


async def test_a_crash_leaves_an_intent_and_the_next_attempt_refuses_to_guess() -> None:
    """The whole reason the intent is written first.

    A worker that dies mid-effect writes no ending. A two-state log reports that as "never
    happened" and replays the effect; here it is `running`, and `run` refuses to decide, because
    only the caller knows whether a second charge is acceptable.
    """
    log = MemorySteps()
    sent: list[str] = []

    async def effect() -> str:
        sent.append("pharmacy")
        return "ok"

    await log.start("run-1", "meds")  # what a dead worker leaves behind

    with pytest.raises(Indeterminate) as raised:
        await Orchestrator("run-1", log).run("meds", effect)

    assert (raised.value.run_id, raised.value.step) == ("run-1", "meds")
    assert sent == []  # not sent again on a guess


async def test_a_raise_is_not_a_crash_and_does_retry() -> None:
    """A step that raises has *reported* that it did not complete, so the intent is cleared.

    That is a contract on the step function, not a guess about it: raise only when a retry is
    safe. Without the distinction, every transient provider error would need a human to unstick.
    """
    log = MemorySteps()
    attempts: list[int] = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("provider down")
        return "ok"

    with pytest.raises(RuntimeError, match="provider down"):
        await Orchestrator("run-1b", log).run("draft", flaky)

    assert (await log.read("run-1b", "draft")).status == "absent"
    assert await Orchestrator("run-1b", log).run("draft", flaky) == "ok"
    assert len(attempts) == 2


async def test_force_retry_is_the_explicit_way_out() -> None:
    """Recovering from indeterminate is a decision someone makes, not a default."""
    log = MemorySteps()
    sent: list[str] = []

    async def effect() -> str:
        sent.append("go")
        return "ok"

    await log.start("run-2", "meds")  # a previous attempt died here

    o = Orchestrator("run-2", log)
    with pytest.raises(Indeterminate):
        await o.run("meds", effect)

    await o.force_retry("meds")
    assert await o.run("meds", effect) == "ok"
    assert sent == ["go"]


async def test_two_workers_cannot_replay_one_run_at_once() -> None:
    """The failure no per-step record can catch: both workers read `absent` and both are right."""
    log = MemorySteps()

    async with Orchestrator("run-3", log, owner="worker-a"):
        with pytest.raises(Contended, match="run-3"):
            async with Orchestrator("run-3", log, owner="worker-b"):
                pass

    # Released on exit, so the next worker gets it.
    async with Orchestrator("run-3", log, owner="worker-b") as o:
        assert o.owner == "worker-b"


async def test_the_lease_does_not_block_the_holder_from_resuming() -> None:
    """A resume is the same owner coming back. Refusing that would deadlock a retry.

    And renewing keeps the same token — the number only moves on a takeover, or a worker would
    fence out its own earlier writes.
    """
    log = MemorySteps()

    first = await log.acquire("run-4", "worker-a", 60.0)
    assert first > 0
    assert await log.acquire("run-4", "worker-a", 60.0) == first  # renewal, same token
    assert await log.acquire("run-4", "worker-b", 60.0) == 0  # refused


async def test_a_stalled_worker_cannot_write_after_being_replaced() -> None:
    """The failure renewal cannot fix: a worker stalls past its TTL, another takes over, and then
    the first one wakes up still believing it holds the run. Only the token stops it."""
    log = MemorySteps()

    stalled = await log.acquire("run-4b", "worker-a", 60.0)
    await log.release("run-4b", "worker-a")  # stands in for the lease lapsing
    took_over = await log.acquire("run-4b", "worker-b", 60.0)

    assert took_over > stalled

    with pytest.raises(Fenced) as raised:
        await log.start("run-4b", "meds", stalled)
    assert (raised.value.presented, raised.value.issued) == (stalled, took_over)

    await log.start("run-4b", "meds", took_over)  # the current holder is fine
    assert (await log.read("run-4b", "meds")).status == "running"


async def test_a_writer_with_no_lease_is_not_fenced() -> None:
    """A signal is answered from outside the run by something that holds no lease — a webhook, an
    operator. Fencing it would refuse the writes the mechanism exists for."""
    log = MemorySteps()
    await log.acquire("run-4c", "worker-a", 60.0)

    await Orchestrator("run-4c", log).resolve("signoff", {"approved": True})

    assert (await log.read("run-4c", "signal:signoff")).status == "done"


async def test_a_step_that_finished_is_never_run_again() -> None:
    log = MemorySteps()
    calls: list[int] = []

    async def effect() -> str:
        calls.append(1)
        return "done"

    for _ in range(3):
        assert await Orchestrator("run-5", log).run("send", effect) == "done"

    assert calls == [1]


async def test_memory_steps_satisfies_the_protocol() -> None:
    """Declared, so the Postgres implementation has a shape to match rather than a guess."""
    assert isinstance(MemorySteps(), StepLog)


async def test_input_queue_reclaims_missing_transcript_but_skips_represented_input() -> None:
    log = MemorySteps()
    first = Orchestrator("input-run", log)
    queued = await first.enqueue_input(
        PendingInput("user_prompt", HumanMessage("hello"), "prompt-1")
    )

    assert [record.status for record in await log.list_inputs("input-run")] == ["pending"]
    claimed = await first.claim_inputs([])
    assert claimed == [queued]
    assert [record.status for record in await log.list_inputs("input-run")] == ["claimed"]

    await first.admit_inputs(claimed)
    assert [record.status for record in await log.list_inputs("input-run")] == ["admitted"]

    # A fresh process with no transcript safely replays even an admitted input.
    assert await Orchestrator("input-run", log).claim_inputs([]) == [queued]
    # Once the transcript carries the queue/message id, it is not injected twice.
    assert await Orchestrator("input-run", log).claim_inputs([queued.message]) == []


async def test_an_interruption_is_not_recorded_as_the_answer() -> None:
    """An aborted run is what a resume exists to get past; freezing it defeats the point.

    Same family as recording a failure or a suspension. `stop_reason == "policy"` is the one that
    does record — a supervisor's decision is an answer, and repeating the run reaches it again.
    """
    from nexora.engines.plain import react_loop
    from nexora.orchestrator import AgentAborted, run_agent
    from tests.test_loop import Tools, says, scripted

    log = MemorySteps()
    calls: list[str] = []

    async def attempt() -> Any:
        calls.append("call")
        return await run_agent(
            react_loop(scripted(says("never")), Tools(), aborted=lambda: True)
        )

    with pytest.raises(AgentAborted):
        await Orchestrator("run-7", log).run("draft", attempt)

    assert (await log.read("run-7", "draft")).status == "absent"

    # Not interrupted this time, so the step actually completes.
    async def finishes() -> Any:
        calls.append("call")
        return await run_agent(react_loop(scripted(says("drafted")), Tools()))

    outcome = await Orchestrator("run-7", log).run("draft", finishes)
    assert outcome["content"] == "drafted"
    assert len(calls) == 2


async def test_a_signal_is_a_step_that_only_an_outsider_can_finish() -> None:
    log = MemorySteps()
    assert (await log.read("run-6", "signal:signoff")) == Step("absent")

    await Orchestrator("run-6", log).resolve("signoff", {"approved": True})

    assert (await log.read("run-6", "signal:signoff")).status == "done"
    assert await Orchestrator("run-6", log).signal("signoff") == {"approved": True}


async def test_the_postgres_log_matches_the_protocol() -> None:
    """Shape only. No test here connects to a database — see the module docstring."""
    from nexora.steps_postgres import SCHEMA, PostgresSteps

    assert issubclass(PostgresSteps, object)
    assert isinstance(PostgresSteps(_NoConnection()), StepLog)  # type: ignore[arg-type]
    assert "nexora_run_lease" in SCHEMA
    assert "nexora_input" in SCHEMA
    assert "expires_at < now()" not in SCHEMA  # the takeover clause lives in the query, not the DDL


class _NoConnection:
    """Enough of a connection to satisfy construction. Any use of it is a test bug."""

    def cursor(self, **_: Any) -> Any:
        raise AssertionError("this test must not talk to a database")
