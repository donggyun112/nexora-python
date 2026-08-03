"""Performance properties the loop must keep.

These are not microbenchmarks. Each one guards a property whose loss would be invisible in
correctness tests and fatal in a server:

* the loop yields at every wait, so one conversation cannot stall the process
* concurrent conversations cost wall-clock like one, not like N
* a batch executor's concurrency survives the gate that runs before it
* per-round cost stays flat as history grows — no accidental quadratic
* event emission is cheap enough to leave on

Assertions are on *ratios* with wide margins rather than absolute times, because absolute
numbers are a CI flake generator. A regression that matters here is order-of-magnitude.
"""

import asyncio
import time
from typing import Any

import pytest

from nexora.contracts import EventEnvelope, EventStream, EventType
from nexora.engines.plain import react_loop
from tests.test_loop import Llm, Tools, a_call, says, scripted

pytestmark = pytest.mark.perf

LATENCY = 0.02
"""Stand-in for provider/tool IO. Long enough to dominate loop overhead, short enough to run."""


class SlowLlm(Llm):
    """A provider that awaits before answering, like a real one."""

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(LATENCY)
        async for chunk in super().astream(*args, **kwargs):
            yield chunk


class SlowTools(Tools):
    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        await asyncio.sleep(LATENCY)
        return await super().execute(name, call_id, arguments)


async def drain(*args: Any, **kwargs: Any) -> None:
    async for _ in react_loop(*args, **kwargs):
        pass


def a_conversation() -> tuple[SlowLlm, SlowTools]:
    """Two rounds: one tool call, then a final answer."""
    llm = SlowLlm(messages=iter([says("", a_call("c1", "read")), says("finished")]))
    llm.seen = []
    return llm, SlowTools()


# ── Concurrency ──────────────────────────────────────────────────────────────


async def test_concurrent_conversations_cost_like_one_not_like_many() -> None:
    """20 conversations must interleave. Serializing them would be ~20x and would mean the
    loop is holding something — a thread, a lock, a sync call — across an await."""
    started = time.perf_counter()
    await drain(*a_conversation(), "hi")
    alone = time.perf_counter() - started

    conversations = [a_conversation() for _ in range(20)]
    started = time.perf_counter()
    await asyncio.gather(*(drain(llm, tools, "hi") for llm, tools in conversations))
    together = time.perf_counter() - started

    assert together < alone * 3, f"20 conversations took {together / alone:.1f}x one"


async def test_the_loop_yields_at_every_wait() -> None:
    """A ticker running alongside must keep getting scheduled throughout.

    The measurement is the longest gap between ticks, not how many there were: a total count
    is diluted by all the time the loop *did* yield, so a run that blocks for a third of its
    duration still looks fine. One stalled stretch shows up in the maximum gap and nowhere
    else.
    """
    gaps: list[float] = []

    async def ticker() -> None:
        previous = time.perf_counter()
        while True:
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - previous)
            previous = now

    beat = asyncio.create_task(ticker())
    try:
        await drain(*a_conversation(), "hi")
    finally:
        beat.cancel()

    assert gaps, "the ticker never ran"
    assert max(gaps) < LATENCY / 2, (
        f"the event loop stalled for {max(gaps) * 1000:.0f}ms — something in the run "
        "is blocking instead of awaiting"
    )


async def test_a_gated_batch_still_runs_concurrently() -> None:
    """The gate runs sequentially before the batch by design. That must not turn the batch
    itself sequential — 5 concurrent tools should cost about one tool."""

    class ConcurrentBatchTools(Tools):
        async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
            async def run(c: dict[str, Any]) -> dict[str, Any]:
                await asyncio.sleep(LATENCY)
                return {"call_id": c["call_id"], "result": {"type": "text", "text": "ok"},
                        "is_error": False}

            return list(await asyncio.gather(*(run(c) for c in calls)))

    llm = scripted(says("", *[a_call(f"c{i}", "read") for i in range(5)]), says("x"))

    async def allow(c: Any) -> None:
        return None

    started = time.perf_counter()
    await drain(llm, ConcurrentBatchTools(), "hi", before_tool_call=allow)
    elapsed = time.perf_counter() - started

    assert elapsed < LATENCY * 3, f"5 concurrent tools took {elapsed / LATENCY:.1f}x one"


# ── Scaling ──────────────────────────────────────────────────────────────────


def _rounds(n: int) -> Llm:
    """n tool rounds followed by a final answer, with no simulated latency."""
    return scripted(*[says("", a_call(f"c{i}", "read")) for i in range(n)], says("finished"))


async def _time_rounds(n: int, best_of: int = 3) -> float:
    """Fastest of several runs.

    Noise only ever adds time — a GC pause, another test's import, a busy core — so the
    minimum is the measurement and the rest is interference. Averaging would let one unlucky
    run decide whether CI is red.
    """
    best = float("inf")
    for _ in range(best_of):
        started = time.perf_counter()
        await drain(_rounds(n), Tools(), "hi")
        best = min(best, time.perf_counter() - started)
    return best


async def test_per_round_cost_does_not_grow_with_history() -> None:
    """History grows every round. If anything rescans or recopies all of it per round, cost
    per round climbs and long conversations degrade quadratically."""
    await _time_rounds(20)  # warm up the interpreter before measuring

    per_round_short = await _time_rounds(20) / 20
    per_round_long = await _time_rounds(200) / 200

    assert per_round_long < per_round_short * 4, (
        f"per-round cost grew {per_round_long / per_round_short:.1f}x "
        "between a 20-round and a 200-round conversation"
    )


async def test_loop_overhead_is_a_small_fraction_of_io() -> None:
    """Whatever the loop does per round must disappear next to a real provider call."""
    rounds = 50
    overhead = await _time_rounds(rounds) / rounds

    assert overhead < LATENCY / 10, f"{overhead * 1000:.2f}ms per round of pure loop overhead"


# ── Event emission ───────────────────────────────────────────────────────────


async def test_emission_is_cheap_enough_to_leave_on() -> None:
    """Every event hashes its coordinates for a stable id. That must stay far below the cost
    of the work it describes, or observability becomes something people switch off."""

    async def sink(envelope: EventEnvelope) -> None:
        return None

    stream = EventStream(sink, session_id="s", thread_id="t", run_id="r")
    payload = {"turn": 0, "call_id": "c1", "name": "read"}

    started = time.perf_counter()
    for _ in range(10_000):
        await stream(EventType.POST_TOOL_USE, payload)
    per_event = (time.perf_counter() - started) / 10_000

    assert per_event < 50e-6, f"{per_event * 1e6:.1f}us per event"


async def test_turning_emission_on_does_not_change_the_shape_of_a_run() -> None:
    """Emission is opt-in and observational; a run with a sink attached must not cost
    materially more than one without."""
    rounds = 50
    without = await _time_rounds(rounds)

    async def sink(envelope: EventEnvelope) -> None:
        return None

    stream = EventStream(sink, session_id="s", thread_id="t", run_id="r")
    started = time.perf_counter()
    await drain(_rounds(rounds), Tools(), "hi", emit=stream)
    with_emission = time.perf_counter() - started

    assert with_emission < without * 4, (
        f"emission made the run {with_emission / without:.1f}x slower"
    )
