"""Loop semantics pinned against react.ts. Fakes, not mocks."""

from collections.abc import AsyncIterator
from typing import Any

from nexora import LLMMessage, react_loop


def call(cid: str, name: str, args: str = "{}") -> list[dict[str, Any]]:
    return [
        {"type": "tool_call_start", "id": cid, "name": name},
        {"type": "tool_call_delta", "id": cid, "delta": args},
    ]


def done(content: str = "") -> dict[str, Any]:
    return {"type": "done", "content": content, "stop_reason": "end_turn"}


class Llm:
    def __init__(self, *turns: list[dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.seen: list[list[LLMMessage]] = []

    def stream(self, messages: list[LLMMessage]) -> AsyncIterator[dict[str, Any]]:
        self.seen.append(list(messages))
        chunks = self.turns.pop(0) if self.turns else [done()]

        async def gen() -> AsyncIterator[dict[str, Any]]:
            for chunk in chunks:
                yield chunk

        return gen()


class Tools:
    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        defs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.results = results or {}
        self.defs = defs or {}
        self.ran: list[str] = []

    async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
        self.ran.append(name)
        return self.results.get(name, {"type": "text", "text": "ok"})

    def get(self, name: str) -> dict[str, Any] | None:
        return self.defs.get(name)


def cap(turns: int) -> Any:
    """Stop after N rounds — the loop has no built-in limit, callers set one."""

    async def hook(turn: int, content: str, calls: list[Any]) -> bool:
        return turn + 1 >= turns

    return hook


async def run(llm: Llm, tools: Tools | None = None, **kw: Any) -> list[dict[str, Any]]:
    kw.setdefault("should_stop_after_turn", cap(10))
    return [e async for e in react_loop(llm, tools or Tools(), "hi", **kw)]


async def test_no_tool_calls_ends_the_turn() -> None:
    events = await run(Llm([{"type": "text_delta", "delta": "yo"}, done("yo")]))
    assert [e["type"] for e in events] == ["text", "done"]
    assert events[-1]["content"] == "yo"


async def test_tool_round_feeds_back_into_the_model() -> None:
    llm = Llm([*call("c1", "read"), done("working")], [done("finished")])
    tools = Tools()
    events = await run(llm, tools)
    assert tools.ran == ["read"]
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "done"]
    assert len(llm.seen) == 2


async def test_malformed_tool_args_degrade_to_empty() -> None:
    llm = Llm([*call("c1", "read", "{bad json"), done()], [done("x")])
    events = await run(llm)
    assert next(e for e in events if e["type"] == "tool_call")["input"] == {}


async def test_exclusive_tool_runs_alone() -> None:
    """loop-helpers.ts:27 — the rest are re-issued next round."""
    llm = Llm([*call("c1", "read"), *call("c2", "danger"), done()], [done("x")])
    tools = Tools(defs={"danger": {"is_exclusive": True}})
    await run(llm, tools)
    assert tools.ran == ["danger"]


async def test_terminating_tool_ends_the_run() -> None:
    """react.ts:263 — `any` over the batch, not `every`."""
    llm = Llm([*call("c1", "submit"), done("bye")])
    tools = Tools(defs={"submit": {"terminates_loop": True}})
    events = await run(llm, tools)
    assert events[-1]["type"] == "done"
    assert len(llm.seen) == 1


async def test_failed_terminating_tool_gets_a_recovery_round() -> None:
    """react.ts:264 — `!isError` guard. An errored submit must not end the run."""
    llm = Llm([*call("c1", "submit"), done()], [done("recovered")])
    tools = Tools(
        results={"submit": {"type": "error", "message": "nope"}},
        defs={"submit": {"terminates_loop": True}},
    )
    events = await run(llm, tools)
    assert events[-1]["content"] == "recovered"
    assert len(llm.seen) == 2


async def test_suspend_stops_the_batch_and_keeps_completed_results() -> None:
    """react.ts:232 — completed results survive; the suspended call is handed off."""
    llm = Llm([*call("c1", "read"), *call("c2", "ask"), done()])
    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "p1", "handle": "job-9"}},
    )
    captured: list[Any] = []

    async def on_suspend(
        call: Any, result: dict[str, Any], msgs: list[LLMMessage], blocks: list[Any]
    ) -> None:
        captured.append((call, result, blocks))

    events = await run(llm, tools, on_suspend=on_suspend)
    assert [e["type"] for e in events][-1] == "suspended"
    assert tools.ran == ["read", "ask"]
    suspended_call, result, blocks = captured[0]
    # The whole suspend result reaches the caller, handle included — the loop names no
    # suspension-record fields of its own.
    assert (suspended_call.id, suspended_call.name) == ("c2", "ask")
    assert result == {"type": "suspend", "pending_id": "p1", "handle": "job-9"}
    assert [b["id"] for b in blocks] == ["c1"]


async def test_policy_hook_runs_even_when_a_tool_already_stopped_the_run() -> None:
    """react.ts:266 — the hook does budget accounting, so no early return."""
    llm = Llm([*call("c1", "submit"), done()])
    tools = Tools(defs={"submit": {"terminates_loop": True}})
    asked: list[int] = []

    async def policy(turn: int, content: str, calls: list[Any]) -> bool:
        asked.append(turn)
        return False

    await run(llm, tools, should_stop_after_turn=policy)
    assert asked == [0]


async def test_steer_arriving_at_the_end_resumes_instead_of_completing() -> None:
    """react.ts:152 — a late steer cancels the stop and buys another model call."""
    llm = Llm([done("almost")], [done("really done")])
    drains = 0

    def drain() -> list[LLMMessage]:
        # Arrives while the first turn is finishing — the second drain, not the first.
        nonlocal drains
        drains += 1
        return [{"role": "user", "content": "wait"}] if drains == 2 else []

    events = await run(llm, drain_steers=drain)
    assert events[-1]["content"] == "really done"
    assert len(llm.seen) == 2


async def test_abort_leaves_a_record_instead_of_a_stream_that_just_stops() -> None:
    llm = Llm([done("never")])
    events = await run(llm, aborted=lambda: True)
    assert [e["type"] for e in events] == ["done"]
    assert events[-1]["stop_reason"] == "aborted"
    assert llm.seen == []


async def test_a_provider_failure_is_reported_not_raised() -> None:
    """react.ts:131 — the run ends with an `error` event, no exception escapes."""

    class Broken:
        def stream(self, messages: list[LLMMessage]) -> Any:
            async def gen() -> Any:
                raise RuntimeError("429 rate limited")
                yield  # pragma: no cover - unreachable, makes this an async generator

            return gen()

    events = [e async for e in react_loop(Broken(), Tools(), "hi")]
    assert events == [{"type": "error", "message": "429 rate limited"}]


async def test_the_gate_can_deny_a_call_without_stopping_the_round() -> None:
    """deny becomes a failed call the model can react to; the rest of the batch still runs."""
    llm = Llm([*call("c1", "rm"), *call("c2", "read"), done()], [done("recovered")])
    tools = Tools()

    async def gate(c: Any) -> dict[str, Any] | None:
        return {"type": "error", "message": "not allowed"} if c.name == "rm" else None

    events = await run(llm, tools, before_tool_call=gate)

    assert tools.ran == ["read"]  # rm never executed
    denied = next(e for e in events if e["type"] == "tool_result" and e["id"] == "c1")
    assert denied["is_error"] is True
    assert events[-1]["content"] == "recovered"


async def test_the_gate_suspends_instead_of_blocking_on_an_answer() -> None:
    """One approval path: a policy `ask` checkpoints the turn exactly like a handraise does,
    so it costs nothing while waiting and is not capped by a transport timeout."""
    llm = Llm([*call("c1", "deploy"), *call("c2", "read"), done()])
    tools = Tools()
    captured: list[Any] = []

    async def gate(c: Any) -> dict[str, Any] | None:
        if c.name != "deploy":
            return None
        return {"type": "suspend", "pending_id": "approval-1", "source": "policy_gate"}

    async def on_suspend(
        c: Any, result: dict[str, Any], msgs: list[LLMMessage], blocks: list[Any]
    ) -> None:
        captured.append(result)

    events = await run(llm, tools, before_tool_call=gate, on_suspend=on_suspend)

    assert tools.ran == []  # nothing ran, not even the ungated read
    assert events[-1]["type"] == "suspended"
    assert captured[0]["source"] == "policy_gate"


async def test_a_suspending_tool_declares_exclusive_and_runs_alone() -> None:
    """tool.ts:38 — the documented contract: exclusive runs apart, and on suspend the rest
    of the batch must not start."""
    llm = Llm([*call("c1", "read"), *call("c2", "ask"), done()])
    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "p1"}},
        defs={"ask": {"is_exclusive": True}},
    )

    events = await run(llm, tools)

    assert tools.ran == ["ask"]  # `read` never started
    assert events[-1]["type"] == "suspended"


async def test_suspend_snapshot_drops_calls_that_never_ran() -> None:
    """loop-helpers.ts:96 — an unanswerable tool_call left in history makes the provider 400."""
    llm = Llm([*call("c1", "read"), *call("c2", "ask"), *call("c3", "write"), done()])
    tools = Tools(results={"ask": {"type": "suspend", "pending_id": "p1"}})
    snapshots: list[list[LLMMessage]] = []

    async def on_suspend(
        call: Any, result: dict[str, Any], msgs: list[LLMMessage], blocks: list[Any]
    ) -> None:
        snapshots.append(msgs)

    await run(llm, tools, on_suspend=on_suspend)

    assistant = next(m for m in snapshots[0] if m["role"] == "assistant")
    blocks = assistant["content"]
    assert isinstance(blocks, list)
    kept = [b["id"] for b in blocks if b["type"] == "tool_call"]
    assert kept == ["c1", "c2"]  # c3 was dispatched but never ran


async def test_a_batch_executor_gets_the_whole_round() -> None:
    """loop-helpers.ts:57 — results come back in call order, whatever order they finish in."""

    class BatchTools(Tools):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list[list[dict[str, Any]]] = []

        async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
            self.batches.append(calls)
            return [
                {"call_id": c["call_id"], "result": {"type": "text", "text": c["name"]},
                 "is_error": False}
                for c in reversed(calls)
            ]

    llm = Llm([*call("c1", "read"), *call("c2", "grep"), done()], [done("x")])
    tools = BatchTools()

    events = await run(llm, tools)

    assert len(tools.batches) == 1
    assert tools.ran == []  # execute() was bypassed entirely
    results = [e["id"] for e in events if e["type"] == "tool_result"]
    assert results == ["c1", "c2"]


async def test_the_caller_sets_the_iteration_cap() -> None:
    """No built-in limit — `should_stop_after_turn` is where a cap lives."""
    llm = Llm(*[[*call(f"c{i}", "read"), done()] for i in range(5)])
    await run(llm, should_stop_after_turn=cap(2))
    assert len(llm.seen) == 2
