"""Loop semantics pinned against react.ts. Fakes, not mocks."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from nexora import react_loop


def a_call(cid: str, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": cid, "name": name, "args": args or {}, "type": "tool_call"}


def says(text: str = "", *calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


class Llm(GenericFakeChatModel):
    """Replays scripted turns and streams them, the way a provider would.

    `GenericFakeChatModel` alone cannot bind tools and stream at once — asking it to stream a
    turn carrying tool calls fails with "No generations found in stream" — so `_stream` is
    implemented here. Text is split into words so callers see real deltas.
    """

    seen: list[list[BaseMessage]] = []  # noqa: RUF012 — pydantic model, set per instance

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
        self.seen.append(list(messages))
        reply = next(self.messages)
        assert isinstance(reply, AIMessage), "script this fake with AIMessages"
        if reply.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls)
            )
            return
        for i, word in enumerate(str(reply.content).split(" ")):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=word if i == 0 else " " + word)
            )


def scripted(*turns: AIMessage) -> Llm:
    model = Llm(messages=iter(turns))
    model.seen = []
    return model


class Tools:
    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        defs: dict[str, dict[str, Any]] | None = None,
        names: list[str] | None = None,
    ) -> None:
        self.results = results or {}
        self.defs = defs or {}
        self.names = names or ["read"]
        self.ran: list[str] = []

    async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
        self.ran.append(name)
        return self.results.get(name, {"type": "text", "text": "ok"})

    def get(self, name: str) -> dict[str, Any] | None:
        return self.defs.get(name)

    def list(self) -> list[dict[str, Any]]:
        return [{"name": n, "description": n, "parameters": {}} for n in self.names]


def cap(turns: int) -> Any:
    """Stop after N rounds — the loop has no built-in limit, callers set one."""

    async def hook(turn: int, content: str, calls: list[Any]) -> bool:
        return turn + 1 >= turns

    return hook


async def run(llm: Llm, tools: Tools | None = None, **kw: Any) -> list[dict[str, Any]]:
    kw.setdefault("should_stop_after_turn", cap(10))
    return [e async for e in react_loop(llm, tools or Tools(), "hi", **kw)]


def shape_of(events: list[dict[str, Any]]) -> list[str]:
    """Event types with runs of the same type collapsed.

    How many `text` deltas a provider splits an answer into is the provider's business, not a
    semantic worth pinning.
    """
    shape: list[str] = []
    for event in events:
        if not shape or shape[-1] != event["type"]:
            shape.append(event["type"])
    return shape


def text_of(events: list[dict[str, Any]]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "text")


# ── Rounds ───────────────────────────────────────────────────────────────────


async def test_no_tool_calls_ends_the_turn() -> None:
    events = await run(scripted(says("all done")))

    assert shape_of(events) == ["text", "done"]
    assert text_of(events) == "all done"
    assert events[-1]["content"] == "all done"
    assert events[-1]["stop_reason"] == "completed"


async def test_tool_round_feeds_back_into_the_model() -> None:
    llm = scripted(says("", a_call("c1", "read")), says("finished"))
    tools = Tools()

    events = await run(llm, tools)

    assert tools.ran == ["read"]
    assert shape_of(events) == ["tool_call", "tool_result", "text", "done"]
    assert len(llm.seen) == 2


async def test_tool_arguments_survive_the_round_trip() -> None:
    """The model's arguments must reach the tool and the event unchanged."""
    llm = scripted(says("", a_call("c1", "read", {"path": "a.py", "lines": 20})), says("done"))
    seen: list[Any] = []

    class Recording(Tools):
        async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
            seen.append(args)
            return {"type": "text", "text": "ok"}

    events = await run(llm, Recording())

    assert seen == [{"path": "a.py", "lines": 20}]
    assert next(e for e in events if e["type"] == "tool_call")["input"] == seen[0]


# ── Stop conditions ──────────────────────────────────────────────────────────


async def test_exclusive_tool_runs_alone() -> None:
    """loop-helpers.ts:27 — the rest are re-issued next round."""
    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "danger")), says("x"))
    tools = Tools(defs={"danger": {"is_exclusive": True}}, names=["read", "danger"])

    await run(llm, tools)

    assert tools.ran == ["danger"]


async def test_terminating_tool_ends_the_run() -> None:
    """react.ts:263 — `any` over the batch, not `every`."""
    llm = scripted(says("bye", a_call("c1", "submit")))
    tools = Tools(defs={"submit": {"terminates_loop": True}}, names=["submit"])

    events = await run(llm, tools)

    assert events[-1]["stop_reason"] == "tool"
    assert len(llm.seen) == 1


async def test_failed_terminating_tool_gets_a_recovery_round() -> None:
    """react.ts:264 — `!isError`. An errored submit must not end the run."""
    llm = scripted(says("", a_call("c1", "submit")), says("recovered"))
    tools = Tools(
        results={"submit": {"type": "error", "message": "nope"}},
        defs={"submit": {"terminates_loop": True}},
        names=["submit"],
    )

    events = await run(llm, tools)

    assert events[-1]["content"] == "recovered"
    assert len(llm.seen) == 2


async def test_policy_hook_runs_even_when_a_tool_already_stopped_the_run() -> None:
    """react.ts:266 — the hook does budget accounting, so no early return."""
    llm = scripted(says("", a_call("c1", "submit")))
    tools = Tools(defs={"submit": {"terminates_loop": True}}, names=["submit"])
    asked: list[int] = []

    async def policy(turn: int, content: str, calls: list[Any]) -> bool:
        asked.append(turn)
        return False

    await run(llm, tools, should_stop_after_turn=policy)

    assert asked == [0]


async def test_the_caller_sets_the_iteration_cap() -> None:
    """No built-in limit — `should_stop_after_turn` is where a cap lives."""
    llm = scripted(*[says("", a_call(f"c{i}", "read")) for i in range(5)])

    await run(llm, should_stop_after_turn=cap(2))

    assert len(llm.seen) == 2


async def test_steer_arriving_at_the_end_resumes_instead_of_completing() -> None:
    """react.ts:152 — a late steer cancels the stop and buys another model call."""
    llm = scripted(says("almost"), says("really done"))
    drains = 0

    def drain() -> list[BaseMessage]:
        nonlocal drains
        drains += 1
        return [HumanMessage("wait")] if drains == 2 else []

    events = await run(llm, drain_steers=drain)

    assert events[-1]["content"] == "really done"
    assert len(llm.seen) == 2


async def test_abort_leaves_a_record_instead_of_a_stream_that_just_stops() -> None:
    llm = scripted(says("never"))

    events = await run(llm, aborted=lambda: True)

    assert shape_of(events) == ["done"]
    assert events[-1]["stop_reason"] == "aborted"
    assert llm.seen == []


async def test_a_provider_failure_is_reported_not_raised() -> None:
    """react.ts:131 — the run ends with an `error` event, no exception escapes."""

    class Broken(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("429 rate limited")
            yield  # pragma: no cover — makes this a generator

    events = await run(Broken(messages=iter([])))

    assert events == [{"type": "error", "message": "429 rate limited"}]


# ── The gate ─────────────────────────────────────────────────────────────────


async def test_the_gate_can_deny_a_call_without_stopping_the_round() -> None:
    llm = scripted(says("", a_call("c1", "rm"), a_call("c2", "read")), says("recovered"))
    tools = Tools(names=["rm", "read"])

    async def gate(call: dict[str, Any]) -> dict[str, Any] | None:
        return {"type": "error", "message": "not allowed"} if call["name"] == "rm" else None

    events = await run(llm, tools, before_tool_call=gate)

    assert tools.ran == ["read"]
    denied = next(e for e in events if e["type"] == "tool_result" and e["id"] == "c1")
    assert denied["is_error"] is True
    assert events[-1]["content"] == "recovered"


async def test_the_gate_suspends_instead_of_blocking_on_an_answer() -> None:
    """One approval path: a policy `ask` checkpoints the turn exactly like a handraise."""
    llm = scripted(says("", a_call("c1", "deploy"), a_call("c2", "read")))
    tools = Tools(names=["deploy", "read"])
    captured: list[Any] = []

    async def gate(call: dict[str, Any]) -> dict[str, Any] | None:
        if call["name"] != "deploy":
            return None
        return {"type": "suspend", "pending_id": "approval-1", "source": "policy_gate"}

    async def on_suspend(
        call: Any, result: dict[str, Any], msgs: list[BaseMessage], done: list[Any]
    ) -> None:
        captured.append(result)

    events = await run(llm, tools, before_tool_call=gate, on_suspend=on_suspend)

    assert tools.ran == []
    assert events[-1]["type"] == "suspended"
    assert captured[0]["source"] == "policy_gate"


# ── Suspension ───────────────────────────────────────────────────────────────


async def test_a_suspending_tool_declares_exclusive_and_runs_alone() -> None:
    """tool.ts:38 — exclusive runs apart, and on suspend the rest must not start."""
    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "ask")))
    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "p1"}},
        defs={"ask": {"is_exclusive": True}},
        names=["read", "ask"],
    )

    events = await run(llm, tools)

    assert tools.ran == ["ask"]
    assert events[-1]["type"] == "suspended"


async def test_suspend_keeps_completed_results_and_hands_over_the_whole_result() -> None:
    """react.ts:232 — completed results survive; the caller names the record's fields."""
    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "ask")))
    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "p1", "handle": "job-9"}},
        names=["read", "ask"],
    )
    captured: list[Any] = []

    async def on_suspend(
        call: Any, result: dict[str, Any], msgs: list[BaseMessage], done: list[Any]
    ) -> None:
        captured.append((call, result, done))

    events = await run(llm, tools, on_suspend=on_suspend)

    assert tools.ran == ["read", "ask"]
    assert events[-1]["type"] == "suspended"
    call, result, done = captured[0]
    assert (call["id"], call["name"]) == ("c2", "ask")
    assert result == {"type": "suspend", "pending_id": "p1", "handle": "job-9"}
    assert [d["id"] for d in done] == ["c1"]


async def test_suspend_snapshot_drops_calls_that_never_ran() -> None:
    """loop-helpers.ts:96 — an unanswerable tool call left in history makes the provider 400."""
    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "ask"), a_call("c3", "write")))
    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "p1"}},
        names=["read", "ask", "write"],
    )
    snapshots: list[list[BaseMessage]] = []

    async def on_suspend(
        call: Any, result: dict[str, Any], msgs: list[BaseMessage], done: list[Any]
    ) -> None:
        snapshots.append(msgs)

    await run(llm, tools, on_suspend=on_suspend)

    assistant = next(m for m in snapshots[0] if isinstance(m, AIMessage) and m.tool_calls)
    assert [c["id"] for c in assistant.tool_calls] == ["c1", "c2"]  # c3 never ran


# ── Batch execution ──────────────────────────────────────────────────────────


async def test_a_batch_executor_gets_the_whole_round() -> None:
    """loop-helpers.ts:57 — results come back in call order, however they finish."""

    class BatchTools(Tools):
        def __init__(self) -> None:
            super().__init__(names=["read", "grep"])
            self.batches: list[list[dict[str, Any]]] = []

        async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
            self.batches.append(calls)
            return [
                {
                    "call_id": c["call_id"],
                    "result": {"type": "text", "text": c["name"]},
                    "is_error": False,
                }
                for c in reversed(calls)
            ]

    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "grep")), says("x"))
    tools = BatchTools()

    events = await run(llm, tools)

    assert len(tools.batches) == 1
    assert tools.ran == []  # execute() was bypassed entirely
    assert [e["id"] for e in events if e["type"] == "tool_result"] == ["c1", "c2"]


# ── Usage ────────────────────────────────────────────────────────────────────


class UsageLlm(Llm):
    """Streams one chunk carrying whatever usage the scripted turn declared."""

    def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
        self.seen.append(list(messages))
        reply = next(self.messages)
        assert isinstance(reply, AIMessage)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=reply.content,
                tool_calls=reply.tool_calls,
                usage_metadata=reply.usage_metadata,
            )
        )


def _spent(text: str, prompt: int, completion: int, *calls: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=list(calls),
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
    )


async def test_usage_is_summed_across_rounds() -> None:
    """A budget policy needs the whole run's cost, not the last round's."""
    llm = UsageLlm(
        messages=iter([_spent("", 10, 2, a_call("c1", "read")), _spent("fin", 30, 5)])
    )
    llm.seen = []

    events = await run(llm)

    assert events[-1]["usage"] == {"prompt_tokens": 40, "completion_tokens": 7}


async def test_usage_is_absent_rather_than_zero_when_nothing_reported_it() -> None:
    """Nothing spent and nothing measured are different facts."""
    events = await run(scripted(says("hi")))

    assert "usage" not in events[-1]
