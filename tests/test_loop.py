"""Loop semantics pinned against react.ts. Fakes, not mocks."""

from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any, cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from nexora import AgentRuntime, react_loop
from nexora.contracts import BatchTools, EventType, PendingInput, ToolCall
from nexora.controls import ControlPlane, Halt, Ingress, Journal, Permissions, gate, writer
from nexora.orchestrator import AgentAborted, AgentFailed, AgentSuspended, PermissionChain
from nexora.tools import InvalidToolResult


def a_call(cid: str, name: str, args: dict[str, Any] | None = None) -> ToolCall:
    return cast(ToolCall, {"id": cid, "name": name, "args": args or {}, "type": "tool_call"})


def says(text: str = "", *calls: ToolCall) -> AIMessage:
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


async def run(
    llm: Llm,
    tools: Tools | None = None,
    *,
    pre_tool_use: Any = None,
    after_tool_call: Any = None,
    durable: bool = True,
    **kw: Any,
) -> list[dict[str, Any]]:
    """Exercise loop semantics through the runtime-owned effect boundary."""
    kw.setdefault("should_stop_after_turn", cap(10))
    if pre_tool_use or after_tool_call:
        stages = [pre_tool_use] if pre_tool_use else []
        kw["controls"] = ControlPlane(
            pre_tool_use=Permissions(*(gate(s) for s in stages)),
            after_tool_call=Journal(writer(after_tool_call)) if after_tool_call else None,
        )
    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    if not durable:
        return [
            event
            async for event in react_loop(
                llm,
                tools or Tools(),
                **kw,
            )
        ]

    with suppress(AgentAborted, AgentFailed, AgentSuspended):
        await AgentRuntime().run(
            "loop-test",
            llm,
            tools or Tools(),
            "hi",
            on_event=collect,
            **kw,
        )
    return events


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


async def test_a_tool_result_reaches_the_caller_unchanged() -> None:
    """A multimodal result must survive the round; flattening it to a string loses the images."""
    multimodal: dict[str, Any] = {
        "type": "content",
        "blocks": [
            {"type": "text", "text": "page 1"},
            {"type": "image", "data": "xxx", "mime_type": "image/png"},
        ],
    }
    llm = scripted(says("", a_call("c1", "render")), says("fin"))

    events = await run(llm, Tools(results={"render": multimodal}, names=["render"]))

    assert next(e for e in events if e["type"] == "tool_result")["result"] == multimodal


async def test_the_system_prompt_reaches_the_provider() -> None:
    llm = scripted(says("ok"))

    async for _ in react_loop(llm, Tools(), system_prompt="너는 도우미다"):
        pass

    assert llm.seen[0][0].content == "너는 도우미다"


# ── Stop conditions ──────────────────────────────────────────────────────────


async def test_exclusive_tool_runs_alone() -> None:
    """loop-helpers.ts:27 — the rest are re-issued next round."""
    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "danger")), says("x"))
    tools = Tools(defs={"danger": {"is_exclusive": True}}, names=["read", "danger"])

    await run(llm, tools)

    assert tools.ran == ["danger"]


async def test_terminating_tool_ends_the_run() -> None:
    """react.ts:263 — `any` over the batch, not `every`.

    Drained (`durable=False`) on purpose: the runtime stops consuming at the terminal event, so a
    loop that kept going would leave the generator parked and `llm.seen` would still read 1.
    """
    llm = scripted(says("bye", a_call("c1", "submit")))
    tools = Tools(defs={"submit": {"terminates_loop": True}}, names=["submit"])

    events = await run(llm, tools, durable=False)

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

    async def drain() -> list[PendingInput]:
        nonlocal drains
        drains += 1
        return [PendingInput("user_steer", HumanMessage("wait"))] if drains == 2 else []

    events = await run(llm, drain_inputs=drain, durable=False)

    assert events[-1]["content"] == "really done"
    assert len(llm.seen) == 2


async def test_an_ingress_screen_masks_an_input_before_the_model_or_the_log_sees_it() -> None:
    """After admission the original exists nowhere: not in context, not in the audit log."""
    llm = scripted(says("done"))
    drains = 0

    async def drain() -> list[PendingInput]:
        nonlocal drains
        drains += 1
        return [PendingInput("user_prompt", HumanMessage("ssn is 123-45"))] if drains == 1 else []

    async def mask(ctx: Any, inputs: list[PendingInput]) -> list[PendingInput]:
        return [
            PendingInput(
                item.kind,
                HumanMessage(str(item.message.content).replace("123-45", "***")),
                item.origin_id,
            )
            for item in inputs
        ]

    logged: list[tuple[Any, dict[str, Any]]] = []

    async def emit(event: Any, payload: dict[str, Any]) -> None:
        logged.append((event, payload))

    await run(
        llm,
        drain_inputs=drain,
        durable=False,
        controls=ControlPlane(on_inputs=Ingress(mask)),
        emit=emit,
    )

    prompt_seen = str(llm.seen[0][-1].content)
    assert "***" in prompt_seen and "123-45" not in prompt_seen
    injected = [payload for event, payload in logged if event == EventType.CONTEXT_INJECTED]
    assert injected and "123-45" not in str(injected)


async def test_a_halting_screen_ends_the_run_before_the_model_is_called() -> None:
    llm = scripted(says("never"))

    async def drain() -> list[PendingInput]:
        return [PendingInput("user_prompt", HumanMessage("raw secrets"))]

    async def block(ctx: Any, inputs: Any) -> Any:
        return Halt("policy")

    events = await run(
        llm,
        drain_inputs=drain,
        durable=False,
        controls=ControlPlane(on_inputs=Ingress(block)),
    )

    assert events[-1]["stop_reason"] == "policy"
    assert llm.seen == []


async def test_abort_leaves_a_record_instead_of_a_stream_that_just_stops() -> None:
    llm = scripted(says("never"))

    events = await run(llm, aborted=lambda: True)

    assert shape_of(events) == ["done"]
    assert events[-1]["stop_reason"] == "aborted"
    assert llm.seen == []


async def test_an_abort_between_a_tool_round_and_the_next_model_call_ends_the_run() -> None:
    """The abort that lands with the round already absorbed, and it ends there.

    `stop_reason` alone cannot see the failure: without the return the loop reaches the next
    round, notices the abort again and reports it again — same reason, same last event, two
    records of one run ending. The count is the assertion.
    """
    tools = Tools()
    llm = scripted(says("", a_call("c1", "read")), says("should not be reached"))

    events = await run(llm, tools, aborted=lambda: bool(tools.ran), durable=False)

    assert tools.ran == ["read"]
    assert [e["stop_reason"] for e in events if e["type"] == "done"] == ["aborted"]
    assert len(llm.seen) == 1


async def test_a_provider_failure_is_reported_not_raised() -> None:
    """react.ts:131 — the run ends with an `error` event, no exception escapes."""

    class Broken(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("429 rate limited")
            yield  # pragma: no cover — makes this a generator

    events = await run(Broken(messages=iter([])), durable=False)

    assert events == [{"type": "error", "message": "429 rate limited"}]  # and nothing after it


async def test_an_abort_cuts_a_generation_instead_of_waiting_it_out() -> None:
    """Checked per chunk. A shutdown arriving early must not have to wait out a long answer.

    Still a poll at a point the loop chose: a callback firing at an arbitrary await would make the
    step sequence depend on timing, and a durable replay could not reproduce it.
    """
    stopping: list[bool] = []

    class TenChunks(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            self.seen.append(list(messages))
            for i in range(10):
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"w{i} "))

    llm = TenChunks(messages=iter([]))
    llm.seen = []
    delivered: list[str] = []

    async for event in react_loop(llm, Tools(), aborted=lambda: bool(stopping)):
        if event["type"] == "text":
            delivered.append(event["text"])
            stopping.append(True)  # the shutdown lands after the first chunk
        if event["type"] == "done":
            assert event["stop_reason"] == "aborted"
            assert event["content"] == "w0 "  # what had streamed, not a whole answer

    assert delivered == ["w0 "]


async def test_repeated_aborts_still_end_the_run_exactly_once() -> None:
    """Shutdown signals arrive in bunches — SIGTERM then SIGINT, or a supervisor retrying.

    The property is that the terminal event is emitted once. There are five places the loop can
    notice an abort and each of them ends the run, so a second notice must find nothing left to
    do. An already-aborted run is the same question asked before the first round.
    """
    fired: list[bool] = []

    class Chunks(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            self.seen.append(list(messages))
            for i in range(20):
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"w{i} "))

    llm = Chunks(messages=iter([]))
    llm.seen = []
    terminal: list[dict[str, Any]] = []
    async for event in react_loop(llm, Tools(), aborted=lambda: bool(fired)):
        if event["type"] == "text":
            fired.extend([True, True, True])  # three shutdowns land at once
        elif event["type"] in ("done", "error", "suspended"):
            terminal.append(event)

    assert [e["stop_reason"] for e in terminal] == ["aborted"]

    # Aborted before it starts: no provider call at all, still one terminal event.
    already = Chunks(messages=iter([]))
    already.seen = []
    terminal = []
    async for event in react_loop(already, Tools(), aborted=lambda: True):
        if event["type"] in ("done", "error", "suspended"):
            terminal.append(event)

    assert [e["stop_reason"] for e in terminal] == ["aborted"]
    assert already.seen == []


async def test_leaving_a_stream_closes_it_so_the_abort_reaches_the_provider() -> None:
    """Abandoning an `async for` does not close the generator — Python waits for a gc pass.

    Invisible in every correctness test and expensive in production: the HTTP connection stays
    open and the model keeps generating tokens nobody reads and someone pays for. Both exits are
    checked: the loop aborting, and the caller walking away from the loop.
    """
    closed: list[str] = []

    class Watched(Llm):
        def _stream(self, messages: Any, *a: Any, **k: Any) -> Any:
            self.seen.append(list(messages))
            try:
                for i in range(20):
                    yield ChatGenerationChunk(message=AIMessageChunk(content=f"w{i} "))
            except GeneratorExit:
                closed.append("closed")
                raise

    # ── the loop aborts out of the stream ──
    stopping: list[bool] = []
    llm = Watched(messages=iter([]))
    llm.seen = []
    async for event in react_loop(llm, Tools(), aborted=lambda: bool(stopping)):
        if event["type"] == "text":
            stopping.append(True)

    assert closed == ["closed"]

    # ── the caller walks away from the loop ──
    closed.clear()
    llm = Watched(messages=iter([]))
    llm.seen = []
    events = cast(AsyncGenerator[dict[str, Any], None], react_loop(llm, Tools()))
    async for event in events:
        if event["type"] == "text":
            break
    await events.aclose()

    assert closed == ["closed"]


# ── Permission is an ordered chain of calls ──────────────────────────────────


async def test_a_deny_beats_an_allow_whatever_the_order() -> None:
    """The invariant a subscription cannot promise. A permissive stage is an opinion, and every
    stage after it is still consulted — which is what stops a hook from being the last word."""
    tools = Tools(names=["rm"])
    seen: list[str] = []

    async def permissive(call: ToolCall) -> Any:
        seen.append("hook")
        return {"type": "allow"}

    async def deny_rules(call: ToolCall) -> Any:
        seen.append("rules")
        return {"type": "error", "message": "not allowed"}

    async def never_reached(call: ToolCall) -> Any:
        seen.append("after-deny")
        return None

    llm = scripted(says("", a_call("c1", "rm")), says("recovered"))
    chain = PermissionChain(permissive, deny_rules, never_reached)
    events = await run(llm, tools, pre_tool_use=chain.resolve)

    assert seen == ["hook", "rules"]  # allow did not stop the chain; deny did
    assert tools.ran == []
    assert next(e for e in events if e["type"] == "tool_result")["is_error"] is True
    assert events[-1]["content"] == "recovered"


async def test_a_record_that_cannot_be_written_stops_the_run() -> None:
    """Fail-closed, and the reason it is a call rather than an event: `EventStream` would log
    this and carry on, and the next resume could not tell which calls already ran."""

    async def broken(call: dict[str, Any], result: dict[str, Any]) -> None:
        raise RuntimeError("disk full")

    llm = scripted(says("", a_call("c1", "read")), says("never"))

    with pytest.raises(RuntimeError, match="disk full"):
        await run(llm, Tools(), after_tool_call=broken)


# ── The gate ─────────────────────────────────────────────────────────────────


async def test_the_gate_can_deny_a_call_without_stopping_the_round() -> None:
    llm = scripted(says("", a_call("c1", "rm"), a_call("c2", "read")), says("recovered"))
    tools = Tools(names=["rm", "read"])

    async def gate(call: dict[str, Any]) -> dict[str, Any] | None:
        return {"type": "error", "message": "not allowed"} if call["name"] == "rm" else None

    events = await run(llm, tools, pre_tool_use=gate)

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

    events = await run(llm, tools, pre_tool_use=gate, on_suspend=on_suspend)

    assert tools.ran == []
    assert events[-1]["type"] == "suspended"
    assert captured[0]["source"] == "policy_gate"


# ── Suspension ───────────────────────────────────────────────────────────────


async def test_a_tool_cannot_suspend_after_crossing_the_effect_boundary() -> None:
    tools = Tools(results={"ask": {"type": "suspend", "pending_id": "p1"}}, names=["ask"])

    with pytest.raises(InvalidToolResult, match="pre_tool_use"):
        async for _ in react_loop(scripted(says("", a_call("c1", "ask"))), tools):
            pass

    assert tools.ran == ["ask"]


# ── Batch execution ──────────────────────────────────────────────────────────


async def test_a_batch_executor_gets_the_whole_round() -> None:
    """loop-helpers.ts:57 — results come back in call order, however they finish."""

    class Batched(Tools):
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
    tools = Batched()

    events = await run(llm, tools, durable=False)

    assert isinstance(tools, BatchTools)  # the capability is declared, not sniffed for
    assert len(tools.batches) == 1
    assert tools.ran == []  # execute() was bypassed entirely
    assert [e["id"] for e in events if e["type"] == "tool_result"] == ["c1", "c2"]


async def test_a_batch_is_journalled_in_call_order() -> None:
    """Call order, not completion order — a retry replays the sequence the journal recorded."""

    class Batched(Tools):
        async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {"call_id": c["call_id"], "result": {"type": "text", "text": c["call_id"]}}
                for c in reversed(calls)  # finishes backwards
            ]

    written: list[str] = []

    async def journal(call: dict[str, Any], result: dict[str, Any]) -> None:
        written.append(call["id"])

    llm = scripted(says("", a_call("c1", "read"), a_call("c2", "read")), says("x"))
    await run(llm, Batched(), after_tool_call=journal, durable=False)

    assert written == ["c1", "c2"]


async def test_a_batch_arrives_gated() -> None:
    """Every gate decides before the batch starts, and refused calls never reach the executor.

    The order matters, not just the outcome: once the executor is running calls concurrently
    there is no way to stop one a later gate would have refused. Gating a call and running it
    before the next call is gated — which is all a per-call hook can do — breaks this.
    """

    class Batched(Tools):
        def __init__(self) -> None:
            super().__init__(names=["read", "write"])
            self.order: list[str] = []

        async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
            self.order.append("batch:" + ",".join(c["call_id"] for c in calls))
            return [
                {"call_id": c["call_id"], "result": {"type": "text", "text": "ok"}} for c in calls
            ]

    tools = Batched()

    async def refuse_the_last(call: dict[str, Any]) -> dict[str, Any] | None:
        tools.order.append(f"gate:{call['id']}")
        return {"type": "error", "message": "no"} if call["id"] == "c3" else None

    llm = scripted(
        says("", a_call("c1", "read"), a_call("c2", "read"), a_call("c3", "write")), says("x")
    )
    events = await run(llm, tools, pre_tool_use=refuse_the_last, durable=False)

    assert tools.order == ["gate:c1", "gate:c2", "gate:c3", "batch:c1,c2"]
    assert [e["is_error"] for e in events if e["type"] == "tool_result"] == [False, False, True]


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


def _spent(text: str, prompt: int, completion: int, *calls: ToolCall) -> AIMessage:
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
