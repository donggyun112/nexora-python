"""Subagents semantics pinned against `builtin/delegate.ts`. Fakes, not mocks."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from nexora import (
    AgentDefinition,
    AgentRuntime,
    Answering,
    Authority,
    BackgroundResult,
    FactoryAgent,
    HttpAgent,
    RunnerAgent,
    Subagents,
    react_loop,
)
from nexora.subagents import Deliver, Reply
from nexora_store import MemorySteps

from tests.test_loop import Tools, a_call, says, scripted


def test_every_subagent_variant_satisfies_the_agent_definition_contract() -> None:
    definitions = [
        FactoryAgent("writer", "writes", "write"),
        RunnerAgent("reviewer", "reviews", answers("done")),
        HttpAgent("researcher", "researches", "https://example.test"),
    ]

    assert all(isinstance(definition, AgentDefinition) for definition in definitions)


def child(*events: dict[str, Any], delay: float = 0.0) -> Any:
    """A subagent's event stream, without a model behind it. Never uses its reply tool."""

    async def run(_prompt: str, _reply: Reply, _run_id: str) -> AsyncIterator[dict[str, Any]]:
        if delay:
            await asyncio.sleep(delay)
        for event in events:
            yield event

    return run


def speaks(answer: str, *, is_error: bool = False, delay: float = 0.0) -> Any:
    """A subagent that answers its parent deliberately, the way a handed-off one must."""

    async def run(_prompt: str, reply: Reply, _run_id: str) -> AsyncIterator[dict[str, Any]]:
        if delay:
            await asyncio.sleep(delay)
        await reply(answer, is_error)
        yield {"type": "done", "content": "(kept talking after answering)"}

    return run


def answers(text: str, *, delay: float = 0.0) -> Any:
    return child({"type": "done", "content": text}, delay=delay)


def collector() -> tuple[list[BackgroundResult], Deliver]:
    """A delivery sink that keeps what it was handed."""
    delivered: list[BackgroundResult] = []

    async def deliver(result: BackgroundResult) -> None:
        delivered.append(result)

    return delivered, deliver


def sole(agents: Any, **kw: Any) -> Subagents:
    return Subagents(Tools(), agents, **kw)


async def call(tools: Subagents, args: dict[str, Any]) -> dict[str, Any]:
    return await tools.execute("delegate", "c1", args)


# ── The hop itself ──────────────────────────────────────────────────────────


async def test_a_sync_hop_answers_with_the_childs_own_answer() -> None:
    """`runRuntime` returns the child's `done` content, not a wrapper around it."""
    tools = sole([RunnerAgent("reviewer", "reviews code", answers("looks fine"))])

    assert await call(tools, {"agent": "reviewer", "input": "check it"}) == {
        "type": "text",
        "text": "looks fine",
    }


async def test_a_child_that_errors_comes_back_as_an_error_result() -> None:
    """A failed child is a failed tool call, so the parent model can react to it."""
    failing = child({"type": "error", "message": "no such file"})
    tools = sole([RunnerAgent("reader", "reads", failing)])

    assert await call(tools, {"agent": "reader", "input": "x"}) == {
        "type": "error",
        "message": "no such file",
    }


async def test_the_input_reaches_the_child_as_the_prompt() -> None:
    """A non-string payload is JSON, not `str(dict)` — the child is a model, not a repl."""
    seen: list[str] = []

    async def run(prompt: str, _reply: Reply, _run_id: str) -> AsyncIterator[dict[str, Any]]:
        seen.append(prompt)
        yield {"type": "done", "content": "ok"}

    tools = sole([RunnerAgent("worker", "works", run)])
    await call(tools, {"agent": "worker", "input": {"file": "a.py"}})

    assert seen == [json.dumps({"file": "a.py"})]


async def test_an_unknown_agent_names_the_ones_that_exist() -> None:
    """An error the model can act on: the roster, not just a refusal."""
    tools = sole([RunnerAgent("reviewer", "reviews", answers("hi"))])

    result = await call(tools, {"agent": "ghost", "input": "x"})

    assert result["type"] == "error" and "reviewer" in result["message"]


async def test_delegation_deeper_than_the_cap_is_refused_before_the_child_runs() -> None:
    """delegate.ts's `maxDepth` guard — what stops A→B→A from running forever."""
    ran: list[str] = []

    async def run(_prompt: str, _reply: Reply, _run_id: str) -> AsyncIterator[dict[str, Any]]:
        ran.append("child")
        yield {"type": "done", "content": "never"}

    tools = sole([RunnerAgent("peer", "peers", run)], depth=5, max_depth=5)

    result = await call(tools, {"agent": "peer", "input": "x"})

    assert result["type"] == "error" and "depth 6" in result["message"]
    assert ran == []


# ── Batch fan-out ───────────────────────────────────────────────────────────


async def test_a_batch_runs_its_children_together_and_answers_once() -> None:
    """Deterministic parallelism: one call fans out, instead of hoping for three tool calls."""
    tools = sole(
        [
            RunnerAgent("a", "first", answers("A", delay=0.03)),
            RunnerAgent("b", "second", answers("B", delay=0.03)),
        ]
    )

    started = asyncio.get_running_loop().time()
    result = await call(
        tools,
        {"tasks": [{"agent": "a", "input": "1"}, {"agent": "b", "input": "2"}]},
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert result["text"] == "## delegate[0] a\nA\n\n## delegate[1] b\nB"
    assert elapsed < 0.06, "the two children ran one after the other"


async def test_one_failing_child_does_not_take_its_siblings_answers_down() -> None:
    """The reason the fan-out gathers with `return_exceptions`, same rule as a tool round."""

    async def explode(_prompt: str, _reply: Reply, _run_id: str) -> AsyncIterator[dict[str, Any]]:
        raise RuntimeError("boom")
        yield  # pragma: no cover - unreachable, marks this an async generator

    tools = sole(
        [RunnerAgent("ok", "fine", answers("kept")), RunnerAgent("bad", "breaks", explode)]
    )

    result = await call(
        tools, {"tasks": [{"agent": "ok", "input": "1"}, {"agent": "bad", "input": "2"}]}
    )

    assert "## delegate[0] ok\nkept" in result["text"]
    assert "RuntimeError: boom" in result["text"]


# ── Launching, and the leash ────────────────────────────────────────────────


async def test_a_handoff_answers_before_the_child_finishes() -> None:
    """The point of the mode: the parent's round ends while the child is still working."""
    _, deliver = collector()
    tools = sole(
        [RunnerAgent("slow", "takes a while", answers("late", delay=0.05))], deliver=deliver
    )

    result = await call(tools, {"agent": "slow", "input": "x", "wait": "handoff"})

    assert result["type"] == "text" and "launched as background task" in result["text"]
    assert [task["status"] for task in tools.tasks.list()] == ["running"]


async def settle(tools: Subagents) -> None:
    """Let every launched child finish, without guessing at a sleep."""
    for listed in tools.tasks.list():
        entry = tools.tasks.get(listed["task_id"])
        assert entry is not None, "the registry lost a task it had just listed"
        await asyncio.gather(entry.task, return_exceptions=True)


async def test_a_launched_childs_answer_reaches_the_sink_when_it_settles() -> None:
    """Without this the background mode is a way to run a child and throw its work away."""
    delivered, deliver = collector()
    tools = sole([RunnerAgent("slow", "takes a while", answers("done at last"))], deliver=deliver)

    await call(tools, {"agent": "slow", "input": "x", "wait": "handoff"})
    await settle(tools)

    assert [(r.kind, r.label, r.content, r.is_error) for r in delivered] == [
        ("subagent", "slow", "done at last", False)
    ]


async def test_wait_false_is_fire_and_forget_rather_than_a_silent_sync_hop() -> None:
    """`resolveWaitMode` — models emit `false`, and a strict check drops it into the sync path."""
    delivered, deliver = collector()
    tools = sole([RunnerAgent("worker", "works", answers("unread"))], deliver=deliver)

    result = await call(tools, {"agent": "worker", "input": "x", "wait": False})
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "opened as independent run" in result["text"]
    assert delivered == []


async def test_an_independent_agent_is_answered_with_its_run_id() -> None:
    """Opening an agent without handing back its address opens one nobody can reach again."""
    _, deliver = collector()
    tools = Subagents(
        Tools(),
        [RunnerAgent("worker", "works", answers("mine to keep"))],
        deliver=deliver,
        run_id="parent-7",
    )

    opening = {"agent": "worker", "input": "x", "wait": "none"}
    result = await tools.execute("delegate", "c9", opening)

    assert '"parent-7:c9"' in result["text"]


async def test_an_independent_agent_is_not_on_the_parents_leash() -> None:
    """A thing `cancel_task` can kill is not independent — so it is not in that registry."""
    _, deliver = collector()
    tools = sole([RunnerAgent("worker", "works", answers("mine", delay=0.05))], deliver=deliver)

    await call(tools, {"agent": "worker", "input": "x", "wait": "none"})

    assert tools.tasks.list() == []


async def test_a_handoff_without_a_sink_says_so_instead_of_dropping_the_answer() -> None:
    """A handoff whose answer has nowhere to land is a refusal, not a silent success."""
    tools = sole([RunnerAgent("worker", "works", answers("x"))])

    result = await call(tools, {"agent": "worker", "input": "x", "wait": "handoff"})

    assert result["type"] == "error" and "delivery sink" in result["message"]


async def test_a_cancelled_task_neither_delivers_nor_reports_done() -> None:
    """`cancel_task` is the parent's leash; a child that outlives it was never cut off."""
    delivered, deliver = collector()
    tools = sole([RunnerAgent("slow", "slow", answers("too late", delay=1.0))], deliver=deliver)

    await call(tools, {"agent": "slow", "input": "x", "wait": "handoff"})
    task_id = tools.tasks.list()[0]["task_id"]

    assert await tools.execute("cancel_task", "c2", {"task_id": task_id}) == {
        "type": "text",
        "text": f"Cancelled task {task_id}.",
    }
    await asyncio.sleep(0.01)

    assert tools.tasks.status(task_id) == "cancelled"
    assert delivered == []


async def test_check_tasks_reports_the_task_a_launch_created() -> None:
    """The id it prints is the one `cancel_task` takes; a mismatch makes the leash unusable."""
    _, deliver = collector()
    tools = sole([RunnerAgent("worker", "works", answers("ok"))], deliver=deliver)

    await call(tools, {"agent": "worker", "input": "x", "wait": "handoff"})
    listed = json.loads((await tools.execute("check_tasks", "c2", {}))["text"])

    assert [(item["kind"], item["label"]) for item in listed] == [("subagent", "worker")]
    assert tools.tasks.get(listed[0]["task_id"]) is not None


async def test_watch_notifies_once_the_tasks_it_names_have_settled() -> None:
    """Non-blocking by construction: the notice is a later input, never a held-open round."""
    delivered, deliver = collector()
    tools = sole([RunnerAgent("worker", "works", answers("fin", delay=0.02))], deliver=deliver)

    await call(tools, {"agent": "worker", "input": "x", "wait": "handoff"})
    task_id = tools.tasks.list()[0]["task_id"]
    armed = await tools.execute("watch_task", "c2", {"task_ids": [task_id]})
    await asyncio.sleep(0.05)

    assert "you will be notified" in armed["text"]
    assert [r.kind for r in delivered] == ["subagent", "watch"]
    assert f"{task_id}=done" in delivered[1].content


# ── Composition with the host's tools ───────────────────────────────────────


async def test_the_hosts_tools_still_run_through_the_wrapper() -> None:
    """`Subagents` wraps, it does not replace — a host tool must survive the composition."""
    tools = Subagents(Tools(results={"read": {"type": "text", "text": "file body"}}), [])

    assert await tools.execute("read", "c1", {}) == {"type": "text", "text": "file body"}


async def test_the_delegation_tools_are_offered_beside_the_hosts() -> None:
    """A tool the model is never shown is a tool it never calls."""
    tools = Subagents(Tools(names=["read"]), [RunnerAgent("r", "reviews", answers("x"))])

    offered = [item["name"] for item in tools.list()]

    assert offered == [
        "delegate",
        "check_tasks",
        "cancel_task",
        "read_task_output",
        "watch_task",
        "read",
    ]


async def test_a_factory_agent_is_built_at_call_time() -> None:
    """The spec carries no runtime, so nothing runs unless the host's factory builds one."""
    built: list[str] = []

    def factory(spec: FactoryAgent, _authority: Authority) -> Any:
        built.append(spec.system_prompt)
        return answers("from a spec")

    tools = sole([FactoryAgent("writer", "writes", "you write")], factory=factory)

    assert await call(tools, {"agent": "writer", "input": "x"}) == {
        "type": "text",
        "text": "from a spec",
    }
    assert built == ["you write"]


# ── Authority, and the no-escalation invariant ──────────────────────────────


async def test_a_childs_authority_is_a_subset_of_its_parents() -> None:
    """The invariant the whole thing rests on: asking for more than the parent held grants less.

    `handraise/authority.ts` — attenuate is an intersection, so a chain only ever narrows.
    """
    granted: list[Authority] = []

    def factory(_spec: FactoryAgent, authority: Authority) -> Any:
        granted.append(authority)
        return answers("done")

    tools = sole(
        [FactoryAgent("writer", "writes", "you write", tools=("read", "write", "deploy"))],
        factory=factory,
        authority=["read", "write"],
        blocked_tools_for_child=(),
    )
    await call(tools, {"agent": "writer", "input": "x"})

    assert granted[0].ceiling == ("read", "write")  # "deploy" was never the parent's to give


async def test_an_unrestricted_parent_grants_exactly_what_was_asked_for() -> None:
    """A root scoping itself down for the first time. `None` is the top, not a wildcard grant."""
    granted: list[Authority] = []

    def factory(_spec: FactoryAgent, authority: Authority) -> Any:
        granted.append(authority)
        return answers("done")

    tools = sole(
        [FactoryAgent("reader", "reads", "you read", tools=("read",))],
        factory=factory,
        blocked_tools_for_child=(),
    )
    await call(tools, {"agent": "reader", "input": "x"})

    assert granted[0].ceiling == ("read",)


async def test_a_blocked_tool_is_refused_even_when_the_parent_held_it() -> None:
    """`blocked_tools_for_child` applies after the intersection, so it cannot be asked around."""
    granted: list[Authority] = []

    def factory(_spec: FactoryAgent, authority: Authority) -> Any:
        granted.append(authority)
        return answers("done")

    tools = sole(
        [FactoryAgent("peer", "peers", "you peer", tools=("read", "delegate"))],
        factory=factory,
        authority=["read", "delegate"],
    )
    await call(tools, {"agent": "peer", "input": "x"})

    assert granted[0].ceiling == ("read",)


async def test_authority_narrows_over_a_chain_and_never_widens() -> None:
    """Composed transitively: whatever a grandchild ends up with, the root could reach it too."""
    root = Authority(ceiling=("read", "write", "deploy"), depth=0)

    child = root.attenuate(["read", "write"])
    grandchild = child.attenuate(["write", "deploy"])

    assert child.ceiling == ("read", "write")
    assert grandchild.ceiling == ("write",)  # deploy is gone for good, not re-granted
    assert (child.depth, grandchild.depth) == (1, 2)


# ── End to end, through the durable queue ───────────────────────────────────


async def test_a_background_answer_re_enters_the_run_as_model_context() -> None:
    """The whole point of `background_sink`: the child's answer becomes a later turn's input.

    The child settles after the launching turn has ended, which is the case the reference needs
    its second delivery path (`deliverResult`) for. Here the durable queue holds it either way.
    """
    runtime = AgentRuntime(store=MemorySteps())
    tools = Subagents(
        Tools(),
        [RunnerAgent("researcher", "digs", answers("40 papers", delay=0.05))],
        deliver=runtime.background_sink("delegated"),
    )
    launch = a_call("c1", "delegate", {"agent": "researcher", "input": "dig", "wait": "handoff"})

    await runtime.run("delegated", scripted(says("", launch), says("launched")), tools, "go")
    await settle(tools)

    later = scripted(says("read it"))
    await runtime.run("delegated", later, tools)

    assert [str(message.content) for message in later.seen[-1]][-1].endswith("40 papers")


async def test_a_background_answer_arriving_mid_run_does_not_fence_it() -> None:
    """A child settling while its parent still works must not take the parent's lease away.

    `submit` opens the same run to enqueue and released the lease on the way out, so the next
    arrival found no holder and moved the token past the one the live attempt was still writing
    with. The parent died on its next durable write — with the child's answer already queued, so a
    retry replayed it into a run that had crashed for reasons the transcript could not show.
    """
    runtime = AgentRuntime(store=MemorySteps())
    deliver = runtime.background_sink("live")

    class Delivering(Tools):
        """A tool that settles two background children while the round is still open."""

        async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
            for index in (1, 2):
                await deliver(BackgroundResult(f"task-{index}", "agent", "digger", "found it"))
            return {"type": "text", "text": "ok"}

    llm = scripted(says("", a_call("c1", "read")), says("", a_call("c2", "read")), says("fin"))

    outcome = await runtime.run("live", llm, Delivering(), "go")

    assert outcome["stop_reason"] == "completed"
    arrived = [str(message.content) for message in llm.seen[-1]]
    assert sum("found it" in content for content in arrived) == 2, "and both answers landed"


# ── HTTP children ───────────────────────────────────────────────────────────


async def test_a_remote_child_is_reached_over_http_and_its_body_is_the_answer() -> None:
    """The third subagent kind. A real socket, because the point is that it leaves the process."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # http.server's spelling, not ours
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"remote says hi")

        def log_message(self, *_: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        tools = sole([HttpAgent("far", "lives elsewhere", url)])

        result = await call(tools, {"agent": "far", "input": "ping"})
    finally:
        server.shutdown()

    assert result == {"type": "text", "text": "remote says hi"}
    assert json.loads(received[0]) == {"input": "ping"}


# ── Answering: the child answers on purpose ───────────────────────────────────


async def test_a_handed_off_child_answers_through_its_reply_tool() -> None:
    """The mode's contract: the answer travels because the child sent it, not because we read it."""
    delivered, deliver = collector()
    tools = sole([RunnerAgent("scout", "scouts", speaks("found three leaks"))], deliver=deliver)

    await call(tools, {"agent": "scout", "input": "look", "wait": "handoff"})
    await settle(tools)

    assert [(r.label, r.content, r.is_error) for r in delivered] == [
        ("scout", "found three leaks", False)
    ]


async def test_a_reply_reaches_the_parent_before_the_child_stops_running() -> None:
    """A handoff that only delivered at process end would be a slower `sync`, not a handoff."""
    delivered, deliver = collector()

    running = asyncio.Event()

    async def lingering(_prompt: str, reply: Reply, _run_id: str) -> AsyncIterator[dict[str, Any]]:
        await reply("early answer", False)
        await running.wait()
        yield {"type": "done", "content": "finally done"}

    tools = sole([RunnerAgent("slow", "lingers", lingering)], deliver=deliver)
    await call(tools, {"agent": "slow", "input": "x", "wait": "handoff"})
    await asyncio.sleep(0.01)

    assert [r.content for r in delivered] == ["early answer"], "the parent waited for the child"
    running.set()
    await settle(tools)


async def test_a_handed_off_child_that_never_replies_still_answers_from_its_last_turn() -> None:
    """Handing work to an agent without the reply tool must not lose the work."""
    delivered, deliver = collector()
    tools = sole([RunnerAgent("quiet", "never replies", answers("said nothing on purpose"))],
                 deliver=deliver)

    await call(tools, {"agent": "quiet", "input": "x", "wait": "handoff"})
    await settle(tools)

    assert [r.content for r in delivered] == ["said nothing on purpose"]


async def test_a_sync_child_that_replies_is_answered_by_the_reply_not_its_last_words() -> None:
    """A child that named its answer is not overruled by whatever it trailed off with."""
    tools = sole([RunnerAgent("scout", "scouts", speaks("the answer"))])

    assert await call(tools, {"agent": "scout", "input": "x"}) == {
        "type": "text",
        "text": "the answer",
    }


async def test_a_child_reporting_a_failure_reaches_the_parent_as_an_error() -> None:
    """`is_error` on the reply is how a child says "I could not", distinct from crashing."""
    delivered, deliver = collector()
    failing = speaks("the repo has no tests to run", is_error=True)
    tools = sole([RunnerAgent("runner", "runs tests", failing)], deliver=deliver)

    await call(tools, {"agent": "runner", "input": "x", "wait": "handoff"})
    await settle(tools)

    assert [(r.content, r.is_error) for r in delivered] == [
        ("the repo has no tests to run", True)
    ]


# ── Answering: the child side ─────────────────────────────────────────────────


async def test_the_reply_tool_is_offered_to_the_child_beside_its_own() -> None:
    """A tool the child is never shown is a way home it never takes."""
    sent: list[tuple[str, bool]] = []

    async def reply(text: str, is_error: bool) -> None:
        sent.append((text, is_error))

    child_tools = Answering(Tools(names=["read"]), reply)

    assert [item["name"] for item in child_tools.list()] == ["respond_to_parent", "read"]
    assert (child_tools.get("respond_to_parent") or {})["terminates_loop"] is True


async def test_the_reply_tool_ends_the_childs_run_when_it_answers() -> None:
    """`terminates_loop` — a child that answered is done, and a second answer would race it."""
    sent: list[tuple[str, bool]] = []

    async def reply(text: str, is_error: bool) -> None:
        sent.append((text, is_error))

    child_tools = Answering(Tools(), reply)
    llm = scripted(says("", a_call("r1", "respond_to_parent", {"result": "done digging"})))

    events = [event async for event in react_loop(llm, child_tools)]

    assert sent == [("done digging", False)]
    assert [e["stop_reason"] for e in events if e["type"] == "done"] == ["tool"]


async def test_an_empty_reply_is_refused_so_the_parent_is_not_answered_with_nothing() -> None:
    """An empty answer ends the child's run and tells the parent nothing — worse than no call."""
    sent: list[tuple[str, bool]] = []

    async def reply(text: str, is_error: bool) -> None:
        sent.append((text, is_error))

    result = await Answering(Tools(), reply).execute("respond_to_parent", "r1", {"result": "  "})

    assert result["type"] == "error"
    assert sent == []


async def test_the_childs_own_tools_still_run_under_the_reply_wrapper() -> None:
    """`Answering` adds a way home; it must not take the child's tools away."""

    async def reply(_text: str, _is_error: bool) -> None:  # pragma: no cover - unused here
        raise AssertionError("not this test")

    child_tools = Answering(Tools(results={"read": {"type": "text", "text": "body"}}), reply)

    assert await child_tools.execute("read", "c1", {}) == {"type": "text", "text": "body"}


# ── The two modes are actually different ────────────────────────────────────


async def test_sync_holds_the_round_and_handoff_releases_it() -> None:
    """The distinction the two modes exist for, measured rather than asserted from the name."""
    _, deliver = collector()
    tools = sole([RunnerAgent("slow", "slow", speaks("eventually", delay=0.05))], deliver=deliver)

    started = asyncio.get_running_loop().time()
    await call(tools, {"agent": "slow", "input": "x", "wait": "handoff"})
    handed_off = asyncio.get_running_loop().time() - started

    started = asyncio.get_running_loop().time()
    await call(tools, {"agent": "slow", "input": "x", "wait": "sync"})
    waited = asyncio.get_running_loop().time() - started

    assert handed_off < waited / 5, "the handoff blocked the round it was supposed to release"
    await settle(tools)


# ── The child's name ────────────────────────────────────────────────────────


async def test_the_child_run_id_is_derived_from_the_call_that_asked_for_it() -> None:
    """Two runs delegating under the same call id are still two different children."""
    seen: list[str] = []

    async def run(_prompt: str, _reply: Reply, run_id: str) -> AsyncIterator[dict[str, Any]]:
        seen.append(run_id)
        yield {"type": "done", "content": "ok"}

    agents = [RunnerAgent("worker", "works", run)]
    await Subagents(Tools(), agents, run_id="run-42").execute(
        "delegate", "c7", {"agent": "worker", "input": "x"}
    )
    await Subagents(Tools(), agents, run_id="run-43").execute(
        "delegate", "c7", {"agent": "worker", "input": "x"}
    )
    await Subagents(Tools(), agents).execute("delegate", "c7", {"agent": "worker", "input": "x"})

    assert seen == ["run-42:c7", "run-43:c7", "c7"]


async def test_a_retried_delegate_resumes_the_child_instead_of_repeating_its_effects() -> None:
    """The point of deriving the id: recovery retries a `delegate` call with the same call id.

    A subagent re-run from nothing does not repeat one write, it repeats everything it did. Given
    a name, the child's effects are in the ledger under it, and the retry becomes the child's own
    recovery — which is what `call_id` being an idempotency key has to mean for this tool.
    """
    steps = MemorySteps()
    executed: list[str] = []

    async def run(_prompt: str, _reply: Reply, run_id: str) -> AsyncIterator[dict[str, Any]]:
        done = await steps.read(run_id, "write")
        if done.status == "done":
            yield {"type": "done", "content": f"resumed, {done.value['wrote']} already written"}
            return
        await steps.start(run_id, "write")
        executed.append(run_id)
        await steps.finish(run_id, "write", {"wrote": "the report"})
        raise RuntimeError("died between the effect and the answer")

    tools = Subagents(Tools(), [RunnerAgent("worker", "writes", run)], run_id="run-42")

    crashed = await tools.execute("delegate", "c7", {"agent": "worker", "input": "write it"})
    retried = await tools.execute("delegate", "c7", {"agent": "worker", "input": "write it"})

    assert executed == ["run-42:c7"], "the retry re-executed the child's committed effect"
    assert crashed["type"] == "error"
    assert retried == {"type": "text", "text": "resumed, the report already written"}
