"""Subagents — children a parent agent owns, ported from `builtin/delegate.ts`.

Wraps a host's `Tools` and adds five: `delegate`, and the four that hold the leash on what it
launches. Composed the way the rest of this package composes — `Subagents(tools, ...)` beside
`Stepped(tools, orchestrator)` and `Concurrent(tools)`.

Three kinds of subagent, as in the reference. `Declarative` is a spec built at call time by a
factory the host supplies; `Compiled` is an already-built child the host closes over; `Remote` is
an HTTP endpoint. A child is reached by *name*, which is the reference's `capability` minus the
registry lookup — Nexora for Python has no transport and no agent registry, so there is nothing to
resolve a capability against and nothing to publish a fire-and-forget envelope to. What the
reference does over a topic, this does in-process against `subagents`.

Two ways to hand work over, and they are genuinely different shapes:

* `sync` — the parent stops and waits. Its round does not end until the child answers, so the
  answer is guaranteed to be in hand before the parent decides anything else. Use it when the
  parent cannot continue without it.
* `async` — the parent gives the task away and carries on. The child answers **when it decides
  to**, by calling `respond_to_parent`, and that answer re-enters the parent's run through the
  same durable input queue a human steer uses. This is why a handed-off child must be built with
  `Answering` around its tools: the parent is no longer reading the child's stream, so the reply
  tool is the only way back. A child that never calls it still answers, from its last turn, so
  handing work to an agent that lacks the tool loses nothing.

  Live turn or finished turn, the queue does not care, which is why this needs neither of the
  reference's two delivery paths (`steerSelf` and `deliverResult`). `async` is the reference's own
  name for this mode; `handoff` is accepted as a synonym, since that is what the shape is usually
  called — but note that in the reference "handoff" names `delegate` itself, the addressed hop, as
  opposed to `publish_topic`'s anonymous broadcast.
`none` is not a third way of waiting — it is a different relationship. It opens an **independent
agent**: the caller gets that agent's run id and nothing else, no result and no way to stop it.
The reference is unambiguous about this even though the mode's name hides it — `none` never
reaches a caller-owned child there at all (`delegate.ts:485-499`, where an inline subagent is
executed synchronously and the `none` branch below is for registry peers, published over
transport). The thing on the other side has its own lifecycle, so its outcome was never the
caller's to collect.

Nothing is lost by not collecting it. `AgentRuntime` is keyed by run id, so the id this hands
back is the whole of what anyone needs to reach that agent again: steer it with `submit`, answer
its permission suspensions, drive it further with `run`, or read what it did out of the ledger.
A person can do all of that too. What would be lost is opening an agent and *not* handing back
its address — an agent nobody could reach again.

Delegation depth rides on the tool instance rather than on a message envelope, because a child
here is built in this process by a factory this object holds. A host that spans processes has to
carry `depth` itself; the reference carries it in `metadata.delegationDepth` for exactly that
reason.

A child gets a run id, and it is derived rather than fresh: `f"{parent_run_id}:{call_id}"`. That
is what keeps `delegate` inside the contract every other tool is held to. A tool call's id is its
idempotency key, so recovery may retry an interrupted call — but a subagent re-run from nothing
does not repeat a write, it repeats *everything*, model rounds and child effects together. Given
the same name, the child's own effects are in the ledger under it, and retrying the parent's call
reaches the child's own recovery instead of a second execution. The parent's key is handed down
rather than re-invented, which is also why `uuid4` appears here only for the task id a person
reads.

The host has to actually use it — drive the child through an `AgentRuntime` on that run id, on the
store the parent uses. Ignoring the argument is allowed and costs exactly this property. `Remote`
children cannot have it at all: what is on the other side of the POST owns its own durability, and
this end has no way to key it.
"""

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from .background import BackgroundResult, BackgroundTasks
from .contracts.events import EventType
from .contracts.types import Emit, Tools

__all__ = [
    "Answering",
    "Compiled",
    "Declarative",
    "Deliver",
    "Remote",
    "Reply",
    "Runner",
    "Subagent",
    "Subagents",
]

Definitions = list[dict[str, Any]]
"""Tool definitions. Named because `Subagents.list` shadows the builtin inside the class body."""

Reply = Callable[[str, bool], Awaitable[None]]
"""A child answering its parent, on purpose. `(text, is_error)`."""

Runner = Callable[[str, Reply, str], AsyncIterator[dict[str, Any]]]
"""Drive a child for one input and yield its engine events. `(prompt, reply, run_id)`.

The `Reply` is handed in rather than closed over because it is per-launch: it names the task the
answer belongs to. A child built for `sync` may ignore it — its last word is its answer either
way — but a handed-off child has no other way home, so `Answering` puts it in reach as a tool.

The `run_id` is the child's, derived from the parent's — see `Subagents._child_run_id`. Give it
to whatever drives the child and its effects land in the ledger under a name a retry reaches
again; ignore it and the child is a fresh anonymous run every time it is asked for.
"""

Deliver = Callable[[BackgroundResult], Awaitable[None]]
"""Where a settled background result goes. Wire it to `AgentRuntime.background_sink(run_id)`."""

DEFAULT_MAX_DEPTH = 5
DEFAULT_TIMEOUT = 300.0
DEFAULT_BLOCKED_TOOLS_FOR_CHILD = ("delegate",)
"""A child that can delegate can delegate back. The reference also strips `handraise` and
`skill-manage`; neither exists here yet, so naming them would be decoration."""


@runtime_checkable
class Named(Protocol):
    """The two fields every subagent kind shows the model."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Declarative:
    """A child described, not built. `Subagents.factory` turns it into a `Runner` at call time."""

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    model: str = ""


@dataclass(frozen=True, slots=True)
class Compiled:
    """A child already wired by the host, reached through the `Runner` it closed over."""

    name: str
    description: str
    run: Runner


@dataclass(frozen=True, slots=True)
class Remote:
    """A child behind an HTTP endpoint. `POST {"input": ...}`, and the body is the answer.

    `urllib` rather than a client library: this is one POST, and the core's dependency list is
    three entries and an HTTP client is not going to be the fourth. It blocks, so it runs in a
    worker thread.
    """

    name: str
    description: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT


Subagent = Declarative | Compiled | Remote


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What draining a child produced."""

    content: str
    is_error: bool


class Answering:
    """Child-side: the one tool a handed-off agent answers its parent with.

    A handoff gives a child the work and lets the parent carry on, which means the parent is no
    longer reading the child's stream for an answer. Something has to close that loop deliberately,
    or the child finishes talking into a room nobody is in. So `respond_to_parent` is a real tool
    with a real effect — and `terminates_loop`, because a child that has answered is done, and
    letting it keep going would let it answer twice.

    Compose it into whatever tools the child already has:

        Answering(child_tools, reply)
    """

    def __init__(self, tools: Tools, reply: Reply) -> None:
        """Wrap a child's tools with the way back to its parent."""
        self._tools = tools
        self._reply = reply
        self.answered = False
        """Whether the child used it. A handed-off child that never did is worth reporting."""

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Answer the parent, or hand the call to the child's own tools."""
        if name != "respond_to_parent":
            return await self._tools.execute(name, call_id, arguments)
        args = arguments if isinstance(arguments, dict) else {}
        answer = str(args.get("result", "")).strip()
        if not answer:
            return _error("result is required — say what you found before you finish")
        await self._reply(answer, bool(args.get("is_error", False)))
        self.answered = True
        return {"type": "text", "text": "Answer delivered to the parent agent."}

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a tool definition by name, ours or the child's."""
        return _REPLY_TOOL if name == "respond_to_parent" else self._tools.get(name)

    def list(self) -> Definitions:
        """The child's tools, plus the way home."""
        return [
            _REPLY_TOOL,
            *(item for item in self._tools.list() if item["name"] != "respond_to_parent"),
        ]


_REPLY_TOOL: dict[str, Any] = {
    "name": "respond_to_parent",
    "description": (
        "Return your answer to the agent that delegated this task to you. Call this exactly once, "
        "when you have the answer — it ends your run, and it is the only way your work reaches "
        "the agent waiting on it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "result": {"type": "string", "description": "What you found, in full."},
            "is_error": {
                "type": "boolean",
                "description": "True when you are reporting a failure rather than an answer.",
            },
        },
        "required": ["result"],
    },
    "terminates_loop": True,
}


class Subagents:
    """Wrap `tools` with `delegate` and the background-task leash."""

    def __init__(
        self,
        tools: Tools,
        subagents: Sequence[Subagent],
        *,
        run_id: str = "",
        factory: Callable[[Declarative], Runner | Awaitable[Runner]] | None = None,
        deliver: Deliver | None = None,
        registry: BackgroundTasks | None = None,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        default_timeout: float = DEFAULT_TIMEOUT,
        background_timeout: float | None = None,
        blocked_tools_for_child: Sequence[str] = DEFAULT_BLOCKED_TOOLS_FOR_CHILD,
        on_child_event: Callable[[str, dict[str, Any]], None] | None = None,
        emit: Emit | None = None,
    ) -> None:
        """Compose delegation over a host's tools.

        `deliver` is what makes `async` delegation worth using; without it a background child runs
        and its answer is logged into the void, so the tool says so rather than pretending.
        `blocked_tools_for_child` is advisory here — it is reported to `factory`, which is the only
        thing that knows how to build a child's toolset.

        `run_id` is the parent's, and every child id is derived from it. Left empty, children are
        keyed by call id alone — stable within a run, which is all a host with no ledger needs, and
        not enough to tell two runs' children apart.
        """
        self._tools = tools
        self._run_id = run_id
        self._subagents: dict[str, Subagent] = {agent.name: agent for agent in subagents}
        self._factory = factory
        self._deliver = deliver
        self.tasks = registry if registry is not None else BackgroundTasks()
        self._depth = depth
        self._max_depth = max_depth
        self._default_timeout = default_timeout
        self._background_timeout = background_timeout
        self.blocked_tools_for_child = tuple(blocked_tools_for_child)
        self._on_child_event = on_child_event
        self._emit = emit
        self._sending: set[asyncio.Task[None]] = set()
        self._running: set[asyncio.Task[None]] = set()
        """Independent agents this tool opened. A reference so the event loop cannot collect
        one mid-run — deliberately not `self.tasks`, which is what `cancel_task` reaches."""

    # ── Tools ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a tool definition by name, ours or the wrapped host's."""
        mine = next((item for item in self._definitions() if item["name"] == name), None)
        return mine if mine is not None else self._tools.get(name)

    def list(self) -> Definitions:
        """Every tool the model may call. Ours shadow a host tool of the same name."""
        ours = self._definitions()
        taken = {item["name"] for item in ours}
        return [*ours, *(item for item in self._tools.list() if item["name"] not in taken)]

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Run one of ours, or hand the call back to the wrapped tools."""
        args = arguments if isinstance(arguments, dict) else {}
        match name:
            case "delegate":
                return await self._delegate(call_id, args)
            case "check_tasks":
                return self._check_tasks()
            case "cancel_task":
                return self._cancel_task(args)
            case "read_task_output":
                return self._read_task_output(args)
            case "watch_task":
                return self._watch_task(args)
            case _:
                return await self._tools.execute(name, call_id, arguments)

    # ── delegate ─────────────────────────────────────────────────────────────

    async def _delegate(self, call_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """One hop, a batch of hops, or a launch."""
        if (tasks := args.get("tasks")) is not None:
            return await self._fan_out(call_id, tasks)

        name = args.get("agent")
        if not isinstance(name, str) or not name:
            return _error("agent is required")
        if "input" not in args:
            return _error("input is required")
        agent = self._subagents.get(name)
        if agent is None:
            known = ", ".join(sorted(self._subagents)) or "(none registered)"
            return _error(f'No subagent named "{name}". Available: {known}')

        depth = self._depth + 1
        if depth > self._max_depth:
            return _error(
                f"Delegation depth {depth} exceeds max {self._max_depth}. This usually means "
                "agents are delegating in a cycle. Review the delegation graph."
            )

        child_run = self._child_run_id(call_id)
        wait = _wait_mode(args.get("wait"))
        if wait == "sync":
            timeout = _positive(args.get("timeout")) or self._default_timeout
            answer = await self._blocking(agent, args["input"], timeout, child_run)
            return _as_result(agent.name, answer)
        if wait == "async":
            return self._hand_off(agent, args["input"], child_run)
        return self._open(agent, args["input"], child_run)

    def _child_run_id(self, call_id: str) -> str:
        """The run id a child is driven under, derived so a retried call reaches the same one."""
        return f"{self._run_id}:{call_id}" if self._run_id else call_id

    async def _blocking(
        self, agent: Subagent, payload: Any, timeout: float, child_run: str
    ) -> _Outcome:
        """Run a child to its answer while the parent's round waits on it.

        A child that used `respond_to_parent` said which of its words were the answer, so that wins
        over whatever it happened to finish with. One that never called it still answers — its last
        turn is its answer, which is how a subagent with no reply tool has always worked.
        """
        spoken: list[_Outcome] = []

        async def reply(text: str, is_error: bool) -> None:
            spoken.append(_Outcome(text, is_error))

        drained = await self._call(agent, payload, timeout, reply, child_run)
        return spoken[-1] if spoken else drained

    async def _fan_out(self, call_id: str, tasks: Any) -> dict[str, Any]:
        """Several children at once, in one call, and one answer covering all of them.

        Deterministic parallelism: the caller gets a fan-out because it asked for one, rather than
        because the model happened to emit several `delegate` calls in a single round.
        """
        if not isinstance(tasks, list) or not tasks:
            return _error("tasks must be a non-empty array")
        for item in tasks:
            if not isinstance(item, dict) or not item.get("agent") or "input" not in item:
                return _error('each item in "tasks" requires { agent, input }')

        async def one(item: dict[str, Any], index: int) -> dict[str, Any]:
            return await self._delegate(
                f"{call_id}#{index}",
                {"agent": item["agent"], "input": item["input"], "timeout": item.get("timeout")},
            )

        # `return_exceptions` for the same reason the concurrent tool round uses it: one child
        # blowing up must not abandon its siblings mid-flight.
        outcomes = await asyncio.gather(
            *(one(item, index) for index, item in enumerate(tasks)), return_exceptions=True
        )
        rendered = []
        for index, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                body = f"ERROR: {type(outcome).__name__}: {outcome}"
            elif outcome.get("type") == "error":
                body = f"ERROR: {outcome.get('message', '')}"
            else:
                body = str(outcome.get("text", ""))
            rendered.append(f"## delegate[{index}] {tasks[index]['agent']}\n{body}")
        return {"type": "text", "text": "\n\n".join(rendered)}

    def _hand_off(self, agent: Subagent, payload: Any, child_run: str) -> dict[str, Any]:
        """Start a child the parent still owns, and answer without waiting for it."""
        if self._deliver is None:
            return _error(
                'wait="async" needs a delivery sink so the child\'s answer can reach you; this '
                'Subagents was built without one. Use wait="sync", or construct it with '
                "deliver=runtime.background_sink(run_id)."
            )
        task_id = f"task-{uuid4().hex[:12]}"
        # The registry holds the reference; without one the loop only weakly references a task and
        # is free to collect a child mid-run.
        task = asyncio.ensure_future(self._pump(task_id, agent, payload, child_run))
        self.tasks.register(task_id, "subagent", agent.name, task)
        return {
            "type": "text",
            "text": (
                f"[subagent {agent.name}] launched as background task {task_id}; it answers you "
                "on a later round, when it has something to say. Keep working. check_tasks for "
                f'status, cancel_task with "{task_id}" to stop it.'
            ),
        }

    def _open(self, agent: Subagent, payload: Any, child_run: str) -> dict[str, Any]:
        """Open an independent agent and hand back its address, not a leash.

        `wait="none"` is not "a child whose answer we throw away" — in the reference it is the one
        mode that never reaches a caller-owned child at all (`delegate.ts:485-499`: an inline
        subagent is executed synchronously and the `none` branch below it is for registry peers
        only). What it hands work to is an autonomous agent with its own lifecycle, and the caller
        gets no result because the result was never the caller's to have.

        So this deliberately does *not* register the run in `self.tasks`. A task in there is
        something `cancel_task` can kill, and a thing the parent can kill is not independent. What
        the parent gets instead is the run id, which is the whole of what it needs: `AgentRuntime`
        is keyed by run id, so anyone holding one can steer that agent (`submit`), answer its
        permission suspensions, drive it further (`run`), or read what it did out of the ledger —
        the user included. An agent opened without its address would be one nobody could ever
        reach again.
        """
        running = asyncio.ensure_future(self._independent_run(agent, payload, child_run))
        # A reference, not a leash: the event loop only weakly references tasks, so an unheld one
        # may be collected mid-run. Nothing here exposes a way to cancel it.
        self._running.add(running)
        running.add_done_callback(self._running.discard)
        return {
            "type": "text",
            "text": (
                f'[agent {agent.name}] opened as independent run "{child_run}". It owns its own '
                "outcome from here — you get no result and no way to stop it. Its work is on "
                "that run id, which is how you, or a person, reach it again."
            ),
        }

    async def _independent_run(self, agent: Subagent, payload: Any, child_run: str) -> None:
        """Drive an agent that answers to nobody here. Its outcome belongs to its own run."""

        async def unheard(_text: str, _is_error: bool) -> None:
            """Accept a reply and drop it: there is no parent waiting on this one."""

        outcome = await self._call(agent, payload, self._background_timeout, unheard, child_run)
        await self._announce(
            EventType.SUBAGENT_STOP, agent, reason=_reason(outcome), run_id=child_run
        )

    async def _pump(self, task_id: str, agent: Subagent, payload: Any, child_run: str) -> None:
        """Drive a handed-off child, and let it answer the moment it decides to.

        The answer travels when the child calls `respond_to_parent`, not when its process ends —
        that is what separates a handoff from waiting: the child owns the timing. A child that
        never calls it still answers, from whatever it finished with, so handing work to an agent
        that was never given the tool loses nothing.
        """
        answered = False

        async def reply(text: str, is_error: bool) -> None:
            nonlocal answered
            answered = True
            await self._settled(task_id, agent, _Outcome(text, is_error), child_run)

        try:
            outcome = await self._call(
                agent, payload, self._background_timeout, reply, child_run
            )
        except asyncio.CancelledError:
            # `cancel_task` already recorded `cancelled` and nobody is waiting on this answer.
            raise
        if not answered:
            await self._settled(task_id, agent, outcome, child_run)
        else:
            self.tasks.settle(task_id, "done")

    async def _settled(
        self, task_id: str, agent: Subagent, outcome: _Outcome, child_run: str
    ) -> None:
        """Record how a handed-off child ended, and send its answer home."""
        self.tasks.settle(task_id, "error" if outcome.is_error else "done")
        await self._announce(
            EventType.SUBAGENT_STOP, agent, reason=_reason(outcome), run_id=child_run
        )
        if self.tasks.status(task_id) == "cancelled" or self._deliver is None:
            return
        await self._deliver(
            BackgroundResult(
                task_id=task_id,
                kind="subagent",
                label=agent.name,
                content=outcome.content or "(no output)",
                is_error=outcome.is_error,
            )
        )

    # ── Running a child ──────────────────────────────────────────────────────

    async def _call(
        self, agent: Subagent, payload: Any, timeout: float | None, reply: Reply, child_run: str
    ) -> _Outcome:
        """One child, run to its answer, bounded when a timeout was asked for."""
        prompt = payload if isinstance(payload, str) else json.dumps(payload, default=str)
        await self._announce(EventType.SUBAGENT_START, agent, task=prompt, run_id=child_run)
        try:
            if timeout is None:
                return await self._drain(agent, prompt, reply, child_run)
            async with asyncio.timeout(timeout):
                return await self._drain(agent, prompt, reply, child_run)
        except TimeoutError:
            return _Outcome(f'subagent "{agent.name}" exceeded {timeout}s and was stopped.', True)
        except asyncio.CancelledError:
            raise
        except Exception as failure:
            return _Outcome(f'subagent "{agent.name}": {type(failure).__name__}: {failure}', True)

    async def _drain(self, agent: Subagent, prompt: str, reply: Reply, child_run: str) -> _Outcome:
        """Read a child to its terminal event. Remote children answer in one piece instead."""
        if isinstance(agent, Remote):
            return await asyncio.to_thread(_post, agent, prompt)

        run = (await self._runner(agent))(prompt, reply, child_run)
        content, is_error = "", False
        async for event in run:
            if self._on_child_event is not None:
                self._on_child_event(agent.name, event)
            match event:
                case {"type": "done", "content": str(text)}:
                    content = text
                case {"type": "error", "message": str(message)}:
                    content, is_error = message, True
        return _Outcome(content, is_error)

    async def _runner(self, agent: Subagent) -> Runner:
        """The callable that drives this child."""
        if isinstance(agent, Compiled):
            return agent.run
        if not isinstance(agent, Declarative):  # pragma: no cover - Remote answered above
            raise TypeError(f"unsupported subagent: {type(agent).__name__}")
        if self._factory is None:
            raise RuntimeError(
                f'declarative subagent "{agent.name}" needs a factory to be built into a runner'
            )
        built = self._factory(agent)
        return await built if isinstance(built, Awaitable) else built

    async def _announce(self, event: EventType, agent: Subagent, **payload: Any) -> None:
        """Say publicly that a child started or stopped.

        `SUBAGENT_START`/`SUBAGENT_STOP` were in the event vocabulary before anything could raise
        them; this is what raises them. Dropped rather than raised when a sink is unwell, the way
        every other observation in this runtime is.
        """
        if self._emit is None:
            return
        await self._emit(event.value, {"agent_id": agent.name, **payload})

    # ── The leash ────────────────────────────────────────────────────────────

    def _check_tasks(self) -> dict[str, Any]:
        listed = self.tasks.list()
        if not listed:
            return {"type": "text", "text": "No background tasks."}
        return {"type": "text", "text": json.dumps(listed, default=str)}

    def _cancel_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id", "")).strip()
        if not task_id:
            return _error("task_id is required")
        if not self.tasks.cancel(task_id):
            return _error(f"No running task with id {task_id}.")
        return {"type": "text", "text": f"Cancelled task {task_id}."}

    def _read_task_output(self, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id", "")).strip()
        if not task_id:
            return _error("task_id is required")
        entry = self.tasks.get(task_id)
        if entry is None:
            return _error(f"No task with id {task_id}.")
        if entry.read_output is None:
            return _error(f"Task {task_id} has no captured output.")
        return {"type": "text", "text": entry.read_output() or "(no output yet)"}

    def _watch_task(self, args: dict[str, Any]) -> dict[str, Any]:
        """Arm a one-shot notification. Returns now; the notice arrives as a background result.

        Non-blocking on purpose. A tool that waited would hold the round open, which is the thing
        background tasks exist to avoid.
        """
        raw = args.get("task_ids")
        task_ids = [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []
        if not task_ids:
            return _error("task_ids must be a non-empty array of task ids")
        if unknown := [item for item in task_ids if self.tasks.get(item) is None]:
            return _error(f"Unknown task id(s): {', '.join(unknown)}. Nothing to watch.")
        if self._deliver is None:
            return _error("watch_task needs a delivery sink; this Subagents was built without one")
        mode = "any" if args.get("mode") == "any" else "all"
        note = str(args.get("message", ""))

        def settled(task_id: str) -> bool:
            status = self.tasks.status(task_id)
            return status is not None and status != "running"

        def satisfied() -> bool:
            return all(map(settled, task_ids)) if mode == "all" else any(map(settled, task_ids))

        deliver = self._deliver

        def fire() -> None:
            states = ", ".join(f"{item}={self.tasks.status(item)}" for item in task_ids)
            body = f"tasks settled (mode={mode}): {states}." + (f" {note}" if note else "")
            notice = BackgroundResult(
                task_id=f"watch:{','.join(task_ids)}",
                kind="watch",
                label="watch",
                content=body,
                is_error=any(self.tasks.status(item) == "error" for item in task_ids),
            )
            # Held until it completes: a bare `ensure_future` is only weakly referenced, and a
            # notification collected before it is delivered is a notification nobody gets.
            sending = asyncio.ensure_future(deliver(notice))
            self._sending.add(sending)
            sending.add_done_callback(self._sending.discard)

        if satisfied():
            fire()
            return {"type": "text", "text": f"Watched {len(task_ids)} task(s) — already settled."}

        unsubscribe: list[Callable[[], None]] = []

        def on_settle(_task_id: str, _status: str) -> None:
            if satisfied():
                unsubscribe[0]()
                fire()

        unsubscribe.append(self.tasks.subscribe(on_settle))
        return {
            "type": "text",
            "text": f"Watching {len(task_ids)} task(s) (mode={mode}); you will be notified.",
        }

    # ── Definitions ──────────────────────────────────────────────────────────

    def _definitions(self) -> Definitions:
        roster = "\n".join(f"- {a.name}: {a.description}" for a in self._subagents.values())
        task_id = {"type": "string", "description": "Task id from check_tasks or a launch."}
        return [
            {
                "name": "delegate",
                "description": (
                    "Hand a subtask to another agent. Available subagents:\n"
                    f"{roster or '(none registered)'}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string", "description": "Which subagent to run."},
                        "input": {"description": "The subtask to hand over."},
                        "wait": {
                            "type": "string",
                            "enum": ["sync", "async", "none"],
                            "description": (
                                '"sync" (default) blocks this round and returns the answer — use '
                                "it when you cannot continue without it. "
                                '"async" gives the task away and returns immediately; the child '
                                "answers you on a later round, when it has something to say. "
                                '"none" opens it as an independent agent instead: you get '
                                "its run id and nothing else — no result, and no way to "
                                "stop it."
                            ),
                        },
                        "timeout": {
                            "type": "number",
                            "description": (
                                f"Seconds to wait, sync only. Default {DEFAULT_TIMEOUT}."
                            ),
                        },
                        "tasks": {
                            "type": "array",
                            "description": (
                                "Run several subagents in parallel within this one call and get "
                                "their answers together. Overrides agent/input when set."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent": {"type": "string"},
                                    "input": {},
                                    "timeout": {"type": "number"},
                                },
                                "required": ["agent", "input"],
                            },
                        },
                    },
                },
                # Delegation is not safe to interleave: two children may touch the same workspace,
                # and the batch form already gives the model deterministic parallelism.
                "is_concurrency_safe": False,
            },
            {
                "name": "check_tasks",
                "description": "List the background tasks you launched and their status.",
                "parameters": {"type": "object", "properties": {}},
                "is_read_only": True,
                "is_concurrency_safe": True,
            },
            {
                "name": "cancel_task",
                "description": "Stop a running background task by its id.",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": task_id},
                    "required": ["task_id"],
                },
            },
            {
                "name": "read_task_output",
                "description": "Read what a background task has produced so far, if it streams.",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": task_id},
                    "required": ["task_id"],
                },
                "is_read_only": True,
                "is_concurrency_safe": True,
            },
            {
                "name": "watch_task",
                "description": (
                    "Be notified when background tasks settle. Returns immediately; the notice "
                    "arrives on a later round."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_ids": {"type": "array", "items": {"type": "string"}},
                        "mode": {"type": "string", "enum": ["all", "any"]},
                        "message": {"type": "string"},
                    },
                    "required": ["task_ids"],
                },
                "is_read_only": True,
                "is_concurrency_safe": True,
            },
        ]


def _post(agent: Remote, prompt: str) -> _Outcome:
    """One blocking POST to a remote child, run off the event loop."""
    # The URL is the host's own configuration, not model input: a subagent is registered by
    # the application, and the model only names one that is already there.
    request = urllib.request.Request(
        agent.url,
        data=json.dumps({"input": prompt}).encode(),
        headers={"Content-Type": "application/json", **agent.headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=agent.timeout) as response:
            return _Outcome(response.read().decode(errors="replace"), False)
    except urllib.error.HTTPError as failure:
        return _Outcome(f'remote subagent "{agent.name}" returned {failure.code}', True)
    except Exception as failure:
        return _Outcome(f'remote subagent "{agent.name}" failed: {failure}', True)


def _wait_mode(value: Any) -> str:
    """Collapse every spelling of a wait mode to one.

    Models emit `false` for fire-and-forget — the only non-string among string siblings — and a
    strict comparison drops it into the synchronous path, which is the opposite of what was asked.
    `async` is the reference's own name for this mode (`delegate.ts:228`); `handoff` is accepted
    because it is what the shape is usually called, but the reference's spelling is the canonical
    one the model is shown.
    """
    if value in (False, "false", "none"):
        return "none"
    return "async" if value in ("async", "handoff") else "sync"


def _reason(outcome: _Outcome) -> str:
    return "error" if outcome.is_error else "completed"


def _positive(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and value > 0 else None


def _as_result(name: str, outcome: _Outcome) -> dict[str, Any]:
    if outcome.is_error:
        return _error(outcome.content)
    return {"type": "text", "text": outcome.content or f'subagent "{name}" returned nothing'}


def _error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}
