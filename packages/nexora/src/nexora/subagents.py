"""Subagent definitions, delegation tools, and child lifecycle management.

The module supports synchronous delegation, managed background handoff, and independent child
runs. Child run identifiers are derived from parent run and tool-call identifiers so retries
address the same durable execution. Delegated authority can only be attenuated.
"""

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .background import BackgroundResult, BackgroundTasks
from .contracts.agent import AgentDefinition
from .contracts.events import EventType
from .contracts.types import Emit, Tools

__all__ = [
    "Answering",
    "Authority",
    "Deliver",
    "FactoryAgent",
    "HttpAgent",
    "Reply",
    "Runner",
    "RunnerAgent",
    "Subagent",
    "Subagents",
]

Definitions = list[dict[str, Any]]
"""Tool definitions exposed to a chat model."""

Reply = Callable[[str, bool], Awaitable[None]]
"""Callback that delivers a child's text and error status to its parent."""

Runner = Callable[[str, Reply, str], AsyncIterator[dict[str, Any]]]
"""Callable that streams a child run for ``(prompt, reply, run_id)``."""

Deliver = Callable[[BackgroundResult], Awaitable[None]]
"""Callback that delivers a completed background result."""

DEFAULT_MAX_DEPTH = 5
DEFAULT_TIMEOUT = 300.0
DEFAULT_BLOCKED_TOOLS_FOR_CHILD = ("delegate",)
"""Tools removed from delegated child authority by default."""


@dataclass(frozen=True, slots=True)
class FactoryAgent:
    """Child specification built by a runner factory at launch time."""

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    model: str = ""


@dataclass(frozen=True, slots=True)
class RunnerAgent:
    """Subagent with an executable runner supplied by the host."""

    name: str
    description: str
    run: Runner


@dataclass(frozen=True, slots=True)
class HttpAgent:
    """Subagent invoked through an HTTP ``POST`` request.

    The request body is ``{"input": prompt}``, and the response body is used as the result.
    """

    name: str
    description: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT


Subagent = FactoryAgent | RunnerAgent | HttpAgent


@dataclass(frozen=True, slots=True)
class Authority:
    """Maximum tool authority available to a delegated child.

    Attributes:
        ceiling: Allowed tool names, or ``None`` when unrestricted.
        depth: Delegation depth assigned to the child.

    This type computes limits but does not enforce them. Use a permission stage such as
    ``nexora_permissions.escalation_guard`` to enforce ``ceiling``.
    """

    ceiling: tuple[str, ...] | None
    depth: int

    def attenuate(self, requested: Sequence[str] | None) -> "Authority":
        """Derive child authority without widening the current ceiling.

        Args:
            requested: Requested tool names. ``None`` inherits the current ceiling.

        Returns:
            Authority with the intersected ceiling and incremented depth.
        """
        if self.ceiling is None:
            granted = tuple(requested) if requested is not None else None
        elif requested is None:
            granted = self.ceiling
        else:
            granted = tuple(name for name in requested if name in set(self.ceiling))
        return Authority(ceiling=granted, depth=self.depth + 1)

    def without(self, blocked: Sequence[str]) -> "Authority":
        """Remove tool names from a bounded authority.

        Args:
            blocked: Tool names to remove.

        Returns:
            Updated authority. An unrestricted ceiling remains unrestricted.
        """
        if self.ceiling is None:
            return self
        removed = set(blocked)
        return Authority(
            ceiling=tuple(name for name in self.ceiling if name not in removed), depth=self.depth
        )


@dataclass(frozen=True, slots=True)
class _Outcome:
    """Normalized terminal result from a child run."""

    content: str
    is_error: bool


class Answering:
    """Add the ``respond_to_parent`` tool to a child's tool executor.

    The reply tool delivers a background child's result and terminates the child loop.
    """

    def __init__(self, tools: Tools, reply: Reply) -> None:
        """Initialize the reply-tool wrapper.

        Args:
            tools: Child tool executor to wrap.
            reply: Callback invoked by ``respond_to_parent``.
        """
        self._tools = tools
        self._reply = reply
        self.answered = False

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Execute the reply tool or delegate to the wrapped executor."""
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
        """Return a reply or wrapped tool definition by name."""
        return _REPLY_TOOL if name == "respond_to_parent" else self._tools.get(name)

    def list(self) -> Definitions:
        """Return the reply tool followed by wrapped tool definitions."""
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
    """Add delegation and background-task tools to an executor."""

    def __init__(
        self,
        tools: Tools,
        subagents: Sequence[Subagent],
        *,
        run_id: str = "",
        factory: Callable[[FactoryAgent, "Authority"], Runner | Awaitable[Runner]] | None = None,
        authority: Sequence[str] | None = None,
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
        """Initialize delegation over a host tool executor.

        Args:
            tools: Host tool executor.
            subagents: Child definitions addressable by name.
            run_id: Parent run identifier used to derive child run identifiers.
            factory: Builder for declarative child runners.
            authority: Maximum tool names available to this agent.
            deliver: Sink for results from background children.
            registry: Background task registry to reuse across attempts.
            depth: Current delegation depth.
            max_depth: Maximum permitted delegation depth.
            default_timeout: Default timeout for synchronous delegation.
            background_timeout: Optional timeout for background and independent children.
            blocked_tools_for_child: Tool names removed from child authority.
            on_child_event: Observer for child engine events.
            emit: Runtime lifecycle event publisher.
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
        self.authority = tuple(authority) if authority is not None else None
        self._on_child_event = on_child_event
        self._emit = emit
        self._sending: set[asyncio.Task[None]] = set()
        self._running: set[asyncio.Task[None]] = set()
        # Hold independent tasks strongly without exposing them through the cancellable registry.

    # ── Tools ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a delegation or wrapped tool definition by name."""
        mine = next((item for item in self._definitions() if item["name"] == name), None)
        return mine if mine is not None else self._tools.get(name)

    def list(self) -> Definitions:
        """Return all model-visible tools, with delegation tools taking precedence."""
        ours = self._definitions()
        taken = {item["name"] for item in ours}
        return [*ours, *(item for item in self._tools.list() if item["name"] not in taken)]

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Execute a delegation tool or forward the call to the wrapped executor."""
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
        # `mode` rides along because the three are three different relationships, and a rail that
        # only hears "a child started" cannot tell a leashed one from an agent that left. Announced
        # before the launch, so the address reaches whoever is listening even when the child
        # detaches — its own first breath happens after this round is over.
        await self._announce(
            EventType.SUBAGENT_START,
            agent,
            task=_as_prompt(args["input"]),
            run_id=child_run,
            mode=wait,
        )
        if wait == "sync":
            timeout = _positive(args.get("timeout")) or self._default_timeout
            answer = await self._blocking(agent, args["input"], timeout, child_run)
            return _as_result(agent.name, answer)
        if wait == "async":
            return self._hand_off(agent, args["input"], child_run)
        return self._open(agent, args["input"], child_run)

    def _child_run_id(self, call_id: str) -> str:
        """Derive a retry-stable child run identifier."""
        return f"{self._run_id}:{call_id}" if self._run_id else call_id

    async def _blocking(
        self, agent: Subagent, payload: Any, timeout: float, child_run: str
    ) -> _Outcome:
        """Run a child synchronously and publish its terminal lifecycle event."""
        spoken: list[_Outcome] = []

        async def reply(text: str, is_error: bool) -> None:
            spoken.append(_Outcome(text, is_error))

        drained = await self._call(agent, payload, timeout, reply, child_run)
        answer = spoken[-1] if spoken else drained
        await self._announce(
            EventType.SUBAGENT_STOP, agent, reason=_reason(answer), run_id=child_run
        )
        return answer

    async def _fan_out(self, call_id: str, tasks: Any) -> dict[str, Any]:
        """Run an explicit batch of child delegations concurrently."""
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
        """Start a managed background child without waiting for completion."""
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
        """Start an independent child and return its run identifier."""
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
        """Drive an independent child without delivering its result to the parent."""

        async def unheard(_text: str, _is_error: bool) -> None:
            """Discard replies from an independent child."""

        outcome = await self._call(agent, payload, self._background_timeout, unheard, child_run)
        await self._announce(
            EventType.SUBAGENT_STOP, agent, reason=_reason(outcome), run_id=child_run
        )

    async def _pump(self, task_id: str, agent: Subagent, payload: Any, child_run: str) -> None:
        """Drive a managed background child and deliver its result."""
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
        """Record a background child's terminal state and deliver its result."""
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
        """Run one child and normalize timeout and execution failures."""
        prompt = _as_prompt(payload)
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
        """Consume child events and return the terminal outcome."""
        if isinstance(agent, HttpAgent):
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

    def authority_for(self, agent: Subagent) -> Authority:
        """Compute the authority inherited by a child.

        Args:
            agent: Child whose requested tools should be evaluated.

        Returns:
            Attenuated authority with blocked child tools removed.
        """
        requested = agent.tools if isinstance(agent, FactoryAgent) and agent.tools else None
        mine = Authority(ceiling=self.authority, depth=self._depth)
        return mine.attenuate(requested).without(self.blocked_tools_for_child)

    async def _runner(self, agent: Subagent) -> Runner:
        """Resolve a supplied or factory-built child runner."""
        if isinstance(agent, RunnerAgent):
            return agent.run
        if not isinstance(agent, FactoryAgent):  # pragma: no cover - HttpAgent answered above
            raise TypeError(f"unsupported subagent: {type(agent).__name__}")
        if self._factory is None:
            raise RuntimeError(
                f'FactoryAgent "{agent.name}" needs a factory to be built into a runner'
            )
        built = self._factory(agent, self.authority_for(agent))
        return await built if isinstance(built, Awaitable) else built

    async def _announce(self, event: EventType, agent: AgentDefinition, **payload: Any) -> None:
        """Publish a child lifecycle event when an emitter is configured."""
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
        """Register a non-blocking notification for background task completion."""
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


def _post(agent: HttpAgent, prompt: str) -> _Outcome:
    """Send a blocking HTTP request to an HTTP child."""
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


def _as_prompt(payload: Any) -> str:
    """Serialize structured child input as JSON."""
    return payload if isinstance(payload, str) else json.dumps(payload, default=str)


def _wait_mode(value: Any) -> str:
    """Normalize supported wait-mode spellings."""
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
