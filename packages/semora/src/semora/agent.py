"""A class is an agent. Everything about it is in the class body, and an instance is one run.

```python
class Reviewer(Agent):
    \"\"\"Reviews a repository.\"\"\"      # its description, shown to a parent agent that delegates

    llm = "openai:gpt-5"
    store = MemorySteps()              # the ledger; leave it out and the agent runs plain
    transcript = MemoryTranscript()
    uses = (web_search, shell, FileSystem("."))   # functions, toolsets, MCP servers, capabilities

    def __init__(self, repo: Path, **kwargs) -> None:
        self.repo = repo
        super().__init__(**kwargs)

    def prompt(self) -> str:           # or a plain string attribute; a method re-renders per round
        return f"Review {self.repo}."

    @tool
    async def read(self, path: str) -> str: ...

    async def pre_tool_use(self, ctx, call): ...   # any of the seven control points

    on_inputs = Ingress(mask_cards, drop_empty)   # or a composed control as a class attribute
```

An instance holds its own state and is bound to one run:

```python
reviewer = Reviewer(Path("."))
outcome = await reviewer.run("review this repository")   # mints a run id, keeps it
if outcome.suspended:                                     # a park is an outcome, not an error
    outcome = await reviewer.resume({"type": "approve"})  # same instance, same run

# another process, hours later: a fresh instance bound to the same run
outcome = await Reviewer(Path("."), run_id="run-1").recover()
```

A control point defined as a method replaces the parent's; call `super()` to keep it. A control
point assigned as a class attribute composes rules the way semora does — `Permissions(a, b)` lets
a denial win and remembers a suspension — so an agent with several rules declares them that way.

The names are ours, not Pydantic AI's, on purpose: `model`, `name`, `description`,
`instructions`, `toolsets` and `output_type` are properties or decorators on Pydantic AI's
`Agent`, and a class attribute with one of those names would shadow them.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any, overload

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import DeferredToolResults, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.toolsets import AbstractToolset
from semora_store import ExecutionContext, ExecutionStore, Transcript

from .contracts import AgentSuspended, PendingInput
from .controls import Controls
from .dispatch import Command, Recover
from .effects import CONCURRENCY_SAFE
from .ids import new_run_id
from .runtime import AgentRuntime, Outcome

__all__ = ["Agent", "tool"]

_MARK = "__runtime_tool__"


@overload
def tool[F: Callable[..., Any]](function: F, /) -> F: ...
@overload
def tool[F: Callable[..., Any]](
    *,
    concurrency_safe: bool = ...,
    requires_approval: bool = ...,
    name: str | None = ...,
    description: str | None = ...,
) -> Callable[[F], F]: ...
def tool[F: Callable[..., Any]](
    function: F | None = None,
    /,
    *,
    concurrency_safe: bool = False,
    requires_approval: bool = False,
    name: str | None = None,
    description: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a method as one of the agent's tools.

    `concurrency_safe=True` says order cannot matter, so the call may run beside others; without
    it the call is a barrier and the round runs in model order.
    """

    def mark(fn: F) -> F:
        setattr(
            fn,
            _MARK,
            {
                "metadata": {CONCURRENCY_SAFE: True} if concurrency_safe else None,
                "requires_approval": requires_approval,
                "name": name,
                "description": description,
            },
        )
        return fn

    return mark(function) if function is not None else mark


class Agent(PydanticAgent[Any, Any]):
    """A Pydantic AI agent whose definition is its class body and whose instance is one run."""

    # ── what the model sees ──────────────────────────────────────────────────
    llm: Model | str | None = None
    """The model: a `"provider:name"` string or a `Model` instance."""
    prompt: Any = None
    """The instructions: a string, or a method that renders them from instance state each round."""
    output: Any = str
    """The output type. A pydantic model here means structured output."""

    # ── what the agent can do, besides its own `@tool` methods ───────────────
    uses: Sequence[
        Callable[..., Any] | Tool[Any] | AbstractToolset[Any] | AbstractCapability[Any]
    ] = ()
    """Implementations written elsewhere: functions, toolsets, MCP servers, capabilities."""

    # ── durability ───────────────────────────────────────────────────────────
    store: ExecutionStore | Callable[[], ExecutionStore] | None = None
    """The ledger. `None` records nothing: a crash may run a tool again, nothing can park.

    An instance is shared by every agent of this class. A class or a zero-argument function is a
    factory, called once per agent class the first time one is instantiated — so a connection
    pool is opened on first use, not at import.
    """
    transcript: Transcript | Callable[[], Transcript] | None = None
    lease_ttl: float = 60.0
    retry_running: bool = False
    runtime: AgentRuntime
    """Built once per class from the four attributes above, unless `runtime=` binds an instance."""

    def __init__(
        self,
        *,
        run_id: str | ExecutionContext | None = None,
        runtime: AgentRuntime | None = None,
        **overrides: Any,
    ) -> None:
        """Assemble the Pydantic AI agent from the class body.

        Args:
            run_id: Bind this instance to an existing run — the one another process parked or
                lost. Left out, the first `run` mints one.
            runtime: Bind this instance to its own runtime instead of the class's.
            overrides: Reach Pydantic AI's constructor untouched.
        """
        self.run_id: str | None = run_id.run_id if isinstance(run_id, ExecutionContext) else run_id
        self.last: Outcome | None = None
        """The outcome of this instance's latest attempt."""
        self._attempting = False
        if runtime is not None:
            self.runtime = runtime
        elif "runtime" not in vars(type(self)):
            type(self).runtime = AgentRuntime(
                _resolve(type(self), "store"),
                transcript=_resolve(type(self), "transcript"),
                lease_ttl=self.lease_ttl,
                retry_running=self.retry_running,
            )
        tools: list[Any] = self._marked_tools()
        toolsets: list[AbstractToolset[Any]] = []
        capabilities: list[AbstractCapability[Any]] = []
        for item in self.uses:
            if isinstance(item, AbstractCapability):
                capabilities.append(item)
            elif isinstance(item, AbstractToolset):
                toolsets.append(item)
            elif isinstance(item, Tool) or inspect.isroutine(item):
                tools.append(item)
            else:
                raise TypeError(
                    f"{type(self).__name__}.uses holds {item!r}; expected a function, a Tool, "
                    "a toolset, or a capability"
                )
        overrides.setdefault("name", type(self).__name__)
        overrides.setdefault("description", inspect.getdoc(type(self)))
        super().__init__(
            self.llm,
            instructions=self.prompt,  # a bound method is re-rendered by Pydantic AI per request
            output_type=self.output,
            tools=tools,
            toolsets=toolsets or None,
            capabilities=capabilities or None,
            **overrides,
        )

    def _marked_tools(self) -> list[Tool[Any]]:
        """Methods marked `@tool`, bound to this instance."""
        return [
            Tool(getattr(self, attr), **getattr(member, _MARK))
            for attr, member in inspect.getmembers(type(self), lambda m: hasattr(m, _MARK))
        ]

    # ── one instance, one run ────────────────────────────────────────────────

    def _bind(self, run_id: str | ExecutionContext | None) -> str:
        """The run this instance is for. Set once; a second run needs a second instance."""
        wanted = run_id.run_id if isinstance(run_id, ExecutionContext) else run_id
        if self.run_id is None:
            self.run_id = wanted or new_run_id()
        elif wanted is not None and wanted != self.run_id:
            raise RuntimeError(
                f"{type(self).__name__} is bound to run {self.run_id!r}; "
                f"make another instance for {wanted!r}"
            )
        return self.run_id

    async def _attempt(self, run_id: str, attempt: Any) -> Outcome:
        """Run one attempt; a park comes back as an outcome, and only one attempt at a time."""
        if self._attempting:
            raise RuntimeError(f"{type(self).__name__} already has an attempt in flight")
        self._attempting = True
        try:
            outcome: Outcome = await attempt  # or a dispatch receipt, see below
        except AgentSuspended as parked:
            outcome = Outcome(None, "suspended", None, tuple(parked.pending))
        finally:
            self._attempting = False
        if isinstance(outcome, Outcome):  # a queued prompt is a receipt, not an attempt
            self.last = outcome
        return outcome

    async def run(  # type: ignore[override]
        self,
        user_prompt: str | None = None,
        *,
        run_id: str | ExecutionContext | None = None,
        controls: Controls | None = None,
        rules_version: str = "",
        prompt_id: str | None = None,
        conversation_id: str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        deps: Any = None,
        capabilities: Sequence[AbstractCapability[Any]] = (),
        **options: Any,
    ) -> Outcome:
        """Drive one attempt at this instance's run. The first call mints a run id if none is bound.

        A gate that parks the round ends the attempt with `outcome.suspended`; answer it with
        `resume`. `options` are Pydantic AI's own `run` arguments, forwarded untouched — that is
        what lets a harness `SubAgents` delegate to this agent.
        """
        bound = self._bind(run_id)
        return await self._attempt(
            bound,
            self.runtime.run(
                bound,
                self,
                user_prompt,
                controls=controls,
                rules_version=rules_version,
                prompt_id=prompt_id,
                conversation_id=conversation_id,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                deps=deps,
                capabilities=capabilities,
                **options,
            ),
        )

    async def resume(
        self,
        answer: dict[str, Any],
        *,
        pending_id: str | None = None,
        run_id: str | ExecutionContext | None = None,
        controls: Controls | None = None,
        rules_version: str = "",
        deps: Any = None,
    ) -> Outcome:
        """Route a person's answer to the parked call; current policy re-decides.

        `pending_id` defaults to the first undecided request. In a batch park, an answer for one
        call comes back as a still-suspended outcome listing the rest.
        """
        bound = self._bind(run_id)
        if pending_id is None:
            undecided = await self.runtime.pending(bound)
            if not undecided:
                raise LookupError(f"run {bound!r} has nothing parked")
            pending_id = undecided[0][0]
        return await self._attempt(
            bound,
            self.runtime.resume(
                bound,
                pending_id,
                answer,
                self,
                controls=controls,
                rules_version=rules_version,
                deps=deps,
            ),
        )

    async def recover(
        self,
        *,
        run_id: str | ExecutionContext | None = None,
        history: Sequence[ModelMessage] | None = None,
        controls: Controls | None = None,
        rules_version: str = "",
        conversation_id: str | None = None,
        deps: Any = None,
    ) -> Outcome:
        """Finish an interrupted run. Without `history`, the transcript supplies it.

        Committed effects replay from the record; missing ones run; the model turn is not paid
        for again. A round parked before the crash stays parked.
        """
        bound = self._bind(run_id)
        if history is None:
            attempt = self.runtime.dispatch(
                bound,
                self,
                Recover(),
                controls=controls,
                rules_version=rules_version,
                conversation_id=conversation_id,
                deps=deps,
            )
        else:
            attempt = self.runtime.recover(
                bound,
                self,
                history,
                controls=controls,
                rules_version=rules_version,
                conversation_id=conversation_id,
                deps=deps,
            )
        return await self._attempt(bound, attempt)

    async def fork(
        self,
        source: str | ExecutionContext,
        at: str | None = None,
        prompt: str | None = None,
        *,
        run_id: str | ExecutionContext | None = None,
        history: Sequence[ModelMessage] | None = None,
        regate: bool = False,
        controls: Controls | None = None,
        rules_version: str = "",
        conversation_id: str | None = None,
        deps: Any = None,
        **options: Any,
    ) -> Outcome:
        """Make this instance's run a branch of `source`, from one transcript entry.

        Effects the source finished before `at` replay here; `regate=True` asks this instance's
        gate about them first. The rest runs under this instance's policy.
        """
        bound = self._bind(run_id)
        return await self._attempt(
            bound,
            self.runtime.fork(
                source,
                at,
                bound,
                self,
                prompt,
                history=history,
                regate=regate,
                controls=controls,
                rules_version=rules_version,
                conversation_id=conversation_id,
                deps=deps,
                **options,
            ),
        )

    async def dispatch(
        self,
        command: Command,
        *,
        run_id: str | ExecutionContext | None = None,
        controls: Controls | None = None,
        **options: Any,
    ) -> Outcome | Any:
        """Route a host command to whatever the run's durable state allows."""
        bound = self._bind(run_id)
        return await self._attempt(
            bound, self.runtime.dispatch(bound, self, command, controls=controls, **options)
        )

    async def submit(
        self, item: PendingInput, *, run_id: str | ExecutionContext | None = None
    ) -> PendingInput:
        """Queue input for the run's next model boundary."""
        return await self.runtime.submit(self._bind(run_id), item)

    async def state(self, run_id: str | ExecutionContext | None = None) -> str:
        """Name the run's durable state."""
        return await self.runtime.state(self._bind(run_id))

    async def pending(self, run_id: str | ExecutionContext | None = None) -> list[tuple[str, str]]:
        """The undecided `(pending_id, tool_call_id)` pairs, in model order."""
        return await self.runtime.pending(self._bind(run_id))


def _resolve(cls: type, attribute: str) -> Any:
    """A store declared in the class body: an instance as is, a class or function called once."""
    raw = inspect.getattr_static(cls, attribute, None)
    if isinstance(raw, staticmethod):
        raw = raw.__func__
    if raw is not None and (inspect.isclass(raw) or inspect.isroutine(raw)):
        return raw()
    return raw
