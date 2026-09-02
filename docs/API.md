# Semora API reference

Written for an agent writing code against Semora. Every signature below is the real one; every
example is minimal and runnable in isolation. Prose that explains *why* a design is what it is
lives in [AGENTS.md](../AGENTS.md) — this file only says what to call.

- [Mental model](#mental-model)
- [Install](#install)
- [Quickstart, four shapes](#quickstart-four-shapes)
- [`semora` — core](#semora--core)
  - [`Agent`](#agent)
  - [`AgentRuntime`](#agentruntime)
  - [Outcome dict](#outcome-dict)
  - [Exceptions](#exceptions)
- [Tools](#tools)
  - [The `Tools` protocol](#the-tools-protocol)
  - [Tool result shape](#tool-result-shape)
  - [`builtin_tools()`](#builtin_tools)
  - [Tool wrappers](#tool-wrappers)
- [Controls — the seven decision points](#controls--the-seven-decision-points)
- [Suspension, resume, recovery](#suspension-resume-recovery)
- [Orchestrators](#orchestrators)
  - [`Orchestrator` — a durable workflow](#orchestrator--a-durable-workflow-in-ordinary-python)
  - [`ModelFailurePolicy`](#modelfailurepolicy--the-shipped-retry-policy)
  - [`semora.orchestration`](#semoraorchestration--replacing-durability-under-agentruntime)
- [`dispatch` — one command table for every host](#dispatch--one-command-table-for-every-host)
- [Stores](#stores)
- [Models](#models)
- [Feature packs](#feature-packs)
- [Workspaces](#workspaces)
- [Pitfalls](#pitfalls)

---

## Mental model

Semora is **not** a workflow engine and **not** a message/model abstraction.

- Messages, tool calls and chat models are **LangChain's** (`BaseMessage`, `ToolCall`,
  `BaseChatModel`). Semora never wraps or translates them.
- Semora owns the **execution boundary**: call-id idempotency, permission parking, crash
  recovery, and the control plane deciding whether a round may start, a tool may run, or a
  finish may stand.
- One planner (`semora.engines.plain.react_loop`), one execution path. Durability lives in the
  orchestrator + store, never in a second graph engine.

Two axes you always choose:

| axis | no | yes |
|---|---|---|
| durability | `AgentRuntime()` — runs, cannot suspend or recover | `AgentRuntime(execution_store=...)` |
| policy | no `controls=` — nothing is gated | `controls=ControlPlane(...)` |

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
uv add semora                  # runtime + in-memory MemorySteps + ChatModel (OpenAI-compatible)
uv add "semora[anthropic]"     # native Anthropic; also: openai, google, xai, openrouter
uv add "semora[postgres]"      # Postgres ledger + transcript
uv add "semora[permissions]"   # rule table
uv add "semora[fork]"          # branch a run
uv add "semora[ui]"            # local console at :8790
```

| extra | package it pulls | import root |
|---|---|---|
| (base) | `semora`, `semora-store`, `semora-llm` | `semora`, `semora_store`, `semora_llm` |
| `openai` / `anthropic` / `google` / `xai` | `langchain-*` | use LangChain's class directly |
| `openrouter` | same adapter as `openai` | `semora_llm.openrouter` |
| `postgres` | `semora-store-pg` | `semora_store_pg` |
| `permissions` | `semora-permissions` | `semora_permissions` |
| `fork` | `semora-fork` | `semora_fork` |
| `ui` | `semora-ui` | `semora_ui` |

## Quickstart, four shapes

**1. Nothing durable.** The tool runs once because the process lived. A crash may re-run it.

```python
from semora import Agent, AgentRuntime, ChatModel, new_run_id

agent = Agent("reviewer", "Reviews a repo", ChatModel(model="gpt-4.1"), tools,
              system_prompt="Inspect evidence before answering.")
outcome = await AgentRuntime().run(new_run_id(), agent, "inspect this repository")
```

**2. With a ledger.** Now suspension has somewhere to park, and recovery has something to replay.

```python
from semora import AgentRuntime, MemorySteps

runtime = AgentRuntime(execution_store=MemorySteps())   # `store=` is the accepted alias
outcome = await runtime.run(run_id, agent, "inspect this repository")
```

**3. With policy.**

```python
from semora import ControlPlane, FinishPolicy, Permissions, gate

await runtime.run(run_id, agent, "review this change", controls=ControlPlane(
    pre_tool_use=Permissions(gate(no_deleting)),
    before_finish=FinishPolicy(require_citation),
))
```

**4. As a host (HTTP / CLI / queue worker).** One question — may this command arrive now?

```python
from semora.dispatch import Prompt
from semora_store import MemoryTranscript

runtime = AgentRuntime(execution_store=MemorySteps(), transcript=MemoryTranscript())
result = await runtime.dispatch(run_id, agent, Prompt("also check the tests"))
```

---

## `semora` — core

Top-level exports: `Agent`, `AgentDefinition`, `AgentRuntime`, `ChatModel`, `Continue`,
`ControlPlane`, `Controls`, `Ctx`, `Deny`, `ExecutionContext`, `FinishPolicy`, `Halt`,
`HostWorkspaceProvider`, `Ingress`, `MemorySteps`, `PendingInput`, `Permissions`, `Proceed`,
`ResumeInput`, `Suspend`, `ToolCall`, `Tools`, `gate`, `new_run_id`, `run`, `__version__`.

Anything else is on the submodule that owns it (`semora.controls`, `semora.tools`,
`semora.dispatch`, `semora.orchestration`, `semora.orchestrator`, `semora.builtins`,
`semora.workspace`, `semora.skills`, `semora.subagents`, `semora.plan_mode`, `semora.goal`,
`semora.tool_search`, `semora.background`, `semora.providers`, `semora.transcript`,
`semora.engines.plain`).

### `Agent`

```python
Agent(name: str, description: str, model: BaseChatModel, tools: Tools,
      system_prompt: str | SystemPromptSource | None = None)
```

Model-visible identity + executable tools. Deliberately **not** on it: prompts, history,
controls, workspaces, orchestration — those are per-attempt policy and live on `AgentRuntime`.

`SystemPromptSource` is a protocol with `async def render(self) -> str`, re-rendered at each
model-round boundary (use it for a prompt that changes mid-run: plan mode, skills catalog).

### `AgentRuntime`

```python
AgentRuntime(
    *,
    execution_store: ExecutionStore | None = None,   # ledger; `store=` is the alias
    store: ExecutionStore | None = None,
    orchestrator: RuntimeOrchestrator | None = None, # full control over durability
    transcript: Transcript | None = None,            # required by dispatch()/committed_history()
    emit: Callable[[str, dict], Awaitable[Any]] | None = None,
    event_sink: Callable[[EventEnvelope], Awaitable[None]] | None = None,
    owner: str = "local",                            # lease owner id for this worker
    lease_ttl: float = 60.0,
    workspace_provider: WorkspaceProvider | None = None,
    workspace_manifest: Mapping[str, Any] | None = None,
    workspace_seed_dirs: Sequence[WorkspaceSeed] = (),
    model_failure_policy: Callable[[ModelFailure], Awaitable[Literal["retry","compact","fail"]]] | None = None,
    compact_context: Callable[[list[BaseMessage], ModelFailure], Awaitable[list[BaseMessage]]] | None = None,
)
```

`execution_store=X` is shorthand for `orchestrator=DurableRuntimeOrchestrator(X)` from
`semora.orchestration`.

#### `run`

```python
async def run(
    run_id: str | ExecutionContext,
    model: BaseChatModel | Agent,
    tools: Tools | str | None = None,     # ← with an Agent, this slot is the prompt
    prompt: str = "",
    *,
    controls: Controls | None = None,
    on_event: Callable[[dict], Awaitable[None]] | None = None,
    on_suspend: OnSuspend | None = None,
    rules_version: str = "",
    prompt_id: str | None = None,
    input_mode: Literal["interactive", "headless"] = "interactive",
    model_identity: str | None = None,
    conversation_id: str | None = None,
    **engine_options,
) -> dict[str, Any]
```

Two accepted call shapes — the second is why `tools` is typed `Tools | str`:

```python
await runtime.run(run_id, model, tools, "prompt")   # model + tools
await runtime.run(run_id, agent, "prompt")          # Agent owns tools and system_prompt
```

Passing both an `Agent` and a `Tools` raises `TypeError("Agent owns tools; ...")`; passing
`system_prompt=` beside an `Agent` raises `TypeError("Agent owns system_prompt")`.

#### `resume`

```python
async def resume(run_id, pending_id: str, answer: dict[str, Any],
                 model: BaseChatModel | Agent, tools: Tools | None = None,
                 *, controls=None, on_event=None, on_suspend=None,
                 rules_version="", model_identity=None, conversation_id=None,
                 **engine_options) -> dict[str, Any]
```

Routes an answer to a suspension by its **external** `pending_id`. Raises `LookupError` when no
active suspension carries that id. The answer is an *input* to `on_resume`, never the decision:
current policy re-decides.

#### `recover`

```python
async def recover(run_id, history: list[BaseMessage],
                  model: BaseChatModel | Agent, tools: Tools | None = None,
                  *, controls=None, aborted=lambda: False, retry_running: bool = True,
                  on_event=None, on_suspend=None, rules_version="",
                  model_identity=None, conversation_id=None, **engine_options) -> dict[str, Any]
```

Finish an interrupted round. Committed tool calls replay from the ledger; only the missing ones
execute; the model turn is not re-charged. `retry_running=False` refuses to re-attempt a step
left in `running` (use when the effect may have happened externally).

#### `dispatch`

```python
async def dispatch(run_id, agent: Agent, command: Prompt | Answer | Recover,
                   *, controls=None, **options) -> dict[str, Any]
```

Requires **both** `execution_store=` and `transcript=`; otherwise `TypeError`. See
[dispatch](#dispatch--one-command-table-for-every-host).

#### `state`

```python
async def state(run_id) -> str
```

| value | meaning |
|---|---|
| `waiting` / `switching` / `resuming` | parked — a continuation is persisted |
| `fresh` | transcript has no run record; never ran |
| `completed` | run record has `ended_at` |
| `interrupted` | open round: a crash, **or** a run another worker is driving (only a lease attempt distinguishes them) |
| `idle` | unparked and no transcript configured |

#### `submit` / `committed_history` / `background_sink`

```python
async def submit(run_id, item: PendingInput, *, input_mode="interactive",
                 conversation_id=None) -> PendingInput
async def committed_history(run_id, conversation_id=None) -> list[BaseMessage]   # needs transcript=
def background_sink(run_id, *, conversation_id=None) -> Callable[[BackgroundResult], Awaitable[None]]
```

#### Module-level `run`

```python
async def run(model: BaseChatModel | Agent, tools=None, prompt="", *,
              run_id: str = "default", runtime: AgentRuntime | None = None, **options)
```

Convenience for one turn. Note the different argument order — `run_id` is keyword-only here.

### Outcome dict

`run` / `resume` / `recover` return the loop's terminal event:

```python
{
  "type": "done",
  "content": str,                # final assistant text
  "tool_calls": [{"name": str, "input": dict}, ...],   # every call made this attempt
  "stop_reason": "completed" | "aborted" | "tool" | "policy",
  "interrupted_mid_turn": True,  # only when aborted mid-stream
  "usage": {"prompt_tokens": int, "completion_tokens": int, ...},  # only if the provider reported
  "model": str,                  # only if the provider named one
  "usage_by_model": {...},       # only when more than one model answered
}
```

`stop_reason` values: `completed` (model stopped on its own), `aborted` (`aborted()` returned
true), `tool` (a tool ended the run), `policy` (a `Halt` from `before_finish`/`before_model`).

`dispatch` returns either an outcome dict or `{"type": "enqueued", "input_id": ...}`.

Streamed events (`on_event=`, or iterating `react_loop`) are: `{"type": "thinking", "content"}`,
`{"type": "text", "text"}`, `{"type": "tool_call", "id", "name", "input"}`, and the terminal
`{"type": "done", ...}`, `{"type": "error", "message", "partial", "error_type", "error_kind"}` or
`{"type": "suspended", "pending_id", "tool_call_id", "pending"}`.

### Exceptions

| exception | from | meaning |
|---|---|---|
| `AgentSuspended` | `semora.orchestrator` | the attempt parked. `.pending_id`, `.tool_call_id`, `.pending` (all undecided `(pending_id, tool_call_id)` in model order) |
| `Suspended` | `semora.orchestrator` | base of every parking signal |
| `ControlSignal` | `semora.contracts.types` | base of runtime signals that are *not* tool failures |
| `Contended` | `semora_store` | another worker holds the run lease |
| `Fenced` | `semora_store` | write with a stale fencing token |
| `Indeterminate` | `semora_store` | a step whose external effect may have occurred |
| `EffectConflict` | `semora_store` | a completion contradicting durable state |
| `InvalidTransition` | `semora.dispatch` | the state cannot accept this command. `.state`, `.command` |
| `InvalidToolCall` / `InvalidToolResult` | `semora.tools` | malformed call / result |
| `ModelStepError` | `semora.contracts.types` | durable-boundary failure, past provider classification |

**A tool that raises did not fail the round.** The exception becomes that call's `error` result
and the model is told. Never wrap `runtime.run(...)` in a broad `except` — it converts exactly
the signals the boundary deliberately lets through (`ControlSignal`, `Contended`, `Fenced`,
`Indeterminate`).

```python
from semora.orchestrator import AgentSuspended

try:
    outcome = await runtime.run(run_id, agent, "delete stale.md", controls=controls)
except AgentSuspended as parked:
    pending_id = parked.pending_id          # bind here; Python unbinds `parked` after the block
```

---

## Tools

### The `Tools` protocol

Structural — implement three methods on any object, no base class, no registration:

```python
class Tools(Protocol):
    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]: ...
    def get(self, name: str) -> dict[str, Any] | None: ...     # this tool's own declaration
    def list(self) -> list[dict[str, Any]]: ...                # model-visible schemas
```

```python
class Files:
    async def execute(self, name, call_id, args):
        return {"type": "text", "text": self.contents.get(args["path"], "not found")}

    def get(self, name):
        return {"name": name, "is_concurrency_safe": name == "read"}

    def list(self):
        return [{"name": "read", "description": "read a file", "parameters": {...}}]
```

`get()["is_concurrency_safe"]` is how a tool opts into batching. **A batch runs sequentially
unless every call in it declares itself safe** — two individually idempotent writes to one file
give different results in different orders.

Related protocols:

| protocol | adds | from |
|---|---|---|
| `BatchTools` | `async execute_batch(calls) -> list[dict]` — own the whole gated round's concurrency | `semora.contracts.types` |
| `DynamicTools` | `prepare(messages)` — schemas that depend on history | `semora.contracts.types` |
| `ContextualTools` | `get_context()` / `with_context(ctx)` — rebind to one attempt's workspace | `semora.workspace` |

### Tool result shape

`execute` returns a dict the model sees. Conventions used throughout:

```python
{"type": "text", "text": "..."}                     # normal result
{"type": "error", "message": "..."}                 # failure the model should react to
{"type": "suspend", "pending_id": "approve-c1"}     # from a gate, not from a tool
```

### `builtin_tools()`

```python
from semora.builtins import ExecToolOptions, builtin_tools

tools = builtin_tools(exec_options=ExecToolOptions(allow_list=("git", "pytest")))
```

```python
def builtin_tools(*, context: ToolContext | None = None,
                  exec_options: ExecToolOptions | None = None,
                  web_fetch_options: WebFetchToolOptions | None = None) -> BuiltinTools
```

Supplies `read`, `write`, `edit`, `grep`, `glob`, `Bash`, `web_fetch`. **`web_search` is not
included.** File and process effects go through the `WorkspaceProvider` that `AgentRuntime`
injects (`semora.workspace`).

```python
ExecToolOptions(allow_list=(), allow_shell=False, env_allow_list=(),
                default_timeout_ms=120_000, require_isolation=True, allowed_domains=())
```

**`Bash` is disabled until `allow_list` is non-empty.** `("*",)` permits every bare executable
and is only appropriate when the workspace is a real OS sandbox.

```python
WebFetchToolOptions(transport=None, summarizer=None, cache_ttl_ms=900_000,
                    max_bytes=5_242_880, fetch_timeout_ms=30_000, now=time.time)
```

`transport` defaults to `UrllibWebFetchTransport` (stdlib). `summarizer` is a
`WebFetchSummarizer` protocol — `async summarize(content, prompt) -> str` — so the model used to
summarize pages stays the caller's.

### Tool wrappers

Each wraps a `Tools` and returns a `Tools`, so they compose by nesting.

| wrapper | from | what it adds |
|---|---|---|
| `Concurrent(tools, aborted=...)` | `semora.tools` | batch execution honouring `is_concurrency_safe` |
| `Stepped(tools, orchestrator)` | `semora.tools` | route each call through the durable ledger |
| `SkillTools(inner, registry)` | `semora.skills` | an on-demand `skill` tool |
| `DeferredTools(inner, deferred=..., initially_active=...)` | `semora.tool_search` | hide schemas until `tool_search` activates them |
| `Subagents(tools, subagents, ...)` | `semora.subagents` | `delegate` + background-task tools |
| `Answering(tools, reply)` | `semora.subagents` | `respond_to_parent` on a child |

```python
from semora.tools import absorb_round, as_model_tools   # round bookkeeping helpers
```

---

## Controls — the seven decision points

Policy is a protocol, not a callback bag. Every method is awaited and its **return value**
decides.

```python
from semora import ControlPlane, Ctx, Continue, Deny, Suspend, Proceed, Halt
```

| hook | signature | returns |
|---|---|---|
| `on_inputs` | `(ctx, inputs: list[PendingInput])` | `list[PendingInput]` (possibly rewritten) or `Halt` |
| `before_model` | `(ctx)` | `Proceed` \| `Halt` |
| `pre_tool_use` | `(ctx, call: ToolCall)` | `Continue` \| `Deny` \| `Suspend` |
| `post_tool_use` | `(ctx, call, result: dict)` | `None` (observation only) |
| `before_finish` | `(ctx, reason: StopReason)` | `Proceed` \| `Halt` — **polarity inverts here**, see below |
| `on_resume` | `(ctx, call, resume: ResumeInput)` | `Continue` \| `Deny` \| `Suspend` |
| `on_suspend` | `(ctx, call, request, snapshot, completed)` | `None` (persist the continuation) |

Decision types:

```python
Continue()                 # allow
Deny(result: dict)         # refuse; `result` is what the model sees for that call
Suspend(request: dict)     # park; `request` MUST carry "pending_id"
Proceed(steers: Sequence[BaseMessage] = ())   # continue, optionally injecting steering messages
Halt(reason: StopReason)   # end the run: "completed" | "aborted" | "tool" | "policy"
```

**`Proceed` / `Halt` mean opposite things at the two turn boundaries.** At `before_model`,
`Proceed` starts the round and `Halt` ends the run. At `before_finish` the run is already trying
to end, so `Halt(reason)` *lets the ending stand* and `Proceed([...])` **vetoes it** — the loop
goes around again with those steering messages. A `before_finish` gate that returns `Proceed()`
unconditionally is an infinite loop.

```python
from langchain_core.messages import HumanMessage

async def require_citation(ctx: Ctx, reason: StopReason) -> Proceed | Halt:
    if "http" in ctx.text:
        return Halt(reason)                                     # fine, let it end
    return Proceed([HumanMessage("Cite a source before finishing.")])   # around again
```

`Ctx` — what a hook sees:

```python
Ctx(turn: int, messages: list[BaseMessage], calls_made: list[dict], text: str, subject: str)
```

`ResumeInput` — what `on_resume` sees:

```python
ResumeInput(answer: dict, request: dict, suspended_rules_version: str, current_rules_version: str)
```

Compare the two versions to detect that policy moved while the approval was outstanding.

### Combinators

One per control point; each chains stages in order.

| combinator | control point | rule |
|---|---|---|
| `Ingress(*screens)` | `on_inputs` | each screen sees the previous one's output; a `Halt` stops the chain |
| `Steering(*sources)` | `before_model` | accumulates every source's `Proceed(steers)`; first `Halt` wins |
| `Permissions(*stages)` | `pre_tool_use` | **denial wins, allowance does not short-circuit** — see below |
| `FinishPolicy(*gates)` | `before_finish` | **any veto wins**; all gates run and their steering accumulates |
| `Journal(*writers)` | `post_tool_use` | all writers run in order; the first failure propagates |
| `Suspending(*persisters)` | `on_suspend` | all persisters run before suspension is announced |

`Permissions` does not stop at the first non-`Continue`. It keeps evaluating: a `Deny` returns
immediately, a `Suspend` is remembered (only the first) and later stages still run, so a stage
after a `Suspend` can still `Deny`. The result is `Deny` > `Suspend` > `Continue`, regardless of
order.

There is no combinator for `on_resume` — its signature takes three arguments, so `Permissions`
(two) will not fit. Pass a bare stage: `on_resume=my_stage` where
`async def my_stage(ctx, call, resume: ResumeInput) -> ToolDecision`.

### Adapters

```python
def gate(answer: Callable[[ToolCall], Awaitable[dict | None]]) -> pre_tool_use stage
def writer(record: Callable[[ToolCall, dict], Awaitable[None]]) -> post_tool_use stage
```

`gate` maps a plain predicate's return value: `None` or `{"type": "allow"}` → `Continue`;
`{"type": "suspend", "pending_id": ...}` → `Suspend`; **anything else → `Deny`**.

```python
async def ask_before_deleting(call):
    if call["name"] == "delete":
        return {"type": "suspend", "pending_id": f"approve-{call['id']}"}
    return None    # allow

controls = ControlPlane(pre_tool_use=Permissions(gate(ask_before_deleting)))
```

---

## Suspension, resume, recovery

There is **one** approval path. An agent asking a human and a policy refusing a tool both end as
a suspension, so neither holds a worker while waiting. Gates must never block synchronously.

```
run() ──pre_tool_use → Suspend({"pending_id": ...})
      └→ continuation written to the ledger
      └→ raise AgentSuspended(pending_id, tool_call_id, pending)     ← the worker exits

  ...hours or days; the process is gone...

resume(run_id, pending_id, answer, model, tools, controls=...)
      └→ on_resume(ctx, call, ResumeInput) decides again under TODAY's rules
      └→ Continue → the parked call runs exactly once → the loop continues
```

```python
store = MemorySteps()          # one store, so the second attempt sees the first's ledger
try:
    await AgentRuntime(store=store).run("run-2", model, files, "delete stale.md", controls=c)
except AgentSuspended as parked:
    pending_id = parked.pending_id

outcome = await AgentRuntime(store=store).resume(
    "run-2", pending_id, {"decision": "approve"}, model2, files, controls=c
)
```

Requirements and guarantees:

- Suspension **requires a store**. Without one there is nowhere to park.
- A suspended call has **not** executed. Assert it: the effect happens only after `on_resume`.
- A stored human approval does **not** outrank current policy — `on_resume` runs the live rules.
- Recovery: a tool call's `id` is its idempotency key. Committed calls replay from the ledger,
  only the uncommitted one executes, and the model turn is not re-run.
- Suspension records carry only what the pause can change. Anything reconstructible from the
  conversation stays out; external facts go in.

---

## Orchestrators

Two different things share the name. Pick by what you are building.

| you want | use | where |
|---|---|---|
| a **durable workflow** whose steps run once ever, with an agent as one of the steps | `Orchestrator` | `semora.orchestrator` |
| to **swap the durability policy** under `AgentRuntime` | `RuntimeOrchestrator` / `DurableRuntimeOrchestrator` | `semora.orchestration` |

`AgentRuntime(execution_store=X)` already builds the second for you. You never need to construct
either to run an agent — reach for `Orchestrator` when the *workflow around* the agent must also
be durable, and for `semora.orchestration` only when you are replacing durability wholesale.

### `Orchestrator` — a durable workflow in ordinary Python

```python
from semora import MemorySteps
from semora.engines.plain import react_loop
from semora.orchestrator import Orchestrator, Suspended, run_agent
```

```python
Orchestrator(run_id: str | ExecutionContext, log: ExecutionStore | None = None, *,
             owner: str = "local", ttl: float = 60.0, emit=None, on_suspend=None,
             on_agent_event=None, rules_version: str = "")
```

One exclusive lease per attempt, renewed at step boundaries; every protected write carries its
fencing token. **Reuse the same `run_id` to resume the workflow** — that is the whole resume
mechanism.

#### The four calls a workflow uses

```python
async def run(step: str, fn: Callable[[], Awaitable[Any] | Any]) -> Any
async def signal(name: str) -> Any               # recorded answer, or suspend the attempt
async def resolve(name: str, answer: Any) -> None  # written from OUTSIDE, then replay
async def force_retry(step: str) -> None         # clear an Indeterminate step
```

```python
async def discharge(orchestrator: Orchestrator, patient_id: str, model) -> dict[str, Any]:
    """Every line is a step. The agent is not a special case."""
    since = await orchestrator.run("day", lambda: date(2026, 8, 6).isoformat())

    plan = await orchestrator.run(
        "draft",
        lambda: run_agent(react_loop(model, Files({"notes.md": f"stable since {since}"}))),
    )

    signed_off = await orchestrator.signal(f"signoff:{patient_id}")   # ends the attempt if unanswered

    await orchestrator.run("meds", lambda: send_to_pharmacy("amoxicillin 500mg"))
    return {"plan": plan["content"], "signed_off": signed_off}
```

Driving it, across three attempts:

```python
log = MemorySteps()

try:                                                    # attempt 1: runs to the signal, stops
    await discharge(Orchestrator("discharge-7", log), "patient-7", model)
except Suspended as waiting:
    print(waiting.signal)                               # "signoff:patient-7"

await Orchestrator("discharge-7", log).resolve("signoff:patient-7", {"by": "dr-kim"})

outcome = await discharge(Orchestrator("discharge-7", log), "patient-7", model)  # attempt 2: finishes
await discharge(Orchestrator("discharge-7", log), "patient-7", model)            # attempt 3: no work
```

**Replay is the mechanism.** The workflow function runs top to bottom on *every* attempt. What
makes that safe is that a finished step returns its recorded value instead of calling `fn` again.
Consequences you must design for:

- `step` names are idempotency keys — **stable and unique within the run**. A name computed from
  a timestamp or a loop index that shifts between attempts re-runs the effect.
- **Put every external effect inside `orchestrator.run(...)`.** A bare `await send_invoice(...)`
  in the function body re-fires on every replay.
- Everything outside a step must be **deterministic and cheap** — it re-executes each attempt.
  `date.today()` belongs inside a step (as `"day"` above), not beside it.
- `signal(name)` raises `Suspended` when the answer does not exist yet. **No worker, no lease, no
  timeout is held** while it waits; the attempt is simply over.
- `resolve()` is called from outside the workflow — an HTTP handler, another process — and then
  the workflow is replayed.

An agent run is a step because `run_agent` collapses its event stream to one value:

```python
async def run_agent(events: AsyncIterator[dict], on_event=None) -> dict[str, Any]
```

Returns the terminal `done` event ([outcome dict](#outcome-dict)) for a completed or
policy-stopped run. Every other ending **raises**, so a step never records a half-run as its
value: an `error` event → `AgentFailed`, a `suspended` event → `AgentSuspended`,
`stop_reason == "aborted"` → `AgentAborted`, and a stream that ends with no terminal event →
`AgentFailed("the agent produced no terminal event")`. That is what stops a human sign-off from
re-running the draft, and stops a retry from sending a prescription twice.

#### Workflow exceptions

| exception | from | meaning |
|---|---|---|
| `Suspended` | `semora.orchestrator` | the attempt is waiting on an unresolved signal. `.signal` names it. A `ControlSignal`, so a tool that suspends from inside its own body stops the attempt instead of being reported as a failed tool |
| `AgentSuspended` | subclass of `Suspended` | a *tool call* parked for approval. `.pending_id`, `.tool_call_id`, `.pending` |
| `AgentFailed` | `semora.orchestrator` | the agent run ended on an error event. `.message`, `.partial`, `.error_type`, `.error_kind` |
| `AgentAborted` | `semora.orchestrator` | interrupted before producing an outcome. `.partial` |
| `InvalidSuspension` | `semora.orchestrator` | an external suspension identity that cannot route one answer unambiguously |
| `Indeterminate` | `semora_store` | a step whose external effect may have occurred; clear it with `force_retry(step)` only when you know it did not |

#### Opaque continuations

```python
async def suspend(key: str, payload: dict[str, Any]) -> None
async def suspension(key: str) -> dict[str, Any] | None
async def active_continuation() -> dict[str, Any] | None   # None when the run is not parked
```

For parking state the framework should not interpret. `AgentRuntime.state()` reads
`active_continuation()` to report `waiting` / `switching` / `resuming`.

#### What is *not* the workflow API

`Orchestrator` also carries the surface `AgentRuntime` drives on your behalf:
`execute_round`, `invoke_model`, `record_pending`, `pending_calls`, `persist_suspension`,
`repark_suspension`, `record_suspension_answer`, `recover_pending`, `resume_effect`,
`post_tool_use_once`, `continue_with_input`, `cancel_and_switch`, `claim_inputs`,
`admit_inputs`, `discard_inputs`, `enqueue_input`, `commit_transition_inputs`,
`complete_continuation`, `compact_suspension`, `emit`.

Calling these yourself means re-implementing the runtime's ordering guarantees. Use
`AgentRuntime` unless you are replacing it.

Two of their contracts leak into behaviour you can observe, so they are worth knowing:

- `invoke_model` clears a provider failure that happened **before the first chunk** (no output was
  exposed, so the loop may retry it). Once a chunk was visible, an interrupted request stays
  `running` and recovery raises `Indeterminate` rather than risk a duplicate charge and a
  different answer.
- `post_tool_use_once` commits an `after:{call_id}` marker so a replayed result skips a hook
  that already ran. A crash *between* the hook and its marker re-runs the hook — **exactly-once
  still requires the hook to be idempotent per call id**; the marker only bounds the repetition.

### `ModelFailurePolicy` — the shipped retry policy

```python
from semora import AgentRuntime
from semora.orchestrator import ModelFailurePolicy

runtime = AgentRuntime(model_failure_policy=ModelFailurePolicy(max_retries=2, max_compactions=1))
```

```python
ModelFailurePolicy(max_retries: int = 2, max_compactions: int = 1,
                   backoff: Callable[[ModelFailure], Awaitable[None]] | None = None)
```

It is the callable `AgentRuntime(model_failure_policy=...)` expects, so pass it rather than
writing the classifier by hand. Its rules: any failure with `partial` output → `fail`;
`context_overflow` within `max_compactions` → `compact` (your `compact_context=` shortens the
history); `rate_limit` / `server` within `max_retries` → `retry`, awaiting `backoff` first;
everything else → `fail`. Negative bounds raise `ValueError`.

### `semora.orchestration` — replacing durability under `AgentRuntime`

```python
from semora.orchestration import (DurableRuntimeOrchestrator, RuntimeOrchestrator,
                                  RuntimeOrchestrationContext, RuntimeOrchestrationSession,
                                  RuntimeInputSession)

runtime = AgentRuntime(orchestrator=DurableRuntimeOrchestrator(store, owner="worker-3",
                                                               lease_ttl=30.0))
```

`DurableRuntimeOrchestrator(execution_store, *, owner="local", lease_ttl=60.0)` attaches the
ledger, suspension and recovery policy described above. `.session(context)` hands you the
underlying `Orchestrator` for one attempt.

To supply your own, implement three small protocols:

```python
class RuntimeOrchestrator(Protocol):
    def open(self, context: RuntimeOrchestrationContext) -> AbstractAsyncContextManager[RuntimeOrchestrationSession]: ...

class RuntimeOrchestrationSession(Protocol):
    def wrap_model(self, inner: ModelStreamFactory) -> ModelStreamFactory: ...
    def wrap_tools(self, inner: ExecuteRound) -> ExecuteRound: ...

class RuntimeInputSession(Protocol):          # if the attempt admits external inputs
    async def submit(self, item: PendingInput) -> PendingInput: ...
    async def claim(self, represented: set[str] | None = None) -> list[PendingInput]: ...
    async def admit(self, items: list[PendingInput]) -> None: ...
    async def discard(self, items: list[PendingInput]) -> None: ...
```

```python
RuntimeOrchestrationContext(execution: ExecutionContext, emit=None, on_suspend=None,
                            on_agent_event=None, rules_version: str = "")
```

The whole seam is two wrappers: everything durable happens because the model stream and the tool
round are wrapped. There is deliberately no second graph engine or checkpointer path — if you
find yourself adding one, the runtime contract in [AGENTS.md](../AGENTS.md) says not to.

---

## `dispatch` — one command table for every host

An HTTP handler, a CLI and a queue worker ask the same question: *given this run's durable
state, may this command arrive now?* `dispatch` answers it once over the public primitives
(`run`, `resume`, `recover`, `submit`).

```python
from semora.dispatch import Answer, Prompt, Recover, InvalidTransition, default_router
```

| command | fields | meaning |
|---|---|---|
| `Prompt(text, prompt_id=None, input_mode="interactive")` | user input | starts an idle run; on a parked run `interactive` cancels the pending request, `headless` queues behind it |
| `Answer(pending_id, payload)` | answer a suspension by its external id | |
| `Recover()` | finish an interrupted run | |

Default table (`default_router()`), in order:

| row | command → primitive |
|---|---|
| `StartRun` | `Prompt` → `run` |
| `QueueSteer` | `Prompt` on a live worker → `submit` (durably enqueued; reached via `StartRun`'s `Contended` hand-off) |
| `ResumeApproval` | `Answer` → `resume` |
| `RecoverInterrupted` | `Recover` on a parked run → `recover` |
| `ReplayJournal` | `Recover` on an open, unparked round → prompt-less `run` |

Behaviour to code against:

- A `Prompt` aimed at a run another worker is driving returns `{"type": "enqueued", "input_id": ...}`.
- `Contended` still propagates from `Answer` / `Recover` — those need the lease themselves; the
  host may retry.
- A command the state cannot accept raises `InvalidTransition`, carrying `.state` and
  `.command` so an adapter maps it (e.g. HTTP 409) without string-matching.
- An `Answer` refused this way may be retried once the park it raced has landed — recording an
  answer is idempotent.

Custom rows implement `Transition` — `applies(command) -> bool` (pure predicate over the command)
plus `states: ClassVar[frozenset[str] | None]` (`None` = every state) — and slot into
`CommandRouter(*transitions)` by order, no subclassing. Drop a row and exactly its behaviour
disappears.

---

## Stores

`semora-store` has no dependencies of its own: a `StepLog` stores opaque values under opaque
keys, so implementing one needs neither a message type nor `semora`.

```python
from semora_store import (ExecutionContext, ExecutionStore, ExecutionTransition,
                          EffectCompletion, InputRecord, MemorySteps, MemoryTranscript,
                          ScopedStore, Step, StepLog, Transcript,
                          Contended, EffectConflict, Fenced, Indeterminate)
```

### `StepLog` / `ExecutionStore`

```python
async def acquire(run_id, owner, ttl_seconds) -> int        # fencing token
async def release(run_id, owner) -> None
async def start(run_id, key, token=0) -> bool               # commit intent BEFORE the effect
async def finish_effect(run_id, key, value, token=0) -> None
async def read(run_id, key) -> Step                         # Step(status, value)
async def forget(run_id, key, token=0) -> None
async def write_control(run_id, key, value, token=0) -> None
async def commit_transition(run_id, transition: ExecutionTransition, token=0) -> set[str]
async def enqueue_input(run_id, input_id, value) -> bool
async def claim_input(run_id, input_id, token=0) -> None
async def admit_inputs(run_id, input_ids, token=0) -> None
async def discard_inputs(run_id, input_ids, token=0) -> None
async def list_inputs(run_id) -> list[InputRecord]
def for_execution(context: ExecutionContext) -> Self
```

- `Step(status: "absent" | "running" | "done", value)`.
- Completed effect results are **immutable**. Mutable continuation/protocol state uses
  `write_control` or `commit_transition`.
- `ExecutionTransition(effects=(EffectCompletion(key, value, expected="running"), ...),
  controls={...}, inputs=((input_id, value), ...))` commits all three atomically.
- Lease-protected writes must present the token from `acquire`; a stale one raises `Fenced`.
- `InputRecord(input_id, status: "pending"|"claimed"|"admitted"|"discarded", value, sequence)`.

### `Transcript`

```python
async def append(entry: dict) -> bool
async def read(conversation_id, *, limit=None) -> list[dict]
async def record_run(run_id, fields: dict) -> None        # keys limited to RUN_FIELDS
async def read_run(run_id) -> dict | None
async def record_model_usage(run_id, model, counts) -> None   # keys limited to MODEL_USAGE_FIELDS
async def read_model_usage(run_id) -> dict[str, dict]
```

`RUN_FIELDS` = `started_at`, `ended_at`, `stop_reason`, `tool_calls`, `interrupted_mid_turn`,
`conversation_id`. `MODEL_USAGE_FIELDS` = `prompt_tokens`, `completion_tokens`, `cached_tokens`,
`cache_write_tokens`, `total_tokens`, `cost_usd`. `check_fields(table, fields, allowed)` raises
`ValueError` on anything else.

Helpers in `semora.transcript`: `TranscriptWriter`, `message_entry`, `marker_entry`, `entry_id`,
`messages_of`, `messages_at`, `active_branch`, `SCHEMA_VERSION`.

### `ExecutionContext`

```python
ExecutionContext(run_id, session_id=None, namespace=None, actor=None, subject=None, attributes={})
```

`run_id` is Semora's execution and idempotency coordinate. **Every other field is opaque to the
framework and must originate at the host trust boundary — never from model output or tool
input.** Pass an `ExecutionContext` anywhere a `run_id: str` is accepted.

### Postgres

```python
from semora_store_pg import PostgresSteps, PostgresTranscript, SCHEMA, TRANSCRIPT_SCHEMA

store = PostgresSteps(pool)          # psycopg_pool.AsyncConnectionPool
transcript = PostgresTranscript(pool)
```

Run `SCHEMA` / `TRANSCRIPT_SCHEMA` once (both are `create table if not exists`).

---

## Models

Any LangChain `BaseChatModel` works. Semora ships one client of its own.

```python
from semora import ChatModel                       # re-export of semora_llm.ChatModel
from semora_llm import openrouter, xai

ChatModel(model="gpt-4.1")                                    # OPENAI_API_KEY
ChatModel(model="llama3", base_url="http://localhost:11434/v1")  # any OpenAI-compatible /v1
openrouter("anthropic/claude-sonnet-4")                       # + the attribution headers
xai("grok-4")
```

```python
ChatModel(model, api_key=None, base_url=None, default_headers=None, extra_body=None,
          tools=(), client=None, timeout=None, recover_dsml=False)
```

`bind_tools` / `astream` are the whole planner contract. This class is the OpenAI-compatible
wire only — for native Anthropic/Google APIs install the extra and pass `ChatAnthropic` /
`ChatGoogleGenerativeAI` directly.

`recover_dsml=True` salvages models that leak tool markup into text.
`semora_llm` also exports `DsmlFilter`, `parse_dsml_tool_calls`, `recover_dsml_chunks`,
`strip_dsml` for doing that by hand.

### Fallback across providers

```python
from semora.providers import FallbackChatModel, ModelProvider, classify_provider_error

model = FallbackChatModel([ModelProvider("primary", ChatModel(model="gpt-4.1")),
                           ModelProvider("backup", openrouter("anthropic/claude-sonnet-4"))])
```

Tries providers in order without translating messages or chunks. **Same-provider retries happen
only before a chunk becomes visible** — once streaming has started, an error propagates, because
switching would duplicate visible output.

`classify_provider_error(exc)` → `"rate_limit" | "authentication" | "server" | "network" |
"abort" | "unknown"`.

### Model failure policy

```python
AgentRuntime(
    model_failure_policy=lambda f: ...,   # ModelFailure -> "retry" | "compact" | "fail"
    compact_context=lambda messages, failure: ...,   # -> shortened list[BaseMessage]
)
```

```python
ModelFailure(error_type, error_kind, message, partial, attempt)
# error_kind: rate_limit | context_overflow | max_output_tokens | authentication
#           | server | invalid_request | unknown
```

---

## Feature packs

### Skills — `semora.skills`

```python
from semora.skills import DirectorySkillSource, SkillRegistry, SkillTools

registry = SkillRegistry([DirectorySkillSource("./skills")], catalog_char_budget=8000)
tools = SkillTools(inner_tools, registry)
```

Metadata-first: `SkillMetadata(name, description, revision)` is what the model sees; the body
loads only when the `skill` tool is called. Later sources override earlier ones by name.
`SkillSource` is a two-method protocol (`list`, `load`), so a database or HTTP source is the
same amount of work as a directory. `Skill.context(arguments="")` renders the injected text.

### Subagents — `semora.subagents`

```python
from semora.subagents import Subagents, FactoryAgent, RunnerAgent, HttpAgent

tools = Subagents(base_tools, [FactoryAgent("researcher", "digs", system_prompt, tools=("read",))],
                  factory=make_runner, authority=("read", "grep"), max_depth=5)
```

| kind | how the child runs |
|---|---|
| `FactoryAgent(name, description, system_prompt, tools=(), model="")` | your `factory(agent, authority)` builds the runner |
| `RunnerAgent(name, description, run)` | you supply the async-iterator runner |
| `HttpAgent(name, description, url, headers={}, timeout=300.0)` | `POST {"input": prompt}`; the response body is the result |

`Authority(ceiling, depth)` is the child's tool ceiling: `.attenuate(requested)` narrows,
`.without(blocked)` subtracts. A child cannot exceed its parent — `delegate` is blocked for
children by default (`blocked_tools_for_child=("delegate",)`).

### Background work — `semora.background`

```python
from semora.background import BackgroundTasks, BackgroundResult

registry = BackgroundTasks()
registry.register(task_id, kind, label, asyncio.create_task(...), read_output=None)
registry.list(); registry.status(task_id); registry.cancel(task_id); registry.cancel_all()
unsubscribe = registry.subscribe(listener)
```

Process-local only. Deliver settled work back into a run with
`AgentRuntime.background_sink(run_id)`; `BackgroundResult.as_message()` renders it.

### Plan mode — `semora.plan_mode`

```python
from semora.plan_mode import PlanMode, plan_mode_enter, plan_mode_exit, plan_mode_gate, plan_mode_prompt
from semora.controls import Journal

mode = PlanMode()
controls = ControlPlane(
    pre_tool_use=Permissions(plan_mode_gate(tools, mode, exit_tool="exit_plan")),
    post_tool_use=Journal(plan_mode_enter(mode, enter_tool="enter_plan"),
                            plan_mode_exit(mode, exit_tool="exit_plan")),
)
```

`plan_mode_gate` denies every non-read-only call while planning. `plan_mode_prompt(mode,
exit_tool=...)` is a volatile system-prompt section that appears only while planning.

### Goals — `semora.goal`

```python
from semora.goal import Goal, goal_complete, goal_gate

goal = Goal("ship the migration")
controls = ControlPlane(before_finish=FinishPolicy(goal_gate(goal, complete_tool="done")),
                        post_tool_use=Journal(goal_complete(goal, complete_tool="done")))
```

The gate refuses to finish while the goal is open; the writer closes it when `done` succeeds.

### Deferred tools — `semora.tool_search`

```python
from semora.tool_search import DeferredTools

tools = DeferredTools(many_tools, deferred={"rare_a", "rare_b"}, initially_active={"read"})
```

Hides schemas until a `tool_search` call activates them. Cuts prompt size when the tool count is
large.

### Permissions — `semora_permissions`

```python
from semora_permissions import PolicyContext, Rule, escalation_guard, resolve_rules

policy = PolicyContext(rules=[Rule("deny", "Bash", "rm "), Rule("ask", "write"),
                              Rule("allow", "read")],
                       mode="default", version="2026-08-30")
controls = ControlPlane(pre_tool_use=Permissions(policy.stage(tools)),
                        on_resume=policy.resume_stage(tools))
```

`Rule(effect: "allow"|"deny"|"ask", tool, content=None)` — `content` is an input prefix, `None`
matches every call to the tool. `mode`: `"default"` (unresolved → ask), `"bypass"`,
`"dont_ask"`. `version` flows into `ResumeInput.current_rules_version`, which is how a resumed
approval notices the rules moved. `escalation_guard(ceiling)` denies tools outside a delegation
ceiling.

### Fork — `semora_fork`

```python
from semora_fork import fork_run, fork_event, EventCheckpoint, ForkCoordinate

await fork_run(runtime, store, from_run_id="run-7", origin_id=input_id, run_id=new_run_id(),
               model=model, tools=tools, controls=alternative_controls)
```

Re-runs a conversation from just before one input entered model context, enqueuing the source
ledger's **pre-screen original** so it crosses whatever `controls` the fork supplies.
`fork_event` does the same from the durable coordinate on an observation edge
(`record_event_checkpoint` / `read_event_checkpoint`). **The source run's ledger is never
touched** — what actually went out stays the record. `semora-fork` adds no authority of its own;
it is composition over seams the core already has.

A coordinate can also say what a branch taken there will do, before it is taken:

```python
from semora_fork import RERUNS, resume_point

point = resume_point(messages_at(entries, coordinate.leaf_uuid), coordinate)
# "on_inputs"     — an input coordinate: the run starts over at that prompt
# "pre_tool_use"  — a leaf that still owes a tool answer: that round's gate decides again,
#                   the effect replays from its ledger record, the journal sees it again
# "post_tool_use" — the same leaf taken with rejournal=True: no gate, the effect is the
#                   source run's record, only the journal runs
# "before_model"  — any other leaf: the recorded round is final, the model continues
RERUNS[point]     # the control points that run again over the recorded round, in loop order

await fork_event(..., rejournal=True)   # re-journal: the gate is skipped, the source run's
                                        # finished records stand in, its ledger is only read
```

`resume_point` uses the predicate recovery uses (`unanswered_tool_calls`), so it is the
decision the fork will make, stated early. A host that lets a person pick a branch point
for a policy can pair it with each control's point and say, per button, which of the
selected policies that branch actually reaches — a journal unit does nothing from a
result coordinate, a finish policy does; neither has to be guessed from its name.

### Events — `semora.contracts.events`

```python
from semora.contracts.events import EventEnvelope, EventStream, EventType, RuntimeEvents

AgentRuntime(event_sink=my_async_sink)      # receives EventEnvelope
```

```python
EventEnvelope(event_type, session_id, thread_id, run_id, sequence, payload,
              turn_id=None, event_id="")
```

Delivery is **best-effort**: a sink failure is logged and never retried. `event_id` is the
stable identity to deduplicate on if you need an outbox. `RuntimeEvents` (on
`AgentRuntime.events`) publishes host lifecycle: `session_start`, `session_end`, `pre_compact`,
`post_compact`, `notification`, `subagent_start`, `subagent_stop`, `task_created`,
`task_completed`, `cwd_changed`, `config_change`, `elicitation`, …

### The bare loop — `semora.engines.plain`

```python
from semora.engines.plain import react_loop

async for event in react_loop(model, tools, system_prompt=..., history=..., controls=...):
    ...
```

The planner under `AgentRuntime`, as an async iterator of events. Use it only when you need to
own the drive loop; `AgentRuntime` is the supported entry point. The loop imposes **no iteration
cap, no timeout and no permission policy of its own** — those are the caller's, supplied as
hooks.

---

## Workspaces

File and process effects resolve through a `WorkspaceProvider`. Internals live on
`semora.workspace` / `semora.sandbox_remote`, not the top level.

```python
from semora import HostWorkspaceProvider

runtime = AgentRuntime(workspace_provider=HostWorkspaceProvider(root="./sandbox", per_run=True))
```

```python
HostWorkspaceProvider(*, root=None, base_dir=None, per_run=False,
                      mode: Literal["read-only","workspace-write","danger-full-access"] = "workspace-write",
                      cleanup: Literal["keep","delete"] | None = None, snapshot_backend=None)
```

**`HostWorkspaceProvider` does not claim network, syscall, or host-filesystem isolation.** It is
a path boundary, not a sandbox. Use `semora.sandbox_remote` for a real one.

`ContinuousWorkspaceProvider(inner, store, conversation_id)` keeps one workspace across
serialized turns: load prior state → resume → acquire fresh on failure → save reconnect state
before cleanup. State-store failures are best-effort by design, so observability never becomes
an availability problem. `MemoryWorkspaceStateStore` is the in-process implementation.

Also on the module: `ToolContext`, `WorkspaceFS`, `WorkspaceSession`, `WorkspaceSeed`,
`WorkspaceAccessMode`, `WorkspaceDirEntry`, `WorkspaceFileStat`, `WorkspaceSnapshot`,
`WorkspaceStateStore`, `WorkspaceViolation`, `SandboxCommand`, `CommandResult`,
`ResolvedWorkspacePath`, `SandboxSessionState`, `SnapshotBackend`, `TarSnapshotBackend`.

---

## Pitfalls

1. **`Agent` + `tools=` is a `TypeError`.** With an `Agent`, the third positional argument is
   the prompt: `runtime.run(run_id, agent, "prompt")`.
2. **No store means no suspension and no recovery.** A `Suspend` with nowhere to park fails, and
   a crash may re-run a tool. `AgentRuntime(execution_store=MemorySteps())` is the floor.
3. **`dispatch()` needs `transcript=` too**, not just a store — state comes from the store,
   history from the transcript.
4. **Do not catch broadly around `run`/`resume`/`recover`.** `ControlSignal` (so, every
   `Suspended`) and `Contended`/`Fenced`/`Indeterminate` are runtime signals, not errors. A tool
   raising is already converted to that call's `error` result at one boundary
   (`_execute_validated` in `tools.py`).
5. **`AgentSuspended` must be bound inside its `except` block** — Python unbinds the name after
   it: `pending_id = parked.pending_id` before the block ends.
6. **A batch is sequential unless every call declares `is_concurrency_safe`.** Order matters for
   writes even when each write is idempotent.
7. **A human's stored approval is an input, not a decision.** `on_resume` re-decides under
   today's rules; that is the point of `rules_version` / `ResumeInput`.
8. **`Bash` is inert until `ExecToolOptions.allow_list` is set**, and `web_search` is not among
   the builtins.
9. **`ExecutionContext` fields other than `run_id` must come from the host trust boundary** —
   never from model output or tool arguments.
10. **`interrupted` state is ambiguous by design.** It names both a crashed round and a run
    another worker is driving; only a lease attempt (`Contended`) distinguishes them.
11. **`gate()`'s fall-through is `Deny`.** Return `None` to allow; a dict that is neither
    `allow` nor `suspend` refuses the call.
12. **`before_finish` inverts `Proceed`/`Halt`.** `Halt(reason)` lets the ending stand;
    `Proceed([...])` vetoes it and sends the loop around. An unconditional `Proceed` never
    terminates — give the gate a condition that eventually stops objecting.
13. **A workflow body re-runs top to bottom on every attempt.** Only `orchestrator.run(step, fn)`
    is replay-protected. An effect outside a step, or a `step` name that changes between
    attempts, fires again.
14. **Resuming a workflow means re-calling the function with the same `run_id`.** A fresh
    `Orchestrator(run_id, log)` over the same log is the resume; there is no `resume()` on it.

---

## Where to look next

| question | file |
|---|---|
| runnable end-to-end examples | [`examples/`](../examples/) — `01_minimal` … `06_bare_loop`, `vs_langgraph` |
| the behavioural contract, pinned | `tests/test_loop.py`, `tests/test_dispatch.py`, `tests/test_resume.py` |
| why a rule exists | [`AGENTS.md`](../AGENTS.md) |
| what changed | [`CHANGELOG.md`](../CHANGELOG.md) |

A test is the behavioural reference. If this document and a test disagree, the test is right —
please fix this document.
