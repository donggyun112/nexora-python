# Nexora for Python

Durable multi-agent runtime for Python.

> **Status:** pre-alpha. The TypeScript Nexora implementation remains the behavioral
> reference while the runtime is ported one contract at a time.

## Direction

Nexora is an **agent runtime with durable execution**, not a general-purpose workflow engine.
Its public API speaks in agent concepts — models, tools, permissions, sessions and subagents.
The orchestrator is an internal execution substrate that records and recovers agent effects; it
is not a second product users have to assemble beside the SDK.

See [examples/](examples/) for runnable scripts — a tool round, permission suspend/resume, crash
recovery, control-plane injection, a durable workflow, and the loop without a ledger. They need
no API key.

```python
from nexora import AgentRuntime

runtime = AgentRuntime(store=steps, emit=events)
outcome = await runtime.run("run-42", model, tools, "inspect this repository")
```

The runtime has one execution path. Its agent planner is an ordinary `async while`; every tool
effect crosses Nexora's durable `Orchestrator` before it runs:

```python
runtime = AgentRuntime(
    store=steps,                 # effect ledger + durable input queue
    emit=events,
)
```

Every incremental model input uses that queue. `run(..., prompt)` is convenience syntax for
enqueueing a `user_prompt` and then driving the run; asynchronous producers use the same boundary:

```python
from nexora import PendingInput
from langchain_core.messages import HumanMessage

await runtime.submit(
    "run-42",
    PendingInput("user_steer", HumanMessage("focus on auth"), "prompt-2"),
)
outcome = await runtime.run("run-42", model, tools)
```

Submitting emits the source event (`USER_PROMPT_SUBMIT` for user input). Admission into the model
transcript separately emits `CONTEXT_INJECTED`. Reuse a stable `origin_id`/`prompt_id` when retrying
the same submission.

An interactive input arriving during permission wait uses **cancel-and-switch**. The model's tool
request is already part of its `AIMessage`, so the runtime first admits an error `ToolMessage`
(`code=cancelled`) for every unanswered request and only then admits the newer `HumanMessage`.
The tool never executes. `input_mode="headless"` keeps the input queued until the suspension is
resolved instead. Answers are routed by the suspension's `pending_id`; run forking is not part of
this mechanism.

The loop keeps only planner control flow — model, delegated effect round, stop. Policy, durable
intent, effect ordering, suspension and recovery live at the mediated execution boundary. There
is no alternate graph engine and no graph checkpointer.

Model invocation crosses that durable effect boundary too. Each request is keyed by a stable
fingerprint of the model identity, bound tool definitions and model-visible messages; completed
stream chunks are recorded in `StepLog` and replayed without calling the provider again. A request
left `running` is indeterminate rather than silently charged twice. Standard LangChain models use
their identifying parameters automatically; pass `model_identity=` for a custom wrapper whose
identity cannot be derived from those parameters.

Pass a transcript store to make conversation persistence and recovery automatic. The runtime
records the exact LangChain messages admitted to model history, restores them on the next process,
and reconstructs an unanswered durable tool round before calling the model again:

```python
from nexora import AgentRuntime
from nexora_store import MemoryTranscript

runtime = AgentRuntime(store=steps, transcript=MemoryTranscript())
```

Use `PostgresTranscript` from `nexora_store_pg` for restart-safe persistence. Without
`transcript=`, callers may still provide `history=` or use `recover(...)` explicitly.
`run_id` identifies one durable execution attempt; pass a separate `conversation_id` when several
attempts belong to the same transcript. Permission continuations then persist only the transcript
cursor and the external facts that can change while parked, rather than another message snapshot:

```python
await runtime.run(
    "attempt-42", model, tools, "continue",
    conversation_id="conversation-7",
)
```

Transient model recovery is opt-in and bounded at the orchestration boundary. The built-in policy
retries rate limits and server failures, asks a caller-supplied compactor to handle context
overflow, and fails authentication, invalid requests, unknown failures, or any request that
already streamed visible text:

```python
from nexora import AgentRuntime, ModelFailurePolicy

async def compact_context(messages, failure):
    return await summarize_for_model(messages)

runtime = AgentRuntime(
    model_failure_policy=ModelFailurePolicy(max_retries=2, max_compactions=1),
    compact_context=compact_context,
)
```

No policy means no automatic recovery. Supply `ModelFailurePolicy.backoff` when a retry must wait
for provider or scheduler timing; the runtime deliberately does not invent a delay.

For ordered provider selection, wrap already-constructed LangChain chat models. The wrapper binds
the same tools to every candidate and never changes providers after a chunk has become visible:

```python
from nexora import FallbackChatModel, ModelProvider

model = FallbackChatModel([
    ModelProvider("primary", primary_model),
    ModelProvider("secondary", secondary_model),
])
```

Workspace execution is exposed through `WorkspaceProvider`/`WorkspaceSession`.
`HostWorkspaceProvider` confines resolved paths, supports read-only/workspace-write modes,
shell-free trusted commands, and optional durable tar snapshots. It reports `isolated=False` and
fails closed when a command requires OS isolation or an egress allowlist. Untrusted commands use a
container, mount-namespace, or remote provider implementing the same contract.

The included `RemoteSandboxClient` speaks the same provider-neutral HTTP wire as the TypeScript
sandbox server: remote exec and filesystem operations, live-session reattach, tar hydrate for cold
recovery, optional fixed roots, manifests, and seed directories. Wrap it with
`ContinuousWorkspaceProvider` to retain one workspace across conversation turns:

```python
import os

from nexora import (
    AgentRuntime,
    ContinuousWorkspaceProvider,
    MemoryWorkspaceStateStore,
    RemoteSandboxClient,
)

remote = RemoteSandboxClient(
    "https://sandbox.example.com",
    token=os.environ["SANDBOX_TOKEN"],
)
workspaces = ContinuousWorkspaceProvider(
    remote,
    MemoryWorkspaceStateStore(),  # replace with a durable implementation in production
    "conversation-7",
)
runtime = AgentRuntime(workspace_provider=workspaces)
```

When a provider is configured, tools must implement `ContextualTools`. The runtime acquires the
session before model/tool execution, replaces the tool context's `workdir` and `workspace`, and
always calls session cleanup after completion, suspension, cancellation, or failure. A tool set
that cannot accept this context is rejected before execution; there is no unjailed fallback.

Messages, tool calls and chat models are LangChain's — owning our own versions of those bought
translation layers and little else.

What Nexora owns is everything the loop decides and everything around it: the control-flow
contract (hooks, event vocabulary, and the semantics ported from the TypeScript reference),
tool execution and ordering, permission policy, authority attenuation, sandboxed workspaces,
delegation, handraise, tenancy, transport, stores, and observability.

That is the boundary, not the changelog. What of it this port has actually built is the list
further down.

Internally, those responsibilities split cleanly:

- **Agent planner:** decides the next agent transition in a plain `async while`.
- **Durable orchestrator:** records pending effects, runs gates, dispatches effects, suspends
  without a worker, and reconstructs interrupted tool rounds.
- **Effect ledger:** owns pending/running/done, call-id idempotency and fencing. It is the recovery
  source of truth rather than a second durability mechanism beside a graph checkpointer.
- **Input ledger:** owns pending/claimed/admitted input order. Initial prompts, steers, background
  results and resume answers enter the planner through one queue contract.
- **Effect executors:** perform mediated tool operations, and delegation composes over them
  as a `Tools` wrapper. Workspace sessions provide the filesystem/process boundary; stronger OS
  sandbox providers replace the explicit best-effort host implementation.
  Streamed model invocation is a durable step beside those tool effects.
- **Events:** expose the same lifecycle to audit, UI and monitoring consumers.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run mypy
uv run pytest
```

`uv sync` alone is enough: the dev group pulls both extras, because mypy needs FastAPI's types to
check `ui/` and the Postgres store test skips itself without psycopg.

### Local OpenRouter test UI

```bash
uv run uvicorn nexora_ui.app:app --reload --port 8790
```

Then open <http://127.0.0.1:8790>. See `packages/nexora-ui/README.md` for the chat, tool-effect and
suspension/resume scenarios.

## Current scaffold

A uv workspace of five distributions, all under `packages/`. The root builds nothing — none of the
members is the product the others orbit, so the core sits beside them rather than above them.

```text
packages/
├── nexora/                 nexora               the core · deps: langchain-core, loguru, nexora-store
│   └── src/nexora/
│       ├── contracts/      message types, the event vocabulary
│       ├── controls.py     the control points and what composes at each
│       ├── tools.py        tool execution, the policy gate, result rendering
│       ├── history.py      suspension snapshots and the resume codec
│       ├── background.py   detached jobs, and the leash on them
│       ├── delegate.py     subagents, and the tools that reach them
│       ├── orchestrator.py durable rounds, suspension, recovery
│       ├── engines/plain/  the planner as an `async while`
│       ├── driver.py       engine stream → one outcome
│       └── runtime.py      the public AgentRuntime facade
├── nexora-store/           nexora_store         StepLog, MemorySteps · deps: none
├── nexora-store-pg/        nexora_store_pg      Postgres StepLog      · nexora[postgres]
├── nexora-permissions/     nexora_permissions   the rule table        · nexora[permissions]
└── nexora-ui/              nexora_ui            local console         · nexora[ui]
```

What is split out is what has its own dependency footprint or its own audience — the line the Python
ecosystem draws for `langchain-openai`, `apache-airflow-providers-*`, `opentelemetry-exporter-*`.
Layers of the core share one footprint, so they stay subpackages the way `django.db` stays inside
`django`, and `tests/test_packaging.py` keeps them layered instead: `contracts` may reach nothing,
and no layer may import above its own.

`nexora-store` having no dependencies at all is the point of it existing — a `StepLog` stores opaque
values under opaque keys, so implementing one needs neither a message type nor `nexora` itself.
`nexora-permissions` is optional because nothing in the core imports it.

The runtime covers the reference's stop conditions, exclusive and terminating tools, steering,
permission suspension/resume, durable tool effects, interrupted-round reconstruction, and
per-call tool failure — a tool that raises is reported to the model as an error result, the way
`tool-executor.ts` catches per call, rather than ending the round.

**Subagents** composes over a host's tools the way durability does, and adds `delegate` plus the
four tools that hold the leash on what it launches:

```python
from nexora import AgentRuntime, Compiled, Subagents

runtime = AgentRuntime()
tools = Subagents(
    my_tools,
    [Compiled("researcher", "digs through papers", researcher_runner)],
    run_id="run-42",
    deliver=runtime.background_sink("run-42"),
)
```

A child is driven under a run id derived from the call that asked for it, `f"{run_id}:{call_id}"`,
and that is what keeps `delegate` inside the contract every other tool is held to. A tool call's
id is its idempotency key, so recovery may retry an interrupted call — but a subagent re-run from
nothing does not repeat one write, it repeats every model round and every effect together. Given
the same name on the same store, the child's own effects are in the ledger under it and the retry
becomes the child's own recovery. A runner that ignores the id it is handed gives up exactly this;
a `Remote` child cannot have it at all, since what is behind the POST owns its own durability.

A subtask is handed over one of three ways, and they are different shapes rather than three speeds
of one thing:

* **`wait="sync"` — the parent stops until it has the answer.** Its round does not end until the
  child replies, so whatever the parent decides next was decided knowing the result.
* **`wait="async"` — the parent gives the task away and carries on.** The child answers *when
  it decides to*, by calling `respond_to_parent`, and that answer re-enters the parent's run
  through the durable input queue on a later round.
* **`wait="none"` — the parent opens an independent agent and gets its address.** Not a handoff
  whose answer is thrown away: the thing on the other side owns its own outcome, so the call
  returns that agent's run id and nothing else — no result, and no `cancel_task`. The id is what
  makes it reachable again, by the parent or by a person, through `submit`/`run`/`resume` on that
  run. In this port it is still an `asyncio` task in the parent's process, so it is independent of
  the parent *run* and not of the parent *process*: a host that restarts continues it by driving
  that run id again.

A handed-off child therefore has to be *able* to answer, which is what `Answering` is for — compose
it into the child's own tools and it gains `respond_to_parent`, marked `terminates_loop` because
an agent that has answered is finished and a second answer would race the first:

```python
def build_child(spec, reply):
    return lambda prompt, _reply, _run_id: react_loop(model, Answering(child_tools, reply), ...)
```

A child that never calls it still answers, from its last turn, so handing work to an agent that
lacks the tool loses nothing. (`handoff` is accepted as a synonym for `async`; the reference spells
the mode `async` and uses "handoff" for the `delegate` hop itself, as opposed to `publish_topic`'s
anonymous broadcast.) `tasks=[...]` fans
several children out inside one call, so parallelism is the caller's decision rather than a hope
that the model emits several tool calls at once. The three subagent kinds are the reference's —
`Declarative` (a spec a host factory builds at call time), `Compiled` (a child already wired), and
`Remote` (an HTTP endpoint). `SUBAGENT_START`/`SUBAGENT_STOP` are published around every child.

The reference needs two delivery paths, `ctx.steerSelf` for a result beating the turn's end and
`ctx.deliverResult` for one that misses it. Here both are the durable input queue, which the
planner drains at every round boundary and which keeps whatever arrives after the last one — so
`background_sink` is the whole of it, and a result is folded in or waits for the next `run`
without the caller knowing which happened. `check_tasks`, `read_task_output`, `cancel_task` and
`watch_task` are the leash; `BackgroundTasks` holds it and is deliberately not durable, because a
record of a job surviving a crash its coroutine did not would describe something no longer running.

What delegation here does *not* have is the reference's other half: `capability`-based routing to
a peer over `transport`, its `AgentRegistry`, and `approvalGate`. Those
depend on subsystems this port has not built — there is no transport to publish an envelope to and
no registry to resolve a capability against — so a child is reached by name, in-process.
Attachments are not ported. Context compaction is a seam rather than a feature: the loop calls a
caller-supplied `compact_context` when the provider reports context overflow, and ships no
compactor of its own.

The former graph path remains removed: model and tool effects share the same `StepLog` recovery
authority instead of introducing a second checkpointer.

## License

MIT
