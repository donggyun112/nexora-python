# Nexora for Python

Durable multi-agent runtime for Python.

> **Status:** pre-alpha. The TypeScript Nexora implementation remains the behavioral
> reference while the runtime is ported one contract at a time.

## Direction

Nexora is an **agent runtime with durable execution**, not a general-purpose workflow engine.
Its public API speaks in agent concepts — models, tools, permissions, sessions and subagents.
The orchestrator is an internal execution substrate that records and recovers agent effects; it
is not a second product users have to assemble beside the SDK.

See the [current architecture map](docs/architecture/structure.html) for the ownership, approval,
recovery, and three-lane control/signal/event model.

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

Model invocation is the current exception to that durable effect boundary: the planner streams the
provider directly, and neither the model result nor the growing transcript is checkpointed by
`StepLog`. Until the separate transcript store and a durable model-invocation contract exist, a
caller recovering after that boundary must supply the committed transcript explicitly.

Messages, tool calls and chat models are LangChain's — owning our own versions of those bought
translation layers and little else.

What Nexora owns is everything the loop decides and everything around it: the control-flow
contract (hooks, event vocabulary, and the semantics ported from the TypeScript reference),
tool execution and ordering, permission policy, authority attenuation, sandboxed workspaces,
delegation, handraise, tenancy, transport, stores, and observability.

Internally, those responsibilities split cleanly:

- **Agent planner:** decides the next agent transition in a plain `async while`.
- **Durable orchestrator:** records pending effects, runs gates, dispatches effects, suspends
  without a worker, and reconstructs interrupted tool rounds.
- **Effect ledger:** owns pending/running/done, call-id idempotency and fencing. It is the recovery
  source of truth rather than a second durability mechanism beside a graph checkpointer.
- **Input ledger:** owns pending/claimed/admitted input order. Initial prompts, steers, background
  results and resume answers enter the planner through one queue contract.
- **Effect executors:** perform mediated tool, sandbox and delegation operations. Model invocation
  is still direct planner I/O and is not yet a durable step.
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
uv run uvicorn nexora.ui.app:app --reload --port 8790
```

Then open <http://127.0.0.1:8790>. See `src/nexora/ui/README.md` for the chat, tool-effect and
suspension/resume scenarios.

## Current scaffold

A uv workspace. `nexora` is what you install; the ledger is separate because it can be — it has no
dependencies, so a store implementation never pulls in a model SDK.

```text
src/nexora/                 # the nexora distribution
├── runtime.py              # public AgentRuntime facade
├── orchestrator.py         # durable execution, suspension, recovery
├── contracts/              # what everything agrees on
│   ├── types.py            #   messages, tool calls, hook signatures
│   └── events.py           #   event vocabulary and envelope
├── controls.py             # the control points and what composes at each
├── engines/plain/          # the planner as an `async while`
├── tools.py                # tool execution, the policy gate, result rendering
└── history.py              # suspension snapshots and resume codec

packages/
├── nexora-store/           # nexora_store — StepLog, MemorySteps. dependencies: none
└── nexora-store-pg/        # nexora_store_pg — Postgres StepLog. `nexora[postgres]`
```

`nexora.orchestrator` re-exports the ledger names, so `from nexora.orchestrator import MemorySteps`
keeps working; `import nexora_store` is the direct path for anyone implementing a store.

The runtime covers the reference's stop conditions, exclusive and terminating tools, steering,
permission suspension/resume, durable tool effects, and interrupted-round reconstruction.
Automatic transcript persistence, context compaction, and attachments are not ported yet; crash
recovery currently accepts the durable transcript explicitly rather than hiding that missing store.

Measured locally over 100 zero-I/O rounds (best of three), the direct durable path costs about
260µs per round. ADR-005 records why the former graph path was removed despite equivalent effect
safety.

## License

MIT
