# Semora

An agent runtime that makes a tool's effect happen once, and gives every decision about
that effect a seam. Not a workflow engine, not a coding agent.

Messages, tool calls and chat models are LangChain's. Semora owns the execution boundary:
a call-id ledger with leases and fencing, permission parking that survives the process,
crash recovery that replays what committed and runs what did not, and a control plane of
seven decision points that decide whether a round may start, a tool may run, a result may
enter context, or a finish may stand. Everything an agent *says* — its tools, its prompts,
its plan-mode copy — lives in packages over the core, not in it.

**Status:** 0.2.0 on PyPI, eight distributions pinned to each other. The guarantees below
are implemented and tested, and the examples assert them. Under 1.0 because the public API
still moves without aliases, not because the runtime is unfinished.

**Writing code against it?** [docs/API.md](docs/API.md) is the reference: every public
signature, the outcome and error shapes, the pitfalls. This file is the argument.

## Hello

```bash
uv add semora
```

```python
from semora import Agent, AgentRuntime, ChatModel, new_run_id

agent = Agent(
    name="reviewer",
    description="Reviews a repository with read-only tools",
    model=ChatModel(model="gpt-4.1"),
    tools=tools,
    system_prompt="Inspect evidence before answering.",
)
outcome = await AgentRuntime().run(new_run_id(), agent, "inspect this repository")
```

That default records nothing. The whole control plane is there, but a crash may run the
same tool again and `Suspend` has nowhere to park. Attach a ledger when those guarantees
matter — the agent does not change:

```python
from semora import AgentRuntime, MemorySteps

runtime = AgentRuntime(execution_store=MemorySteps())          # or the Postgres store
outcome = await runtime.run(run_id, agent, "inspect this repository")
```

## What the core promises

**The effect happens once.** Every tool call is a durable step keyed by its call id, taken
under a run lease and written with a fencing token. A second worker on the same run gets
`Contended`; a worker back from the dead gets `Fenced` and its writes are refused. A step
that started and never reported is `Indeterminate`: the ledger will not claim the effect
went out, and will not claim it did not — it says so, and the next move is a person's or
a retry under the same key. `examples/03_recovery.py` kills a worker mid-round; the
committed calls replay from the record and the missing one runs.

**Every decision has a seam.** Policy is seven decision points, not callbacks: `on_inputs`,
`before_model`, `pre_tool_use`, `post_tool_use`, `before_finish`, `on_resume`, `on_suspend`.
A gate answers `Continue`, `Deny` or `Suspend`; a finish policy answers `Proceed` or
`Halt`. `Suspend` parks the run durably — another process, hours later, can answer it —
and `on_resume` re-decides the parked call under the rules in force *then*, so a policy
turned on while a person was deciding still applies. A denial is a result the model sees,
not a tool that ran: nothing after the gate fires for it.

```python
from semora import AgentRuntime, ControlPlane, FinishPolicy, Permissions, gate

await AgentRuntime(execution_store=store).run(
    run_id, agent, "review this change",
    controls=ControlPlane(
        pre_tool_use=Permissions(gate(no_deleting)),
        before_finish=FinishPolicy(require_citation),
    ),
)
```

**A run can be branched, and the branch says what it will re-decide.** With `semora[fork]`
a finished run is a set of durable coordinates: an input, the gate before a call, the
result after it. `resume_point` names the control point a branch taken at each one enters
first — `on_inputs`, `pre_tool_use`, `before_model` — so a host can tell, before branching,
which of the policies it changed will actually run. A branch taken to re-journal a result
resumes at `post_tool_use`: the gate is not asked again, the effect is the record, only the
journal runs. The source run's ledger is read and never written; what actually went out
stays the record.

## Host commands

An HTTP handler, a CLI and a queue worker all ask one question: given this run's durable
state, may this arrive now? `dispatch` answers it once, over the same public primitives.

```python
from semora.dispatch import Answer, Prompt, Recover

await runtime.dispatch(run_id, agent, Prompt("also check the tests"))
```

A `Prompt` aimed at a run another worker is driving is durably enqueued for that worker; a
command the state cannot accept raises `InvalidTransition` carrying the observed state, so
an adapter maps it without string-matching. Behind it is an ordered table of transitions —
drop a row and exactly its behaviour disappears; a host's own row slots into the order.

## What stays yours

The core draws its line at the effect. On the far side of it:

- **The effect's identity beyond one run.** A step is keyed by call id inside a run. A charge
  that must not repeat across *runs* — a retry, a fork, a second operator — is a step of
  something longer-lived than a run. `Orchestrator.run(step, fn)` is the primitive; the key
  is your business's, not the runtime's.
- **Isolation.** `semora.workspace` is the contract the tool boundary runs inside and
  `semora.sandbox_remote` speaks a remote protocol; a host workspace is a reference, not a
  sandbox. A real one lives outside this repository.
- **What the agent is.** Tools, prompts, plan mode, goals, skills, deferred tool search — a
  coding agent's worth of them — ship as `semora-coding`, a reference assembly the core
  does not import. Replace any of it.
- **Policy.** `semora-permissions` is a rule table. The seven points are the contract; what
  they decide is the host's.

## See it run

[semora-console](https://github.com/donggyun112/semora-console) is the operator's view of
all of this against a real model: compose policies across the seven points, watch a worker
die mid-charge and recover with the charge made once, approve a parked send, then branch
the finished run with a policy changed and see exactly which decisions run again.

## Examples

No API key. The model is scripted; swap it for a real one without changing the runtime.

```bash
uv run python examples/01_minimal.py
```

| | shows |
|---|---|
| `01_minimal.py` | default runtime, one tool round, no ledger |
| `02_approval.py` | `pre_tool_use` parks the worker; `on_resume` re-decides under current rules |
| `03_recovery.py` | crash mid-round; committed calls replay, the missing one runs |
| `04_control_plane.py` | `on_inputs` vs a `Tools` wrapper vs `before_finish` — policy lands at a seam |
| `05_workflow.py` | `Orchestrator` outside, an agent run as one durable step |
| `06_bare_loop.py` | the control plane without a ledger, and the one decision that needs one |
| `vs_langgraph.py` | the same mid-batch crash in a graph framework and here; needs `langgraph` |

Each file asserts what it claims. `tests/test_examples.py` runs them.

## Distributions

One workspace, eight wheels. The core does not import the ones above it.

| distribution | import | what |
|---|---|---|
| `semora` | `semora` | contracts, control points, tool boundary, orchestrator, engine, `AgentRuntime`, `dispatch` |
| `semora-store` | `semora_store` | the ledger and transcript protocols with in-memory implementations; no dependencies |
| `semora-store-pg` | `semora_store_pg` | the same on Postgres — `semora[postgres]` |
| `semora-llm` | `semora_llm` | `ChatModel` and provider presets |
| `semora-fork` | `semora_fork` | branch a run at a durable coordinate — `semora[fork]` |
| `semora-permissions` | `semora_permissions` | a permission rule table — `semora[permissions]` |
| `semora-coding` | `semora_coding` | a coding agent's tools, prompts, plan mode, goals, skills — `semora[coding]` |
| `semora-ui` | `semora_ui` | a local console at :8790 — `semora[ui]` |

Provider SDKs are extras and the core imports none of them: `semora[openai]`,
`[anthropic]`, `[google]`, `[xai]`, `[openrouter]`.

## Develop

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run mypy
uv run pytest
```

`tests/test_packaging.py` enforces the two boundaries above: no distribution imports beyond
what its manifest declares, and no layer of the core imports above itself.

## License

MIT
