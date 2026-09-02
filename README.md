# Semora

An agent runtime that makes tool effects happen once. Not a workflow engine.

Messages, tool calls, and chat models are LangChain's. Semora owns the execution boundary:
call-id idempotency, permission parking, crash recovery, and the control plane that decides
whether a round may start, a tool may run, or a finish may stand.

**Status:** 0.2.0. The guarantees below are implemented and tested; the
version stays under 1.0 because the public API is still young enough to move, not because the
runtime is unfinished. The default install includes `ChatModel`, an OpenAI-compatible client.
Native Anthropic/Google APIs remain extras.

**Writing code against Semora?** [docs/API.md](docs/API.md) is the reference: every public
signature, the outcome and error shapes, and the pitfalls, with no prose you have to read first.

## Hello

```python
from semora import Agent, AgentRuntime, ChatModel, new_run_id

model = ChatModel(model="gpt-4.1")
agent = Agent(
    name="reviewer",
    description="Reviews a repository with read-only tools",
    model=model,
    tools=tools,
    system_prompt="Inspect evidence before answering.",
)

run_id = new_run_id()
outcome = await AgentRuntime().run(run_id, agent, "inspect this repository")
```

```bash
uv add semora
# OpenRouter / xAI: ChatModel(..., base_url="https://openrouter.ai/api/v1")
# or from semora_llm import openrouter; openrouter("anthropic/claude-sonnet-4")
```

That default records nothing. The whole control plane is there; a crash may run the same
tool again, and `Suspend` has nowhere to park. Attach a ledger when those guarantees matter:

```python
from semora import AgentRuntime, MemorySteps

runtime = AgentRuntime(execution_store=MemorySteps())
outcome = await runtime.run(run_id, agent, "inspect this repository")
```

`execution_store=` is shorthand for `orchestrator=DurableRuntimeOrchestrator(store)` from
`semora.orchestration`; `store=` remains accepted. The agent does not change.

## Control plane

Policy is seven decision points, not callbacks: `on_inputs`, `before_model`, `pre_tool_use`,
`post_tool_use`, `before_finish`, `on_resume`, `on_suspend`. `before_finish` can refuse an
ending (`Proceed([...])`) and send the loop around again.

```python
from semora import AgentRuntime, ControlPlane, FinishPolicy, Halt, Permissions, gate

runtime = AgentRuntime()
await runtime.run(
    run_id,
    model,
    tools,
    "review this change",
    controls=ControlPlane(
        pre_tool_use=Permissions(gate(no_deleting)),
        before_finish=FinishPolicy(require_citation),
    ),
)
```

## Host commands

An HTTP handler, a CLI, and a queue worker all ask one question: given this run's durable
state, may this arrive now? `dispatch` answers it once, over the same public primitives
(`run`, `resume`, `recover`, `submit`).

```python
from semora.dispatch import Answer, Prompt, Recover
from semora_store import MemoryTranscript

runtime = AgentRuntime(execution_store=MemorySteps(), transcript=MemoryTranscript())
await runtime.dispatch(run_id, agent, Prompt("also check the tests"))
```

A `Prompt` aimed at a run another worker is driving is durably enqueued for that worker
(`{"type": "enqueued", ...}`); a command the state cannot accept raises `InvalidTransition`
carrying the observed state, so an adapter maps it without string-matching. Behind it is
`default_router()`, an ordered table of rows: drop one and exactly its behaviour disappears,
and a host's own `Transition` slots into the order without subclassing.

## Examples

No API key. The model is scripted; swap it for a real chat model without changing the runtime.

```bash
uv run python examples/01_minimal.py
```

| | shows |
|---|---|
| `01_minimal.py` | default runtime, one tool round, no ledger |
| `02_approval.py` | `pre_tool_use` parks the worker; `on_resume` re-decides under current rules |
| `03_recovery.py` | crash mid-round; committed calls replay, the missing one runs |
| `04_control_plane.py` | `on_inputs` vs a `Tools` wrapper vs `before_finish` |
| `05_workflow.py` | `Orchestrator` outside, an agent run as one durable step |
| `06_bare_loop.py` | the control plane without a ledger, and the one decision that needs one |
| `vs_langgraph.py` | the same mid-batch crash in a graph framework and here; needs `langgraph` |

Each file asserts what it claims. `tests/test_examples.py` runs them.

## Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run mypy
uv run pytest
```

| extra | what |
|---|---|
| `semora` | runtime + in-memory `MemorySteps` |
| `semora[openai]` | `langchain-openai` (`ChatOpenAI`) |
| `semora[anthropic]` | `langchain-anthropic` (`ChatAnthropic`) |
| `semora[google]` | `langchain-google-genai` |
| `semora[xai]` | `langchain-xai` (`ChatXAI`) |
| `semora[openrouter]` | same adapter as `openai`; point `base_url` at OpenRouter |
| `semora[postgres]` | Postgres ledger |
| `semora[permissions]` | rule table |
| `semora[fork]` | branch a run from before one injected input |
| `semora[ui]` | local OpenRouter console at :8790 |

`semora-store` has no dependencies of its own. A `StepLog` stores opaque values under opaque
keys, so implementing one needs neither a message type nor `semora`.

`semora-fork` adds no authority of its own — it is composition over seams the core already
has. `fork_run` re-runs a conversation from just before one input entered model context,
enqueuing the source ledger's pre-screen original so it crosses whatever `controls` the fork
supplies; `fork_event` does the same from the durable coordinate recorded on an observation
edge. The source run's ledger is never touched: what actually went out stays the record.

Workspace internals (`ToolContext`, snapshot backends, sandbox HTTP types) live on
`semora.workspace` and `semora.sandbox_remote`, not the top-level package. Feature packs
(skills, subagents, builtins, plan mode) and power-user seams (`react_loop`,
`DurableRuntimeOrchestrator`) stay on the submodule that owns them.

## License

MIT
