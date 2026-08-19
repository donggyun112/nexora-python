# Nexora for Python

An agent runtime that makes tool effects happen once. Not a workflow engine.

Messages, tool calls, and chat models are LangChain's. Nexora owns the execution boundary:
call-id idempotency, permission parking, crash recovery, and the control plane that decides
whether a round may start, a tool may run, or a finish may stand.

**Status:** pre-alpha 0.1.0. This Python runtime is the product. The default install
includes `ChatModel`, an OpenAI-compatible client. Native Anthropic/Google APIs remain extras.

## Hello

```python
from nexora import Agent, AgentRuntime, ChatModel

model = ChatModel(model="gpt-4.1")
agent = Agent(
    name="reviewer",
    description="Reviews a repository with read-only tools",
    model=model,
    tools=tools,
    system_prompt="Inspect evidence before answering.",
)

outcome = await AgentRuntime().run("run-42", agent, "inspect this repository")
```

```bash
uv add nexora
# OpenRouter / xAI: ChatModel(..., base_url="https://openrouter.ai/api/v1")
# or nexora.openrouter("anthropic/claude-sonnet-4")
```

That default records nothing. The whole control plane is there; a crash may run the same
tool again, and `Suspend` has nowhere to park. Attach a ledger when those guarantees matter:

```python
from nexora import Agent, AgentRuntime, MemorySteps

runtime = AgentRuntime(store=MemorySteps())
outcome = await runtime.run("run-42", agent, "inspect this repository")
```

`store=` is shorthand for `orchestrator=DurableRuntimeOrchestrator(steps)`. The agent does
not change.

## Control plane

Policy is seven decision points, not callbacks: `on_inputs`, `before_model`, `pre_tool_use`,
`after_tool_call`, `before_finish`, `on_resume`, `on_suspend`. `before_finish` can refuse an
ending (`Proceed([...])`) and send the loop around again.

```python
from nexora import AgentRuntime, ControlPlane, FinishPolicy, Halt, Permissions, gate

runtime = AgentRuntime()
await runtime.run(
    "run-42",
    model,
    tools,
    "review this change",
    controls=ControlPlane(
        pre_tool_use=Permissions(gate(no_deleting)),
        before_finish=FinishPolicy(require_citation),
    ),
)
```

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
| `nexora` | runtime + in-memory `MemorySteps` |
| `nexora[openai]` | `langchain-openai` (`ChatOpenAI`) |
| `nexora[anthropic]` | `langchain-anthropic` (`ChatAnthropic`) |
| `nexora[google]` | `langchain-google-genai` |
| `nexora[xai]` | `langchain-xai` (`ChatXAI`) |
| `nexora[openrouter]` | same adapter as `openai`; point `base_url` at OpenRouter |
| `nexora[postgres]` | Postgres ledger |
| `nexora[permissions]` | rule table |
| `nexora[ui]` | local OpenRouter console at :8790 |

`nexora-store` has no dependencies of its own. A `StepLog` stores opaque values under opaque
keys, so implementing one needs neither a message type nor `nexora`.

Workspace internals (`ToolContext`, snapshot backends, sandbox HTTP types) live on
`nexora.workspace` and `nexora.sandbox_remote`, not the top-level package.

## License

MIT
