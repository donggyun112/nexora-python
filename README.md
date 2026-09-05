# Semora

Execution controls and durable effect recovery for **Pydantic AI agents**.

Pydantic AI owns the agent loop, messages, models and tools. Semora adds a call-id ledger, worker leases and fencing, durable approval suspension, policy revalidation on resume, and seven composable control points. Use an ordinary Pydantic AI agent or Semora's optional class-based agent interface.

**Why not durable execution alone?** Durable execution replays steps. It does not say which effect may already have gone out, and it does not make policy decide again when a parked call comes back. Semora handles those two things: a tool call that started and never reported stays `Indeterminate` until the caller says a retry is safe, and a person's approval is an input to a fresh policy decision, never the decision itself. The ledger is a protocol, so a durable-execution backend can sit underneath it.

**0.3 is the Pydantic AI successor to Semora 0.2.** The implementation developed in `contribution/pydantic-ai-runtime` now lives here under the Semora package names. The LangChain implementation is retired and remains in Git at `156e4b1`. 0.3.0 is a breaking release; see [migration](docs/MIGRATION-0.3.md) and [API](docs/API.md).

## Run an agent

From this checkout:

```bash
uv sync --dev
uv run python examples/reviewer.py
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from semora import AgentRuntime, MemorySteps, MemoryTranscript


def reply(messages, info):
    return ModelResponse(parts=[TextPart("Hello from Semora")])


agent = Agent(FunctionModel(reply))
runtime = AgentRuntime(MemorySteps(), transcript=MemoryTranscript())
# Inside an async application:
# outcome = await runtime.run("example-run", agent, "hello")
# print(outcome.output)
```

Provider SDKs are optional. `semora[openai]` enables Pydantic AI's OpenAI-compatible provider support. `semora[postgres]` installs the PostgreSQL adapter. Memory stores are for tests and local experiments: they do not survive a process restart.

## What Semora adds

- **Effect records:** completed tool calls replay their recorded results. A started but unreported effect is `Indeterminate` by default; the runtime does not guess that a retry is safe.
- **Worker coordination:** run leases reject competing workers, and fencing rejects stale ledger writes. External services still require their own idempotency or reconciliation contract.
- **Approval revalidation:** `Suspend` parks the run and releases the worker. `on_resume` receives the human answer and both policy-version labels before deciding whether the effect may execute.
- **Policy composition:** `on_inputs`, `before_model`, `pre_tool_use`, `post_tool_use`, `before_finish`, `on_resume`, `on_suspend`. Gate composition makes denial take precedence over suspension.
- **Transcript and dispatch:** `Prompt`, `Answer` and `Recover` route through durable run state. Native Pydantic AI messages are preserved.

The ledger's tool-call key is scoped to one run. Business operations that must deduplicate across runs or branches need a host-owned stable key. `retry_running=True` is an explicit assertion that retrying is safe. Post-tool hooks are at least once across a crash before their completion marker and must be idempotent if they have external effects.

The [operator console](https://github.com/donggyun112/semora-console) demonstrates policy composition, request-scoped payment deduplication, indeterminate effects, approvals and policy forks. Its payment is a simulated effect; the demonstration does not certify any external payment provider.

## Packages

| Distribution | Import | Responsibility |
|---|---|---|
| `semora` | `semora` | Effects capability, controls, runtime, class agent, transcript and dispatch |
| `semora-store` | `semora_store` | Storage protocols and in-memory implementations; no dependencies |
| `semora-store-pg` | `semora_store_pg` | PostgreSQL adapters |

## Development

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build --all-packages
```

Set `SEMORA_TEST_DSN` to a scratch PostgreSQL database to run the durable store conformance tests. CI supplies PostgreSQL. A local run without it explicitly skips those tests.

MIT license.
