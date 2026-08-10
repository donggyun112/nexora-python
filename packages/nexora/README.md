# nexora

Durable multi-agent runtime for Python. This is the core: the contracts every layer agrees on, the
control points, tool execution, the durable orchestrator, the plain `async while` planner, and the
`AgentRuntime` facade over them.

```python
from nexora import AgentRuntime
from nexora_store import MemoryTranscript

runtime = AgentRuntime(store=steps, transcript=MemoryTranscript(), emit=events)
outcome = await runtime.run("run-42", model, tools, "inspect this repository")
```

Every tool effect crosses the orchestrator's durable boundary before it runs, keyed by the model's
own `call_id`, so a run that dies mid-round is reconstructed from its transcript without replaying
the model turn and a permission gate can park a run for days without holding a worker. Omit
`transcript=` to keep transcript persistence caller-owned.

Install what you need beside it:

| | |
|---|---|
| `nexora` | the runtime, with an in-memory `StepLog` |
| `nexora[postgres]` | a Postgres-backed ledger (`nexora_store_pg`) |
| `nexora[permissions]` | a Claude-Code-shaped permission rule table (`nexora_permissions`) |
| `nexora[ui]` | a local console for driving runs (`nexora_ui`) |

`nexora-store` — the ledger/transcript protocols and their in-memory implementations — is a
separate distribution with no dependencies of its own, so implementing either store needs neither
a message type nor this package.

See the [repository README](../../README.md) for the architecture map and the runnable examples.
