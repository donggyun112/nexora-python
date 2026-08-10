# nexora

Durable multi-agent runtime for Python. This is the core: the contracts every layer agrees on, the
control points, tool execution, the durable orchestrator, the plain `async while` planner, and the
`AgentRuntime` facade over them.

```python
from nexora import AgentRuntime

runtime = AgentRuntime(store=steps, emit=events)
outcome = await runtime.run("run-42", model, tools, "inspect this repository")
```

Every tool effect crosses the orchestrator's durable boundary before it runs, keyed by the model's
own `call_id`, so a run that dies mid-round can be recovered without replaying the model turn and a
permission gate can park a run for days without holding a worker.

Install what you need beside it:

| | |
|---|---|
| `nexora` | the runtime, with an in-memory `StepLog` |
| `nexora[postgres]` | a Postgres-backed ledger (`nexora_store_pg`) |
| `nexora[permissions]` | a Claude-Code-shaped permission rule table (`nexora_permissions`) |
| `nexora[ui]` | a local console for driving runs (`nexora_ui`) |

`nexora-store` — the ledger protocol and its in-memory implementation — is a separate distribution
with no dependencies of its own, so writing a `StepLog` needs neither a message type nor this
package.

See the [repository README](../../README.md) for the architecture map and the runnable examples.
