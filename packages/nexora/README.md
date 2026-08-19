# nexora

Durable agent runtime for Python. This is the core: contracts, control points, tool
execution, the orchestrator, the plain `async while` planner, and `AgentRuntime`.

By default `AgentRuntime` drives the planner directly — no orchestrator, no ledger:

```python
from nexora import Agent, AgentRuntime

agent = Agent("reviewer", "Reviews repositories", model, tools, system_prompt)
outcome = await AgentRuntime().run("attempt-42", agent, "inspect this repository")
```

This path cannot suspend or recover a crashed round. Attach a ledger when those
guarantees are required:

```python
from nexora import AgentRuntime, MemorySteps

runtime = AgentRuntime(store=MemorySteps())
outcome = await runtime.run("attempt-42", agent, "inspect this repository")
```

Policy lives on `Controls` / `ControlPlane` (`on_inputs`, `before_model`, `pre_tool_use`,
`after_tool_call`, `before_finish`, `on_resume`, `on_suspend`). `before_finish` can refuse
an ending and send the loop around again.

`builtin_tools()` supplies `read`, `write`, `edit`, `grep`, `glob`, `Bash`, and `web_fetch`.
`web_search` is not included. `Bash` stays disabled until `ExecToolOptions.allow_list` is
set. File and process effects use the `WorkspaceProvider` injected by `AgentRuntime`.

Install extras beside it:

| | |
|---|---|
| `nexora` | the runtime, with in-memory `MemorySteps` |
| `nexora[openai]` | `langchain-openai` |
| `nexora[anthropic]` | `langchain-anthropic` |
| `nexora[google]` | `langchain-google-genai` |
| `nexora[xai]` | `langchain-xai` |
| `nexora[openrouter]` | OpenAI adapter for OpenRouter |
| `nexora[postgres]` | Postgres ledger (`nexora_store_pg`) |
| `nexora[permissions]` | permission rule table (`nexora_permissions`) |
| `nexora[ui]` | local console (`nexora_ui`) |

See the [repository README](../../README.md) for examples.
