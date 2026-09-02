# semora

Durable agent runtime for Python. This is the core: contracts, control points, tool
execution, the orchestrator, the plain `async while` planner, and `AgentRuntime`.

By default `AgentRuntime` drives the planner directly — no orchestrator, no ledger:

```python
from semora import Agent, AgentRuntime, ChatModel

model = ChatModel(model="gpt-4.1")
agent = Agent("reviewer", "Reviews repositories", model, tools, system_prompt)
outcome = await AgentRuntime().run("attempt-42", agent, "inspect this repository")
```

This path cannot suspend or recover a crashed round. Attach a ledger when those
guarantees are required:

```python
from semora import AgentRuntime, MemorySteps

runtime = AgentRuntime(store=MemorySteps())
outcome = await runtime.run("attempt-42", agent, "inspect this repository")
```

Policy lives on `Controls` / `ControlPlane` (`on_inputs`, `before_model`, `pre_tool_use`,
`post_tool_use`, `before_finish`, `on_resume`, `on_suspend`). `before_finish` can refuse
an ending and send the loop around again.

`semora_coding.builtins.builtin_tools()` supplies `read`, `write`, `edit`, `grep`, `glob`, `Bash`,
and `web_fetch`. `web_search` is not included. `Bash` stays disabled until
`ExecToolOptions.allow_list` is set. File and process effects use the `WorkspaceProvider`
injected by `AgentRuntime` (`semora.workspace`).

Install extras beside it:

| | |
|---|---|
| `semora` | the runtime, with in-memory `MemorySteps` |
| `semora[openai]` | `langchain-openai` |
| `semora[anthropic]` | `langchain-anthropic` |
| `semora[google]` | `langchain-google-genai` |
| `semora[xai]` | `langchain-xai` |
| `semora[openrouter]` | OpenAI adapter for OpenRouter |
| `semora[postgres]` | Postgres ledger (`semora_store_pg`) |
| `semora[permissions]` | permission rule table (`semora_permissions`) |
| `semora[ui]` | local console (`semora_ui`) |

See the [repository README](../../README.md) for examples.
