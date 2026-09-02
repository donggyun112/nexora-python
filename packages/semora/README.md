# semora

An agent runtime that makes a tool's effect happen once, and gives every decision about
that effect a seam. Not a workflow engine, not a coding agent.

This is the core: contracts, the seven control points, the tool execution boundary, the
orchestrator with its call-id ledger, the plain `async while` planner, `AgentRuntime` and
`dispatch`. Messages, tool calls and chat models are LangChain's. What an agent *says* —
tools, prompts, plan mode — is not here; `semora-coding` is one such assembly, over this.

```python
from semora import Agent, AgentRuntime, ChatModel, new_run_id

agent = Agent("reviewer", "Reviews repositories", ChatModel(model="gpt-4.1"), tools, system_prompt)
outcome = await AgentRuntime().run(new_run_id(), agent, "inspect this repository")
```

That default records nothing: a crash may run the same tool again and `Suspend` has
nowhere to park. Attach a ledger when those guarantees matter; the agent does not change:

```python
from semora import AgentRuntime, MemorySteps

runtime = AgentRuntime(execution_store=MemorySteps())          # or the Postgres store
outcome = await runtime.run(run_id, agent, "inspect this repository")
```

With a ledger, every tool call is a durable step keyed by its call id under a run lease:
a second worker gets `Contended`, a stale one gets `Fenced`, and a step that started and
never reported is `Indeterminate` — the ledger claims neither that the effect went out
nor that it did not. Policy is seven decision points on `ControlPlane`: `on_inputs`,
`before_model`, `pre_tool_use`, `post_tool_use`, `before_finish`, `on_resume`,
`on_suspend`. `Suspend` parks the run durably and `on_resume` re-decides it under the
rules in force when a person answers.

| extra | what |
|---|---|
| `semora` | the runtime, with in-memory `MemorySteps` |
| `semora[postgres]` | Postgres ledger and transcript (`semora_store_pg`) |
| `semora[fork]` | branch a finished run at a durable coordinate (`semora_fork`) |
| `semora[permissions]` | permission rule table (`semora_permissions`) |
| `semora[coding]` | a coding agent's tools, prompts, plan mode, goals, skills (`semora_coding`) |
| `semora[ui]` | local console at :8790 (`semora_ui`) |
| `semora[openai]` / `[anthropic]` / `[google]` / `[xai]` / `[openrouter]` | provider adapters; the core imports none |

Reference: [docs/API.md](https://github.com/donggyun112/semora/blob/main/docs/API.md).
The argument, with asserting examples:
[github.com/donggyun112/semora](https://github.com/donggyun112/semora).
