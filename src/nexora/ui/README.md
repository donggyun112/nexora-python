# Nexora Durable Agent Lab

The local UI is an optional module inside the Nexora package. Its dependency direction is:

```text
app → api → execution → AgentRuntime
          ↘ provider
          ↘ tools / state
```

Run it with:

```bash
uv sync --extra ui
uv run --extra ui uvicorn nexora.ui.app:app --reload --port 8790
```

Open <http://127.0.0.1:8790>. Configuration is loaded from `src/nexora/ui/.env`. Supported key
names are `OPENROUTER_API_KEY`, `OPENROUTER_KEY`, and the imported fixture spelling
`OPEN_ROTURE`.

The two suspension samples exercise different boundaries:

* **Tool waits** executes `request_approval`; that tool returns a suspension and waits for its
  eventual result.
* **Pre-tool gate** asks permission before `remember_note` crosses the effect boundary. No
  `post_tool_use` or `post_tool_batch` event exists until approval actually executes the tool.
