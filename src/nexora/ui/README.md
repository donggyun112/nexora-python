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

The **Pre-tool gate** sample asks permission before `remember_note` crosses the effect boundary. No
  `post_tool_use` event exists until approval actually executes the tool.
Tools cannot return a suspension after execution; `suspend` is reserved for `pre_tool_use` policy.

The **Tool failure** sample calls `simulate_api_failure`, which returns a recorded, non-retryable
503 result. The UI must show `post_tool_use_failure` and a failed Tool Result; neither the agent
loop nor the demo tool retries the call automatically.

The **Step recovery** sample arms a worker-only fault immediately after a tool result is committed
to `MemorySteps`, but before the ToolMessage reaches the agent transcript. The UI host remains
alive, and **Recover from committed Step** calls `AgentRuntime.recover`: the ledger result is
reused and the demo tool must execute exactly once (`execution_count: 1` remains visible in the
recovered result). This simulates an execution worker crash, not a full UI server restart; the
latter requires a persistent StepLog and transcript store.

The UI labels the model output **TOOL REQUEST**, not tool execution. If a new chat message arrives
while the sample is waiting, the interactive default cancels the unanswered request, writes its
protocol-closing ToolMessage, and continues the same run with the new message. Approval responses
use `pending_id`; a late approval for the cancelled request is rejected.
