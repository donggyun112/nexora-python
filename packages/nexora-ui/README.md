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
uv run uvicorn nexora_ui.app:app --reload --port 8790
```

Open <http://127.0.0.1:8790>. Configuration is loaded from `packages/nexora-ui/.env`. Supported key
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

## Subagents

The `SUBAGENTS` row of sample prompts drives every delegation shape the runtime has, against three
demo children — `note-keeper`, `echoer`, and `flaky`. They are real runs, not canned strings: each
one goes through `AgentRuntime` on the run id `Subagents` derived for it, on the same ledger the
parent uses, with `respond_to_parent` in reach.

| Button | What it shows |
| --- | --- |
| Sync | The parent's round waits. The child's own stream renders beside it, then its answer. |
| Handoff | The parent answers immediately; the child appears on the subagent rail as `running` and its answer arrives on a later round. |
| Fan-out | Two children in one call, both answers in one result. |
| Open independent | `wait="none"` — no leash, just a run id, listed under INDEPENDENT with a **Talk to it** button. |
| check_tasks | What the model itself sees of the children it launched. |
| Child failure | A child reporting a failure, distinct from a child that crashed. |

The rail keeps the two relationships apart because they are different: what is `ON THE PARENT'S
LEASH` can be cancelled from here and by the model's `cancel_task`; what is `INDEPENDENT` has an
address and nothing else, which is the entire result of opening one.

**Talk to it** spends that address — the same durable queue a steer crosses, aimed at the child's
run — and the agent remembers its earlier turns. That continuity is the console's, not the
runtime's: `AgentRuntime` persists no transcript yet, so `recording.py` keeps what the console
watched and hands it back as explicit `history`, which is the path the core README documents. It
ends where this process does, and `GET /api/transcript/{run_id}` shows exactly what was kept.

One thing to know when testing it: the demo system prompt tells the agent to use a tool whenever
asked to *recall a note*, so "what did I tell you earlier?" makes it call `recall_note`, find
nothing, and answer that it does not remember — with the whole conversation sitting in its context.
Ask without that word and it answers from the transcript.
