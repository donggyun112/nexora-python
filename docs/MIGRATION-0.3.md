# Migrating to Semora 0.3

Semora now extends Pydantic AI. The independently developed `pydantic-ai-runtime` port is the source of this successor, not a second maintained product.

## Preserved implementation

The LangChain implementation remains in Git through commit `156e4b1`. Read or check out that revision for 0.2 behavior. Existing 0.2 releases are unaffected. The old loop, provider adapters, coding assembly, UI package and fork package are no longer built by this workspace; only three 0.3 distributions are maintained.

## Application changes

| 0.2 | 0.3 |
|---|---|
| LangChain messages and tool-call dictionaries | Pydantic AI `ModelRequest`, `ModelResponse`, `ToolCallPart` |
| `semora.Agent(model=..., tools=...)` | Plain `pydantic_ai.Agent(...)`, or a `semora.Agent` subclass |
| Own loop and provider facade | Pydantic AI loop and providers |
| `runtime.run(run_id, model, tools, prompt)` | `runtime.run(run_id, agent, prompt)` |
| Outcome dictionaries | `Outcome.output`, `.stop_reason`, `.all_messages()` |
| `runtime.recover(run_id, history, model, tools)` | `runtime.recover(run_id, agent, history)` |
| Default `retry_running=True` | Default `retry_running=False` |
| `semora-llm`, coding/UI/fork extras | Native Pydantic AI capabilities and application-level console adapters |

For the class interface, `llm`, `prompt`, `uses` and `@tool` configure the agent. An instance represents one run; bind a fresh instance to the same `run_id` and durable stores after restart. Instance fields themselves are not persisted automatically.

The seven control-point names remain, but they receive native Pydantic AI values. Policies must use `call.tool_name`, `call.tool_call_id`, and `call.args_as_dict()`, not legacy dictionary indexing. `PendingInput.part` replaces the old `message` field and holds a native Pydantic AI request part.

`AgentRuntime` raises `AgentSuspended`; the optional class-agent interface represents suspension as `Outcome.suspended` and `Outcome.pending`. Do not confuse the two entry points.

## Durable-data boundary

No automatic conversion of existing LangChain transcripts or in-flight 0.2 runs is provided. Finish them with 0.2 or keep them read-only. Start 0.3 runs with fresh run/conversation identifiers and a separate deployment/database namespace. New transcripts identify their message schema as `pai-v1`; a schema label is not an old-data converter.

Keep host-owned business idempotency keys across retries and forks. Do not infer cross-run deduplication from the presence of `MemorySteps` or a per-run tool ledger. Approval policy versions are host-supplied labels, not automatically generated policy fingerprints.

## Intentional limits

The original port's parity percentages were manual classifications, not a compatibility test score. Semora 0.3 does not claim drop-in 0.2 compatibility. Cancel-and-switch, generic control-point forks and deterministic child-run recovery were not completed by that port; the console's adaptation must demonstrate its own supported scenarios.
