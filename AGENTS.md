# AGENTS.md

## Project contract

- This repository is the Python 3.12+ port of Nexora, intended to replace the TypeScript
  runtime rather than sit beside it.
- Preserve observable TypeScript Nexora behavior. `packages/architectures/src/react.ts` is the
  behavioral reference for the loop; cite the line a rule comes from when porting one.
- **Messages, tool calls and models are LangChain's.** `BaseMessage`, `ToolCall`,
  `BaseChatModel`. Owning our own bought translation layers and a re-implementation of
  `ChatOpenAI`; see [ADR-001](docs/architecture/adrs/adr-001-plain-loop-over-graph-engine.md).
- **What is ours is the control-flow contract**: the hooks in `contracts/types.py`, the event
  vocabulary in `contracts/events.py`, and the react.ts semantics.
- **One planner, one execution path.** `engines/plain` is the agent planner. Durable policy,
  effect execution, suspension and recovery belong to `Orchestrator`; do not add a second graph
  engine or checkpointer path. See ADR-005.
- The loop imposes no iteration cap, no timeout, and no permission policy of its own. Those
  are the caller's, supplied as hooks.
- **There is one approval path.** An agent asking a human and a policy refusing a tool both end
  as a suspension, so neither holds a worker while waiting. Gates must not block synchronously.
- **Gate logic is centralized at the execution boundary.** An agent planner requests an effect;
  `Orchestrator` and the shared controls decide whether it may run.
- Nexora owns providers' *selection*, tools, permissions, authority, sandboxing, and transport.
- **Retry safety needs two things, not one** ([ADR-002](docs/architecture/adrs/adr-002-retry-safety-needs-order-determinism.md)):
  - *per-call idempotency* — a tool call's id is its idempotency key, because a crash between
    executing a tool and recording its result is indistinguishable from never running it;
  - *batch order determinism* — two individually idempotent writes to the same file give
    different results in different orders, so a batch runs sequentially unless every call in
    it is declared concurrency-safe.
- Suspension records carry only what the pause can change. Anything deterministically
  reconstructible from the conversation stays out; external facts go in.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest                     # -m "not perf" to skip the timing-sensitive ones
```

Use `src/` layout, strict typing, and a test for every loop semantic ported from the reference.
Test conventions — one property per test, fakes not mocks, ratios not absolute times — live in
[.claude/skills/writing-tests/SKILL.md](.claude/skills/writing-tests/SKILL.md).
