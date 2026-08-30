# AGENTS.md

## Project contract

- This repository is the runtime. It began as a port of a TypeScript predecessor, which is
  retired: nothing is kept in sync with it any more, and no behavior needs its permission to
  change.
- `react.ts` citations in docstrings and test names are historical — they record *why* an edge
  case exists, not an authority to consult. The checkout is gone, so never cite it in new work
  and never claim to have read it. A test is the behavioral reference now; if a rule has no
  test, write one rather than appealing to the port.
- **Messages, tool calls and models are LangChain's.** `BaseMessage`, `ToolCall`,
  `BaseChatModel`. Owning our own bought translation layers and a re-implementation of
  `ChatOpenAI`.
- **What is ours is the control-flow contract**: the hooks in `contracts/types.py`, the event
  vocabulary in `contracts/events.py`, and the loop semantics `tests/test_loop.py` pins.
- **One planner, one execution path.** `engines/plain` is the agent planner. Durable policy,
  effect execution, suspension and recovery belong to `Orchestrator`; do not add a second graph
  engine or checkpointer path.
- The loop imposes no iteration cap, no timeout, and no permission policy of its own. Those
  are the caller's, supplied as hooks.
- **There is one approval path.** An agent asking a human and a policy refusing a tool both end
  as a suspension, so neither holds a worker while waiting. Gates must not block synchronously.
- **Gate logic is centralized at the execution boundary.** An agent planner requests an effect;
  `Orchestrator` and the shared controls decide whether it may run.
- Semora owns providers' *selection*, tools, permissions, authority, sandboxing, and transport.
- **Retry safety needs two things, not one**:
  - *per-call idempotency* — a tool call's id is its idempotency key, because a crash between
    executing a tool and recording its result is indistinguishable from never running it;
  - *batch order determinism* — two individually idempotent writes to the same file give
    different results in different orders, so a batch runs sequentially unless every call in
    it is declared concurrency-safe.
- Suspension records carry only what the pause can change. Anything deterministically
  reconstructible from the conversation stays out; external facts go in.
- **A tool that raises failed; the round did not.** The exception becomes that call's `error`
  result, its step commits `done`, and the model is told so it can try something else — one
  boundary, `_execute_validated` in `tools.py`, matching `tool-executor.ts`. A runtime signal is
  not a tool failure and passes through: `ControlSignal` (so, any `Suspended`) and the ledger's
  `Contended`/`Fenced`/`Indeterminate`. Never catch broadly anywhere else — a second catch above
  that boundary converts exactly the signals it deliberately let through.

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
