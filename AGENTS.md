# AGENTS.md

## Project contract

- This repository is the Python 3.12+ port of Nexora, intended to replace the TypeScript
  runtime rather than sit beside it.
- Preserve observable TypeScript Nexora behavior. `packages/architectures/src/react.ts` is the
  behavioral reference for the loop; cite the line a rule comes from when porting one.
- The agent loop is a plain `async while`. Control flow belongs to the language — no graph
  engine, no DSL. Durability is a separate concern layered around the loop, not inside it.
- The loop imposes no iteration cap, no timeout, and no permission policy of its own. Those
  are the caller's, supplied as hooks.
- There is one approval path. An agent asking a human and a policy refusing a tool both end as
  a suspension, so neither holds a worker while waiting. Gates must not block synchronously.
- Nexora owns providers, tools, permissions, authority, sandboxing, transport, and public APIs.
- External side effects must be idempotent. A tool call's id is its idempotency key: a crash
  between executing a tool and recording its result is indistinguishable from never running it.
- Suspension records carry only what the pause can change. Anything deterministically
  reconstructible from the conversation stays out; external facts go in.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest
```

Use `src/` layout, strict typing, and a test for every loop semantic ported from the reference.
