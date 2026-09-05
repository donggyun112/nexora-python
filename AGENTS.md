# AGENTS.md

## Context tools

Prefer lean-ctx MCP tools (`ctx_shell`, `ctx_read`, `ctx_search`) or `/opt/homebrew/bin/lean-ctx -c` for shell output and reads. Full rules: `/Users/dongkseo/.codex/LEAN-CTX.md`. Use codecanvas for Python call-graph and control-flow questions.

## Project contract — Semora 0.3

- The user approved replacing the LangChain implementation with the Pydantic AI port. Legacy implementation is preserved in Git through `156e4b1`; do not maintain a second engine or restore retired message abstractions.
- Pydantic AI owns messages, tool calls, models, tool definitions and the agent loop. Use its native types.
- Semora owns the execution boundary in `effects.py`, policy vocabulary/composition in `controls.py`, durable suspension/revalidation, transcripts, and run/dispatch lifecycle in `runtime.py`.
- One planner and one execution path: delegate to Pydantic AI. Do not rebuild the legacy agent loop.
- Gates return `Continue`, `Deny`, or `Suspend`. Suspension releases the worker; approval is an input to `on_resume`, not unconditional authorization.
- Commit tool results before journaling. Failed tools become recorded error results; runtime signals (`ControlSignal`, `Contended`, `Fenced`, `Indeterminate`) pass through the single catch boundary.
- An incomplete intent is indeterminate by default. Retrying it requires caller opt-in and an appropriate external idempotency contract. A ledger/fencing token alone does not guarantee exactly-once external effects.
- Tool-call identity belongs to one run. Cross-run business identity belongs to the host. Preserve this distinction in APIs, examples and claims.
- No iteration cap, timeout or permission policy is imposed beyond caller configuration.
- `semora-store` has no dependencies and knows nothing of Pydantic AI. `semora-store-pg` depends only on the store contract and PostgreSQL driver.
- Public API changes must update `docs/API.md`. Document intentional migration differences; tests are the behavior reference.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build --all-packages
```

PostgreSQL conformance tests use `SEMORA_TEST_DSN`; skipped tests are not durable verification. Prefer fakes and meaningful behavior tests. Keep strict typing and `src/` layout.
