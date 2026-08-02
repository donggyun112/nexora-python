# Nexora for Python

Durable multi-agent runtime for Python.

> **Status:** pre-alpha. The TypeScript Nexora implementation remains the behavioral
> reference while the runtime is ported one contract at a time.

## Direction

The agent loop is an ordinary `async while`. Control flow — when to stop, when to inject a
steer, when to hand a decision to a human — is expressed in Python rather than as a graph,
because a graph would re-describe `if` and `while` as data and turn local variables into
serialized state. Durability is layered around the loop instead of built into it.

Nexora owns the product semantics: model providers and fallback, tool execution and ordering,
permission policy, authority attenuation, sandboxed workspaces, delegation, handraise,
tenancy, transport, stores, and observability.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest
```

## Current scaffold

```text
src/nexora/
├── contracts/       # what everything agrees on
│   ├── types.py     #   messages, tool calls, hook signatures
│   └── events.py    #   event vocabulary and envelope
├── engines/         # the interchangeable part
│   ├── plain/       #   the loop as an `async while`
│   └── langgraph/   #   the same loop on LangChain's `create_agent`
├── providers/       # provider stream → the chunk vocabulary engines read
├── tools.py         # tool policy and execution — shared by both engines
└── history.py       # suspension snapshots — shared by both engines
```

Both engines take the same inputs and emit the same events;
`tests/test_engine_conformance.py` is what makes that a fact rather than a claim. The
LangGraph one needs the `langgraph` extra:

```bash
uv sync --extra langgraph
```

The loop covers the reference's stop conditions, exclusive and terminating tools, steering,
suspension, and the permission gate. Transcript persistence, resume, context compaction, and
attachments are not ported yet.

## License

MIT
