# Migrating to Semora 0.4

Semora 0.4 renames its durable unit and scopes the ledger by conversation. Nothing about what
runs, gates, parks or replays changed; what changed is what things are called and where records
are filed.

## Three ids, three layers

| Layer | Name | Owner | One per |
|---|---|---|---|
| Conversation | `conversation_id` | Pydantic AI | thread: every loop that continues the same dialogue |
| Branch | `branch_id` | Semora | durable unit: a first loop plus its resumes and recoveries; a fork is a new branch |
| Loop | `run_id` | Pydantic AI | `Agent.run()` invocation; a resume is a new loop with a new `run_id` |

In 0.3 the branch was called `run_id`, the same name Pydantic AI gives the loop. It is not the
loop, so it is now the branch.

## Renames

| 0.3 | 0.4 |
|---|---|
| `ExecutionContext(run_id=...)` | `ExecutionContext(branch_id=..., conversation_id=...)` |
| `ExecutionContext.session_id` (0.3.4, unreleased) | `ExecutionContext.conversation_id` |
| `AgentRuntime.run(run_id, ...)` and every other first argument | `branch_id` |
| `Agent(run_id=...)`, `agent.run(..., run_id=...)` | `branch_id` |
| `new_run_id()` | `new_branch_id()` (prefix `branch-`) |
| `Fenced.run_id`, `Contended.run_id`, `Indeterminate.run_id`, `EffectConflict.run_id` | `.branch_id` |
| `Transcript.record_run` / `read_run`, `RUN_FIELDS` | `record_branch` / `read_branch`, `BRANCH_FIELDS` |
| entry `metadata.run_id` | `metadata.branch_id`; `SCHEMA_VERSION` is `pai-v2` |
| `SessionScopedSteps` (0.3.4, unreleased) | `ConversationScopedSteps` |

Pydantic AI's own `run_id` is no longer intercepted. `Agent.run_sync(prompt, run_id=...)` hands it
to Pydantic AI as the loop id; bind the branch in the constructor instead.

## The ledger is scoped by conversation

`for_execution(ExecutionContext(branch_id, conversation_id))` on `MemorySteps` and
`PostgresSteps` returns that conversation's view of the ledger, and `AgentRuntime` reads and
writes every step, lease and queued input through it. A branch id under one conversation, under
another, and under none are three separate branches, so:

- a recorded effect replays only inside the conversation that recorded it;
- two conversations holding the same branch id never contend for a lease;
- a branch that parked inside a conversation is resumed by naming that conversation, through
  `conversation_id=` or the `ExecutionContext`. Without it the park is not found.

`conversation_id=` on `run`, `recover`, `fork`, `dispatch` and `committed_history` fills the
context; `resume`, `submit`, `state` and `pending` accept it too. A context that already names a
conversation keeps it. Without a conversation nothing is scoped, as before.

## Durable-data boundary

Postgres columns named `run_id` are `branch_id`, and the tables `ledger_run`, `ledger_run_model`,
`ledger_run_lease` are `ledger_branch`, `ledger_branch_model`, `ledger_branch_lease`. The schema
is applied with `create table if not exists`, so it does not alter an existing 0.3 database. As
with 0.3, start 0.4 on a fresh database and finish or freeze 0.3 branches with 0.3.
