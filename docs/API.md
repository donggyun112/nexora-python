# Semora 0.3 API

Semora extends Pydantic AI. Use native `ToolCallPart` and model messages throughout; 0.2 message dictionaries and positional runtime calls are not compatible. See [migration](MIGRATION-0.3.md).

## Imports

```python
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from semora import (
    Agent,
    AgentRuntime,
    AgentSuspended,
    Answer,
    Continue,
    ControlPlane,
    ControlSignal,
    Ctx,
    Deny,
    Effects,
    FinishPolicy,
    Halt,
    Ingress,
    Journal,
    MemorySteps,
    MemoryTranscript,
    Outcome,
    PendingInput,
    Permissions,
    Proceed,
    Prompt,
    Recover,
    ResumeInput,
    Steering,
    Suspend,
    Suspending,
    gate,
    new_run_id,
    tool,
    writer,
)
from semora_store import Contended, ExecutionContext, ExecutionStore, Fenced, Indeterminate
from semora_store_pg import PostgresSteps, PostgresTranscript
```

## Runtime

`AgentRuntime(execution_store=None, *, transcript=None, lease_ttl=60.0, retry_running=False)`

All methods below are async. `run_id` accepts a string or `ExecutionContext`.

| Method | Result and contract |
|---|---|
| `run(run_id, agent, prompt=None, *, controls=None, rules_version="", prompt_id=None, conversation_id=None, message_history=None, deferred_tool_results=None, deps=None, capabilities=(), **options)` | `Outcome`; `agent` is a Pydantic AI agent. Extra model/run options reach Pydantic AI. Acquires and renews a run lease. |
| `resume(run_id, pending_id, answer, agent, *, controls=None, rules_version="", deps=None)` | Records an answer, then revalidates when all parked calls are answered. Unanswered siblings raise `AgentSuspended` again. Unknown pending IDs raise `LookupError`. |
| `recover(run_id, agent, history, *, controls=None, rules_version="", conversation_id=None, deps=None)` | Continues from native message history. Reuses committed effects; unreported effects raise `Indeterminate` unless retry was explicitly enabled. |
| `fork(source, at, target, agent, prompt=None, *, history=None, regate=False, controls=None, rules_version="", source_conversation_id=None, conversation_id=None, deps=None, **options)` | Starts `target` from `source`'s transcript at entry uuid `at` (`None`: the active tip), or from `history` when the host keeps its own coordinates. Effects the source finished in that history are copied to the new run's ledger and replay; `regate=True` asks the new run's `pre_tool_use` about each first, and only `Continue` replays. A call the source started and never reported is copied as started, so `retry_running` decides. The rest runs under the new run's policy. The source is never written. |
| `committed_history(run_id, conversation_id=None)` | `list[ModelMessage]`; requires a transcript. Supply this to `recover`. |
| `submit(run_id, item)` | Enqueues and returns a `PendingInput`; requires an execution store. |
| `dispatch(run_id, agent, command, *, controls=None, **options)` | Routes `Prompt`, `Answer`, or `Recover` using durable state. Attach both ledger and transcript. |
| `pending(run_id)` | Undecided `(pending_id, tool_call_id)` pairs in model order. |
| `state(run_id)` | `fresh`, `interrupted`, `completed`, or suspension state `waiting`/`resuming`; without a transcript an unparked run is `idle`. |

`interrupted` includes a still-running worker. Only acquiring the lease distinguishes it from a dead one.

Approval updates and finalization hold the run lease; concurrent `resume` calls may raise `Contended` before accepting the answer, so the host should retry that answer. A prompt submitted through `run` while a fully answered continuation is resuming is enqueued before `Contended` is raised. Keep a stable `prompt_id` on retries to avoid enqueueing it twice.

`Outcome` exposes `output`, `stop_reason`, optional native `result`, `pending`, `pending_id`, `suspended`, and `all_messages()`. A runtime-level park is raised as `AgentSuspended`, carrying `pending_id`, `tool_call_id` and ordered `pending` pairs. The class-agent interface below converts that signal to a suspended outcome, whose `all_messages()` is empty because no native completed result exists.

Use `{"type": "approve"}` for approval, `{"type": "approve", "args": {...}}` to approve with replaced arguments, and `{"type": "error", "message": "declined"}` for refusal. Replaced arguments are validated by Pydantic AI and are what `on_resume` sees as the call; the original request stays in `ResumeInput.request`. Refusal cannot be lifted by `on_resume`. Policy-version strings are host-provided labels.

## Optional class agent

Subclass `semora.Agent`; this is a Pydantic AI Agent subclass with run-bound convenience methods.

| Class member | Meaning |
|---|---|
| `llm` | Pydantic AI model instance or model name |
| `prompt` | Instructions string or instance method rendering instructions |
| `output` | Pydantic AI output type |
| `uses` | Functions, native tools, toolsets, or capabilities |
| `@tool` | Expose an instance method as a tool |
| `store`, `transcript` | Shared instance or a factory resolved once per subclass |
| Seven control-point methods | Default policies, overridden by an explicit controls object |

`@tool` also accepts `concurrency_safe=False`, `requires_approval=False`, `name=None`, and `description=None`. A `requires_approval=True` tool parks through the `pre_tool_use` gate like a `Suspend`, with the call's `tool_call_id` as its `pending_id`; a gate's `Deny` still wins, and `on_resume` re-decides the answer.

Construct with `run_id=None`, `runtime=None`, and supported configuration overrides. An instance binds to one run ID; changing its ID or overlapping attempts raises `RuntimeError`. `run(prompt=None, ...)` mints an ID if needed. `resume(answer, pending_id=None, ...)` defaults to the first pending request. `recover(history=None, ...)` loads committed history when omitted. `fork(source, at=None, prompt=None, *, history=None, regate=False, ...)` makes this instance's run a branch of another run. `dispatch(command, ...)`, `submit(item, ...)`, `state()` and `pending()` operate on that instance's run. `last` holds the last outcome.

Instance fields are not automatically durable. Restore trusted tool configuration when constructing a replacement instance. Mutable class attributes are shared; do not put per-run working state there.

## Control points

`ControlPlane` accepts any subset of these async functions. `Ctx` contains `turn`, native `messages`, `calls_made`, `text`, and `subject`. A tool call is a native `ToolCallPart`: access `tool_name`, `tool_call_id`, `args_as_dict()`.

| Point | Signature | Composition |
|---|---|---|
| `on_inputs` | `(ctx, list[PendingInput]) -> list[PendingInput] \| Halt` | `Ingress(*screens)` chains rewrites |
| `before_model` | `(ctx) -> Proceed \| Halt` | `Steering(*sources)` accumulates native request parts |
| `pre_tool_use` | `(ctx, call) -> Continue \| Deny \| Suspend` | `Permissions(*stages)`, denial outranks suspension |
| `post_tool_use` | `(ctx, call, result) -> None` | `Journal(*writers)` runs in order; may mutate a mutable result |
| `before_finish` | `(ctx, stop_reason) -> Proceed \| Halt` | `FinishPolicy(*gates)` combines steering or stops |
| `on_resume` | `(ctx, call, resume) -> Continue \| Deny \| Suspend` | Receives human input plus suspended/current version labels |
| `on_suspend` | `(ctx, call, request, snapshot, completed) -> None` | `Suspending(*persisters)` completes before durable park |

`Continue()` allows; `Deny(result)` gives the model a result without executing the tool; `Suspend({"pending_id": ...})` requests a durable pause. `Proceed(steers=())` continues with optional native request parts. `Halt(reason)` ends the attempt. `PendingInput(kind, part, origin_id=None)` holds a native request part. `ResumeInput` contains `answer`, `request`, `suspended_rules_version`, `current_rules_version`.

`gate(fn)` adapts a call-only function returning `None` or an error/suspend result. `writer(fn)` adapts a call/result-only journal function.

## Effect and storage contract

`Effects(store, run_id, token=0, *, controls=None, retry_running=False, rules_version="", subject="", resumed=None, inputs=None, record=None)` is the capability the runtime installs per attempt. Prefer `AgentRuntime` so lease and continuation ownership stay centralized.

- `tool:{call_id}` stores the original effect result envelope. Ordinary tool exceptions produce committed error results. `ControlSignal`, ledger signals and cancellation propagate.
- `after:{call_id}` stores the completed model-visible journal projection separately. Recovery reuses that projection without leaking the unredacted original. A journal can execute again if the process dies before committing its projection: external journal effects must be idempotent.
- `running` without a result is indeterminate. Fencing protects ledger writes, not arbitrary external APIs. A forced retry needs the host's idempotency/reconciliation contract.
- Run-scoped keys do not deduplicate a business operation across runs. The console supplies stable request/customer keys for its simulated payment separately.
- `MemorySteps` and `MemoryTranscript` do not survive a process restart. PostgreSQL adapters use an async psycopg pool and share the same protocols; see the conformance tests for setup and behavior.

`Contended` means another worker holds the run lease. `Fenced` means a stale writer's token was rejected. `Indeterminate` includes `run_id` and `step`. `InvalidTransition` (from `semora.dispatch`) carries the observed `state` and rejected `command`. Do not convert these signals into ordinary tool errors in host adapters.
