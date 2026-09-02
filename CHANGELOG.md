# Changelog

Only what changes for a caller: behaviour, and names that were exported. Internal refactors and
documentation corrections belong in the commit log, not here.

## Unreleased

### Added

- **`semora_fork.resume_point` and `semora_fork.RERUNS`.** A fork coordinate can now say
  which control point the run enters first when restored there — `on_inputs` for an input
  coordinate, `pre_tool_use` for a leaf that still owes a tool answer, `before_model` for
  any other leaf — using the predicate recovery already uses. `RERUNS` lists, in loop order,
  the control points that run again over the recorded round from each of those. Together
  they answer the question a person holds while choosing where to branch: will the policy I
  just turned on actually run from here. `unanswered_tool_calls` is public for the same
  reason.

- **A branch can re-journal a result without re-gating it.** `fork_event(...,
  rejournal=True)` resumes the round still owed at a leaf with the source run's finished
  effect records standing in for the new run's absent ones — `Orchestrator.recover_pending`
  and `AgentRuntime.run` take `replay_from` for the same purpose. The gate is not asked
  again, the tool does not run again, and the journal alone sees the recorded result, as the
  tool returned it. Until now the only way to re-mask a result was to resume at the gate,
  which under an approval policy asked a person to approve an effect that had already
  happened. `resume_point(..., rejournal=True)` reports `post_tool_use`, and `RERUNS` lists
  what runs from there. The source ledger is read, never written; a call it never finished
  falls back to the gate.

### Fixed

- **`MemorySteps` keeps a copy of what it records.** It kept the caller's dict, so a journal
  rewriting a tool result in place after the step recorded it rewrote the record too, and the
  "raw" result read back masked. The Postgres store serialized on write and never had the
  problem; the memory store now copies on write and on read, so the two keep the same promise.
  Surfaced by the re-journal branch, which is the first reader that needed the raw copy.

- **A replayed message now moves the branch every reader sees.** `TranscriptWriter.record`
  advances its own chain position when an entry turns out to be a duplicate, but nothing was
  appended, so `active_branch` — which reads the tip off the entries — still reported the leaf
  the replay had started from. A run that resumed an unanswered tool round therefore left its
  before-tool and after-tool coordinates on one leaf: forking "after the result" resumed before
  it and ran the whole tool boundary again. A duplicate record now publishes a `leaf` marker at
  the entry it landed on, so the writer and every reader agree on where the conversation is.

### Changed

- **`after_tool_call` is now `post_tool_use`.** The hook fires the `post_tool_use` event and
  pairs with `pre_tool_use`, so it carried two names for one seam and a control plane read as
  though the two sides of a tool call belonged to different vocabularies. Renamed everywhere it
  is a name: the `Controls` method, the `ControlPlane(post_tool_use=...)` keyword, the
  `PostToolUse` stage type, and `Orchestrator.post_tool_use_once`. The durable marker is still
  `after:{call_id}`, so ledgers written before this release resume unchanged. No alias: callers
  rename the keyword.

## 0.1.0 — 2026-08-30

### Added

- **Composable command routing.** `semora.dispatch` grew from the command vocabulary into the
  assembly layer: `CommandRouter` is an ordered transition table over the runtime's public
  primitives, and `default_router()` — `StartRun`, `QueueSteer`, `ResumeApproval`,
  `RecoverInterrupted`, `ReplayJournal` — is the preset behind `AgentRuntime.dispatch`, which
  now just delegates to it. A row matches on two axes split on purpose: `applies(command)` is a
  pure predicate, and `states` declares — as data — which observed states the row accepts, so
  the router reads the run's state only when a candidate row declared one or a refusal must
  name it; a `Prompt` or `Answer` routes with no state reads at all. `Contended` from a row is
  a hand-off to the next matching row, which is how a busy run's `Prompt` becomes a durable
  enqueue. Drop a row and exactly its behavior disappears — without `QueueSteer`,
  contention propagates for the host to handle; without the recover rows, `Recover` is refused
  with the observed state. Host-written transitions slot into the order without subclassing.

- **`AgentRuntime.state` and `AgentRuntime.committed_history`.** The two reads dispatch routed
  by were private; transitions written outside the package need the same primitives, so they
  are public. `state()` names the run from one observation: a parked continuation
  (`waiting`/`switching`/`resuming`), or the run record's verdict — `fresh`, `completed`, or
  `interrupted` (an open round; a crash and a live worker look the same until a lease attempt
  tells them apart).

### Changed

- **`InvalidTransition.state` names the precise state.** An `Answer` refused on an unparked
  run used to say `idle`; it now carries the run record's vocabulary (`fresh`/`completed`),
  the same names `Recover` refusals already used.

- **`ChatModel`.** `semora-llm` is a workspace distribution the core depends on. It wraps
  the official `openai` SDK and streams LangChain chunks. OpenAI, OpenRouter, and xAI
  share that wire; `openrouter()` / `xai()` are presets. Anthropic and Google native
  APIs stay extras.

- **Provider extras.** `semora[openai]`, `semora[anthropic]`, `semora[google]`,
  `semora[xai]`, and `semora[openrouter]` install the matching LangChain chat adapter.
  The core still depends only on `langchain-core`. Construct the model from that
  adapter (`from langchain_openai import ChatOpenAI`); Semora does not re-export it.

- **Control plane on the public package.** The top-level `semora` export is the every-run
  vocabulary: `AgentRuntime`, `ChatModel`, the control-plane types, `MemorySteps`,
  `ExecutionContext`, `HostWorkspaceProvider`. Feature packs and power-user seams live on
  their submodules; see Breaking. Workspace internals (`ToolContext`, snapshot backends,
  sandbox HTTP types) stay on `semora.workspace` and `semora.sandbox_remote`.

- **Shared agent definitions.** `AgentDefinition` is the common identity contract implemented by
  `Agent`, `FactoryAgent`, `RunnerAgent`, and `HttpAgent`. The executable local `Agent` binds a
  LangChain model, `Tools`, and system prompt independently from run input and orchestration policy.
  `AgentRuntime.run(run_id, agent, prompt)` accepts it while the previous model/tools call shape
  remains supported.

- **Detachable runtime orchestration.** `AgentRuntime()` now drives the ReAct planner directly,
  without constructing an `Orchestrator` or recording model/tool steps. Pass
  `orchestrator=DurableRuntimeOrchestrator(steps)` when suspension and recovery are required;
  `store=steps` remains shorthand. Custom orchestrators can wrap the model and tool boundaries.

- **Explicit built-in execution context.** `builtin_tools(context=...)` accepts a caller-managed
  `ToolContext` for direct execution while retaining the same `WorkspaceFS` confinement used by
  `AgentRuntime`; missing-workspace read errors now point callers to both supported setup paths.

- **Progressive model context.** `SystemPrompt` composes cache-stable and explicitly volatile
  sections; source-neutral `SkillRegistry` discovers metadata through asynchronous `SkillSource`
  adapters, while the `skill` tool schema alone exposes their bounded catalog and `SkillTools`
  loads full bodies only on invocation. `DirectorySkillSource` provides `SKILL.md` support.
  `DeferredTools` adds transcript-recoverable `tool_search` activation, and the ReAct planner
  rebuilds LangChain tool bindings at every model round so a newly selected schema becomes callable
  without restarting the run.

- **Model invocations are durable effect steps.** A request is identified by its model, bound tools
  and model-visible context. Completed stream chunks are replayed from `StepLog` after a crash
  without calling the provider again; a request left `running` raises `Indeterminate` rather than
  risking a duplicate charge and a different answer. Failures before the first visible chunk are
  cleared so the existing bounded retry and compaction policy can still act. Custom model wrappers
  can supply a stable `model_identity` when LangChain identifying parameters are insufficient.

### Packaging

**`semora` is now a uv workspace of five distributions.** What was split out is what has its own
dependency footprint or its own audience — the line the Python ecosystem draws for
`langchain-openai`, `apache-airflow-providers-*`, `opentelemetry-exporter-*`. Layers sharing one
footprint stay subpackages of `semora`, the way `django.db` stays inside `django`, so
`semora.contracts`, `semora.controls`, `semora.tools`, `semora.history`, `semora.orchestrator` and
`semora.engines.plain` are all unchanged, as is `from semora import AgentRuntime`.

| was | is | install |
|---|---|---|
| part of `semora.orchestrator` | `semora_store` (**no dependencies**) | with `semora` |
| `semora.steps_postgres` | `semora_store_pg` | `semora[postgres]` |
| `semora.permissions` | `semora_permissions` | `semora[permissions]` |
| `semora.ui` | `semora_ui` | `semora[ui]` |

Three of those change what a default install contains. `semora-permissions` is optional because
nothing in the runtime imports it — it is a rule table a host opts into. `semora-store-pg` keeps a
compiled database driver out of the base install, and `semora-ui` keeps FastAPI and uvicorn out.

`semora_store` is the only distribution with an empty dependency list, and that is the point of it
existing: a `StepLog` stores opaque values under opaque keys, so implementing one needs neither a
message type nor `semora` itself.

`tests/test_packaging.py` checks both boundaries — each distribution against the dependencies it
declares, and each layer inside `semora` against what it is allowed to reach.

### Fixed

- **An aborted model stream no longer freezes the turn it was serving.** Stopping mid-generation
  closes the model stream, and the durable step it belonged to kept its `running` intent even
  though the partial reply was discarded and never appended to history. The next attempt rebuilt
  the identical request, hashed to the same step key, and raised `Indeterminate` instead of asking
  the provider — so the run could not get past its own stop button without an explicit
  `force_retry`. Closing the stream now clears the intent, which a crash still does not do: a dead
  worker never runs the close path, and cancellation arrives separately as `CancelledError`.

- **Clearing an unfinished step is lease-protected.** `forget` was the only mutating ledger write
  that skipped its fencing token. A worker whose lease had lapsed did not know it — renewal keeps
  the old token deliberately — so on its way out it could delete the running intent of the worker
  that had taken over that step, and the effect in flight would replay on the attempt after that.

- **Model failures retain a stable policy classification at the orchestrator boundary.** Error
  events now carry both the provider exception class (`error_type`) and a normalized
  `error_kind`; `AgentFailed` preserves both instead of discarding the classification before
  retry or compaction policy can inspect it. `ModelFailurePolicy` now applies bounded transient
  retries or caller-supplied context compaction inside the failed model round, so recovery does
  not replay earlier tool effects. Failures after streamed text are never retried automatically.

- **Caller turn limits now cover tool-free rounds and `before_finish` vetoes.**
  `should_stop_after_turn` previously ran only after tool execution, so a verifier that always
  returned `Proceed` could bypass the caller's only iteration bound. The hook now observes every
  completed model round and ends a capped tool-free run with `stop_reason="policy"` before another
  model call.

- **`Controls.before_finish` is actually called now.** The protocol method, the `FinishPolicy`
  composer and the `ControlPlane` slot all existed with no caller anywhere, so a registered
  verifier silently did nothing. `react_loop` now asks it on a round the model ended without tool
  calls, after the late-steer check: `Proceed` sends the run around again with its steers admitted
  beside the next model call, `Halt` ends it. Registering nothing keeps the previous behaviour
  exactly. `FinishPolicy` is also exported now — it was missing from `controls.__all__`.

  Note that `FinishPolicy` returns the engine's stop reason, so a gate decides only whether the run
  continues; it cannot relabel an ending it did not object to.

### Breaking

- **The top-level package is the every-run vocabulary.** `semora.__all__` is a closed
  contract, not an accumulating re-export of every feature pack. Names that moved stay
  public on the submodule that already owned them; there is no deprecation shim.

  | name | now |
  |---|---|
  | `Answering`, `Authority`, `FactoryAgent`, `HttpAgent`, `RunnerAgent`, `Subagent`, `Subagents` | `semora.subagents` |
  | `DirectorySkillSource`, `SkillRegistry`, `SkillTools` | `semora.skills` |
  | `SystemPrompt`, `prompt_section`, `volatile_prompt_section` | `semora.prompts` |
  | `DeferredTools` | `semora.tool_search` |
  | `Goal`, `goal_complete`, `goal_gate` | `semora.goal` |
  | `PlanMode`, `plan_mode_exit`, `plan_mode_gate` | `semora.plan_mode` |
  | `BuiltinTools`, `builtin_tools`, `ExecToolOptions` | `semora.builtins` |
  | `RemoteSandboxClient` | `semora.sandbox_remote` |
  | `FallbackChatModel`, `ModelProvider` | `semora.providers` |
  | `openrouter`, `xai` | `semora_llm` |
  | `react_loop` | `semora.engines.plain` |
  | `DurableRuntimeOrchestrator` | `semora.orchestration` |
  | `ModelFailurePolicy` | `semora.orchestrator` |
  | `ObservationEventSink` | `semora.contracts` |
  | `WorkspaceProvider` | `semora.workspace` |

- **`StepLog` declares `forget`, and `semora_store.ClearableSteps` is gone.** Clearing was modelled
  as an optional capability the orchestrator checked for, but three core paths depend on it — a step
  that raises, a model request that failed before its first chunk, and an aborted stream all end by
  removing their intent. A ledger without it turned every one of those into a permanent
  `Indeterminate` and said nothing. A custom `StepLog` must now implement
  `forget(run_id, key, token=0)`, removing only unfinished intent and leaving a `done` step alone;
  both bundled stores already did. `Orchestrator.force_retry` no longer raises `NotImplementedError`.

- **`controls.gate()` maps `{"type": "allow"}` to `Continue` instead of `Deny`.** A stage answering
  `allow` is an opinion that later stages still overrule — the rule `Permissions` and
  `permissions.resolve_rules` already stated and this helper contradicted. A host that wired an
  external hook returning that shape was having its permissive answer inverted into a refusal.
  No action needed unless you relied on the inversion; a gate that means "deny" should return an
  `error` result.

- **`orchestrator.PermissionChain` removed.** It was `controls.Permissions` under another name.
  Replace `PermissionChain(a, b).resolve` with
  `ControlPlane(pre_tool_use=Permissions(gate(a), gate(b)))`; the composition rules (a deny wins,
  an ask is remembered, an allow decides nothing) are identical. A `record=` argument becomes
  `after_tool_call=Journal(writer(record))`.

- **`contracts.types.Permissions` protocol removed.** The engine depends on `controls.Controls`,
  and nothing had referenced this shape since. `controls.Permissions` is a different thing — the
  stage composer — and is unaffected.

- **`Orchestrator.claim_inputs` takes `represented: set[str]`, not a message list.** It only ever
  read the ids. Callers pass `{m.id for m in history if m.id is not None}`.

- **`USER_PROMPT_SUBMIT` no longer carries `prompt`.** The event is published at inbox time, before
  `Controls.on_inputs` can screen anything, so the text on it was the pre-mask original and it
  stayed in every downstream sink. The admitted text is published by `CONTEXT_INJECTED` after
  screening; join on `input_id` / `origin_id`. `PRE_COMPACT` and `ELICITATION` also left
  `contracts.BLOCKING`, which is now a mapping from event to the `Controls` method that decides it
  — neither had one.
