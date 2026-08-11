# Changelog

Only what changes for a caller: behaviour, and names that were exported. Internal refactors and
documentation corrections belong in the commit log, not here.

## Unreleased

### Added

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

**`nexora` is now a uv workspace of five distributions.** What was split out is what has its own
dependency footprint or its own audience — the line the Python ecosystem draws for
`langchain-openai`, `apache-airflow-providers-*`, `opentelemetry-exporter-*`. Layers sharing one
footprint stay subpackages of `nexora`, the way `django.db` stays inside `django`, so
`nexora.contracts`, `nexora.controls`, `nexora.tools`, `nexora.history`, `nexora.orchestrator` and
`nexora.engines.plain` are all unchanged, as is `from nexora import AgentRuntime`.

| was | is | install |
|---|---|---|
| part of `nexora.orchestrator` | `nexora_store` (**no dependencies**) | with `nexora` |
| `nexora.steps_postgres` | `nexora_store_pg` | `nexora[postgres]` |
| `nexora.permissions` | `nexora_permissions` | `nexora[permissions]` |
| `nexora.ui` | `nexora_ui` | `nexora[ui]` |

Three of those change what a default install contains. `nexora-permissions` is optional because
nothing in the runtime imports it — it is a rule table a host opts into. `nexora-store-pg` keeps a
compiled database driver out of the base install, and `nexora-ui` keeps FastAPI and uvicorn out.

`nexora_store` is the only distribution with an empty dependency list, and that is the point of it
existing: a `StepLog` stores opaque values under opaque keys, so implementing one needs neither a
message type nor `nexora` itself.

`tests/test_packaging.py` checks both boundaries — each distribution against the dependencies it
declares, and each layer inside `nexora` against what it is allowed to reach.

### Fixed

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
