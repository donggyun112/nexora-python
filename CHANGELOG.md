# Changelog

Only what changes for a caller: behaviour, and names that were exported. Internal refactors and
documentation corrections belong in the commit log, not here.

## Unreleased

### Fixed

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
