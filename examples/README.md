# Examples

Run any of them with no credentials:

```bash
uv run python examples/01_minimal.py
```

The model is scripted (`_scripted.py`), so each run is deterministic and the interesting part is
the runtime rather than the LLM. Swap `scripted(...)` for a real chat model and `Files` for your own
executor — nothing else changes, because the runtime only asks a model to stream and a tool box to
execute.

| | shows |
|---|---|
| `01_minimal.py` | the default runtime: one tool round, no ledger, and what an outcome looks like |
| `02_approval.py` | `pre_tool_use` parks the run, the worker exits, `on_resume` re-decides under *current* policy — an approval a person gave is refused once the rules move |
| `03_recovery.py` | a worker dies mid-round; recovery restores the committed call, runs only the missing one, and never replays the model turn |
| `04_control_plane.py` | where a mask actually reaches: `on_inputs` changes what the model sees, a `Tools` wrapper changes what the ledger stores. Plus `before_finish` vetoing an ending |
| `05_workflow.py` | the other composition — `Orchestrator` on the outside, a whole agent run as one durable step beside `sendToPharmacy`, and `signal()` ending the attempt until a person answers |
| `06_bare_loop.py` | the default `AgentRuntime`: no ledger at all. The full control plane with no orchestrator and no store, and the one decision that needs one: parking a call |

Each file asserts what its prose claims, and `tests/test_examples.py` runs all of them, so an
example that stops being true fails the suite rather than quietly misleading someone.
