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
| `01_minimal.py` | a tool round through the durable boundary, and what an outcome looks like |
| `02_approval.py` | `pre_tool_use` parks the run, the worker exits, `on_resume` re-decides under *current* policy — an approval a person gave is refused once the rules move |
| `03_recovery.py` | a worker dies mid-round; recovery restores the committed call, runs only the missing one, and never replays the model turn |
| `04_control_plane.py` | where a mask actually reaches: `on_inputs` changes what the model sees, a `Tools` wrapper changes what the ledger stores. Plus `before_finish` vetoing an ending |

Each file asserts what its prose claims, and `tests/test_examples.py` runs all of them, so an
example that stops being true fails the suite rather than quietly misleading someone.
