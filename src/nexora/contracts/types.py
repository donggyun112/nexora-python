"""What the engines and their collaborators agree on.

Messages and tool calls are LangChain's (`BaseMessage`, `ToolCall`) rather than ours. Owning
those types bought a translation layer in every direction and a provider adapter that
re-implemented `ChatOpenAI`; the types themselves were never the interesting part. What stays
ours is the loop's control-flow contract — the hooks below, and the event vocabulary in
`events.py`.

Tool results stay plain tagged dicts. Their `type` values:

    text | image | content | error | suspend
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage, ToolCall

__all__ = [
    "Aborted",
    "AdmitInputs",
    "BaseMessage",
    "BatchTools",
    "DrainInputs",
    "Emit",
    "OnSuspend",
    "PendingInput",
    "PreToolUse",
    "ShouldStopAfterTurn",
    "StopReason",
    "ToolCall",
    "Tools",
]

StopReason = Literal["completed", "aborted", "tool", "policy"]
"""Why a run ended, carried on the terminal `done` event.

`aborted` matters most: a cancelled run must leave a record, not just a stream that stops.
"""


class Tools(Protocol):
    """Runs tool calls and describes them. Implementations own permissions and sandboxing."""

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]: ...

    def get(self, name: str) -> dict[str, Any] | None:
        """The tool's definition, read for its `is_exclusive` / `terminates_loop` flags."""
        ...

    def list(self) -> list[dict[str, Any]]:
        """Every available tool as `{name, description, parameters}`, for binding to a model."""
        ...


@runtime_checkable
class BatchTools(Tools, Protocol):
    """An executor that takes a whole round at once and applies its own concurrency policy.

    Declared rather than discovered. This capability used to be found with
    `getattr(tools, "execute_batch", None)` — behaviour that existed but was not a contract, so
    nothing type-checked it and nothing said what it promised. What it promises:

    - **The batch arrives gated.** Every call in the round has already been through
      the `PRE_TOOL_USE` control point; refused ones are absent. Once calls are running there is no
      way to stop one a later gate would have refused, so gating cannot be interleaved.
    - **Order is the executor's to decide, within the round.** Concurrency is opt-in *per batch*
      because safety is a property of a pair of calls, not of one tool: two calls that are each
      safe alone may still conflict with each other. Only the executor can see both.
    - **Results may come back in any order** — `execute_calls` re-sorts them to call order.
    - **A suspend cuts the round short.** Calls the executor never answered are dropped rather
      than faked, and the model re-issues them on resume.

    Without this, `execute_calls` runs the round one call at a time. That is the fail-closed
    default: sequential is what makes a retried batch replay identically (ADR-002).
    """

    async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run `{call_id, name, input}` and return `{call_id, result}` for each one answered."""
        ...


PreToolUse = Callable[[ToolCall], Awaitable[dict[str, Any] | None]]
"""A simple gate registered at the `PRE_TOOL_USE` control point.

    None                      allow  → the tool runs
    {"type": "error", ...}    deny   → the model sees a failed call and can react
    {"type": "suspend", ...}  ask    → the turn is checkpointed and stops here

`ask` deliberately does not block waiting for a human. A gate that awaits an answer holds
the worker for as long as the person takes, which caps approvals at whatever timeout the
transport allows. Suspending instead costs nothing while stopped, so a policy approval can take
days — the supervisor answers `suspend`, then waits on its own `signal`.

There is one approval path, not two. An agent asking a human (`handraise`) and a policy refusing
to let a tool run both end as a suspension, so both can take days and neither holds a worker.
"""


# ── Hooks ────────────────────────────────────────────────────────────────────

Aborted = Callable[[], bool]
"""True once the run has been cancelled. Polled, never pushed.

Checked at every round boundary **and at every streamed chunk**, so a shutdown arriving early in
a long generation does not have to wait it out. Polling rather than a callback is deliberate: a
handler firing at an arbitrary await would make the sequence of steps depend on timing, and a
durable replay could not reproduce it. The loop picks the points where observing is safe.

Idempotent and cheap, please — it is called once per chunk. Being asked repeatedly, or being
already true before the run starts, is normal and produces exactly one terminal event.

**Must not raise.** It is consulted inside the provider's own error path, where the loop is asking
"was that failure an abort?", and an exception there would be reported as a provider error — a
caller's bug wearing the model's name.

Installing an OS signal handler is the caller's job, not the loop's; a library that grabs
`signal.signal` breaks its host. The seam is here:

    stopping = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
    react_loop(..., aborted=stopping.is_set)
"""

class PendingInput(NamedTuple):
    """One queued message plus the provenance that a bare `BaseMessage` cannot preserve."""

    kind: str
    message: BaseMessage
    origin_id: str | None = None


DrainInputs = Callable[[], Awaitable[list[PendingInput]]]
"""Claims inputs waiting to enter model context. Returns [] when nothing is queued."""

AdmitInputs = Callable[[list[PendingInput]], Awaitable[None]]
"""Commits claimed inputs after they have been appended to the model context."""

ShouldStopAfterTurn = Callable[[int, str, list[dict[str, Any]]], Awaitable[bool]]
"""Asked after every tool round with (turn, text, calls_made). True ends the run.

This is also where an iteration cap lives — the loop does not impose one.
"""

Emit = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""Publish one observation. Any subscriber return value is ignored.

Control and observation share lifecycle names but not transport semantics: `Controls.pre_tool_use`
returns the decision at the `PRE_TOOL_USE` point, while `Emit(PRE_TOOL_USE, ...)` tells UI and
audit consumers that the point was reached. A published event never grants authority.
"""

OnSuspend = Callable[
    [ToolCall, dict[str, Any], list[BaseMessage], list[dict[str, Any]]], Awaitable[None]
]
"""Called when a pre-tool permission gate suspends an unexecuted tool request.

Calls the round never reached are deliberately **not** listed here. They are steps — a tool call's
id is its step name, so the ledger already knows which ones finished and which never started, and a
second list of them would be a second answer to the same question. Wrap the executor in
`nexora.tools.Stepped` and a resumed run re-executes exactly the ones that never ran.

The whole permission request is handed over, not just its `pending_id`: any external approval
handle lives in fields the loop has no business naming. Persisting it is the caller's job.
"""
