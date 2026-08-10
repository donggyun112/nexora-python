"""Control-flow contracts shared by Nexora engines and runtime collaborators.

Messages and tool calls use LangChain types. Tool results are tagged mappings whose ``type`` is
``text``, ``image``, ``content``, ``error``, or ``suspend``.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCall

__all__ = [
    "Aborted",
    "AdmitInputs",
    "BaseMessage",
    "BatchTools",
    "CompactContext",
    "ControlSignal",
    "DrainInputs",
    "Emit",
    "InvokeModel",
    "ModelErrorKind",
    "ModelFailure",
    "ModelFailureAction",
    "ModelStepError",
    "ModelStreamFactory",
    "OnModelFailure",
    "OnSuspend",
    "PendingInput",
    "PreToolUse",
    "RecordMessages",
    "ShouldStopAfterTurn",
    "StopReason",
    "ToolCall",
    "Tools",
]

class ControlSignal(Exception):
    """Signal that stops an attempt without becoming a model-visible tool failure."""


StopReason = Literal["completed", "aborted", "tool", "policy"]
"""Why a run ended, carried on the terminal `done` event.

`aborted` matters most: a cancelled run must leave a record, not just a stream that stops.
"""

ModelErrorKind = Literal[
    "rate_limit",
    "context_overflow",
    "max_output_tokens",
    "authentication",
    "server",
    "invalid_request",
    "unknown",
]
"""Stable model-failure categories used by retry and compaction policy."""

ModelFailureAction = Literal["retry", "compact", "fail"]
"""A caller-owned recovery decision for one failed model request."""


class ModelFailure(NamedTuple):
    """Structured input to model recovery policy, independent of provider prose."""

    error_type: str
    error_kind: ModelErrorKind
    message: str
    partial: str
    attempt: int


class ModelStepError(Exception):
    """Carry a durable-boundary failure past provider failure classification."""

    def __init__(self, cause: Exception) -> None:
        """Preserve the original ledger or codec error for the runtime caller."""
        super().__init__(str(cause))
        self.cause = cause


OnModelFailure = Callable[[ModelFailure], Awaitable[ModelFailureAction]]
"""Choose bounded recovery for a model request; returning `fail` preserves the error event."""

CompactContext = Callable[
    [list[BaseMessage], ModelFailure], Awaitable[list[BaseMessage]]
]
"""Replace model-visible context after policy chooses `compact`."""

ModelStreamFactory = Callable[[], AsyncIterator[AIMessageChunk]]
"""Open one provider stream only when its durable result is not already recorded."""

InvokeModel = Callable[
    [str, ModelStreamFactory], AsyncIterator[AIMessageChunk]
]
"""Execute or replay one model request under a stable durable step identifier."""


class Tools(Protocol):
    """Runs tool calls and describes them. Implementations own permissions and sandboxing."""

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        """Execute a tool call and return its tagged result."""
        ...

    def get(self, name: str) -> dict[str, Any] | None:
        """The tool's definition, read for its `is_exclusive` / `terminates_loop` flags."""
        ...

    def list(self) -> list[dict[str, Any]]:
        """Every available tool as `{name, description, parameters}`, for binding to a model."""
        ...


@runtime_checkable
class BatchTools(Tools, Protocol):
    """Tool executor that owns concurrency policy for a complete gated round.

    Results may be returned in any order and are reordered by the caller. If execution suspends,
    unanswered calls are omitted and reissued by the model after resumption.
    """

    async def execute_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run `{call_id, name, input}` and return `{call_id, result}` for each one answered."""
        ...


PreToolUse = Callable[[ToolCall], Awaitable[dict[str, Any] | None]]
"""Gate returning ``None`` to allow, ``error`` to deny, or ``suspend`` for approval."""


# ── Hooks ────────────────────────────────────────────────────────────────────

Aborted = Callable[[], bool]
"""Cheap, non-raising predicate polled for cancellation at deterministic loop boundaries."""

class PendingInput(NamedTuple):
    """One queued message plus the provenance that a bare `BaseMessage` cannot preserve."""

    kind: str
    message: BaseMessage
    origin_id: str | None = None


DrainInputs = Callable[[], Awaitable[list[PendingInput]]]
"""Claims inputs waiting to enter model context. Returns [] when nothing is queued."""

AdmitInputs = Callable[[list[PendingInput]], Awaitable[None]]
"""Commits claimed inputs after they have been appended to the model context."""

RecordMessages = Callable[[list[BaseMessage]], Awaitable[None]]
"""Persist messages in the exact shape admitted to model-visible history."""

ShouldStopAfterTurn = Callable[[int, str, list[dict[str, Any]]], Awaitable[bool]]
"""Asked after every model round with (turn, text, calls_made). True ends the run.

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
"""Callback that persists a suspended, unexecuted tool request and continuation state."""
