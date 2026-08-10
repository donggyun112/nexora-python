"""What the planner, orchestrator, and callers agree on: types and event vocabulary.

Nothing here knows how the loop is driven.
"""

from .events import BLOCKING, EventEnvelope, EventStream, EventType, RuntimeEvents, Sink
from .types import (
    Aborted,
    AdmitInputs,
    BaseMessage,
    BatchTools,
    CompactContext,
    DrainInputs,
    Emit,
    InvokeModel,
    ModelErrorKind,
    ModelFailure,
    ModelFailureAction,
    ModelStepError,
    ModelStreamFactory,
    OnModelFailure,
    OnSuspend,
    PendingInput,
    PreToolUse,
    ShouldStopAfterTurn,
    StopReason,
    ToolCall,
    Tools,
)

__all__ = [
    "BLOCKING",
    "Aborted",
    "AdmitInputs",
    "BaseMessage",
    "BatchTools",
    "CompactContext",
    "DrainInputs",
    "Emit",
    "EventEnvelope",
    "EventStream",
    "EventType",
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
    "RuntimeEvents",
    "ShouldStopAfterTurn",
    "Sink",
    "StopReason",
    "ToolCall",
    "Tools",
]
