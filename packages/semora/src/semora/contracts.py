"""Control-flow contracts shared across the effect boundary.

Messages and tool calls are Pydantic AI's. What is ours is the vocabulary of signals that stop an
attempt without being a tool failure, and the shape of an input waiting to enter model context.
"""

from typing import Literal, NamedTuple

from pydantic_ai.messages import ModelRequestPart, ToolCallPart

__all__ = ["AgentSuspended", "ControlSignal", "PendingInput", "StopReason", "Suspended", "ToolCall"]

ToolCall = ToolCallPart
"""One model-issued tool call. `tool_call_id` is its idempotency key and its durable step name."""

StopReason = Literal["completed", "aborted", "tool", "policy", "suspended"]
"""Why a run ended. `policy` is a `Halt` from a control point; `suspended` means it parked."""


class ControlSignal(Exception):
    """Signal that stops an attempt without becoming a model-visible tool failure."""


class Suspended(ControlSignal):
    """Base of every parking signal."""

    def __init__(self, signal: str) -> None:
        """Name the signal the attempt is waiting on."""
        super().__init__(signal)
        self.signal = signal


class AgentSuspended(Suspended):
    """The attempt parked: a gate asked a person, and nothing waits on the answer in-process.

    Attributes:
        pending_id: External id of the first undecided request, in model order.
        tool_call_id: The call it parks.
        pending: Every undecided `(pending_id, tool_call_id)` of the round, in model order.
    """

    def __init__(
        self, pending_id: str, tool_call_id: str, *, pending: list[tuple[str, str]] | None = None
    ) -> None:
        """Carry the undecided requests a host must route answers to."""
        super().__init__(pending_id)
        self.pending_id = pending_id
        self.tool_call_id = tool_call_id
        self.pending = list(pending) if pending else [(pending_id, tool_call_id)]


class PendingInput(NamedTuple):
    """One queued request part plus the provenance a bare part cannot preserve."""

    kind: str
    part: ModelRequestPart
    origin_id: str | None = None
