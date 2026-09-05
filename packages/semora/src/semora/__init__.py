"""Semora: an execution boundary for Pydantic AI agents.

Completed effects replay from a ledger; unreported effects remain indeterminate unless the host
explicitly allows retry. Messages, tool calls, models and the agent loop are Pydantic AI's.
"""

from semora_store import (
    Contended,
    ExecutionContext,
    Fenced,
    Indeterminate,
    MemorySteps,
    MemoryTranscript,
)

from .agent import Agent, tool
from .contracts import AgentSuspended, ControlSignal, PendingInput, StopReason, Suspended
from .controls import (
    Continue,
    ControlPlane,
    Controls,
    Ctx,
    Deny,
    FinishPolicy,
    Halt,
    Ingress,
    Journal,
    Permissions,
    Proceed,
    ResumeInput,
    Steering,
    Suspend,
    Suspending,
    gate,
    writer,
)
from .dispatch import Answer, InvalidTransition, Prompt, Recover
from .effects import CONCURRENCY_SAFE, Effects
from .ids import new_branch_id
from .runtime import AgentRuntime, Outcome

__version__ = "0.4.0"

__all__ = [
    "CONCURRENCY_SAFE",
    "Agent",
    "AgentRuntime",
    "AgentSuspended",
    "Answer",
    "Contended",
    "Continue",
    "ControlPlane",
    "ControlSignal",
    "Controls",
    "Ctx",
    "Deny",
    "Effects",
    "ExecutionContext",
    "Fenced",
    "FinishPolicy",
    "Halt",
    "Indeterminate",
    "Ingress",
    "InvalidTransition",
    "Journal",
    "MemorySteps",
    "MemoryTranscript",
    "Outcome",
    "PendingInput",
    "Permissions",
    "Proceed",
    "Prompt",
    "Recover",
    "ResumeInput",
    "Steering",
    "StopReason",
    "Suspend",
    "Suspended",
    "Suspending",
    "__version__",
    "gate",
    "new_branch_id",
    "tool",
    "writer",
]
