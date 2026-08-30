"""Semora's public Python package."""

from importlib.metadata import PackageNotFoundError, version

from semora_llm import ChatModel
from semora_store import ExecutionContext, MemorySteps

from .contracts import Agent, AgentDefinition, PendingInput, ToolCall, Tools
from .controls import (
    Continue,
    ControlPlane,
    Controls,
    Ctx,
    Deny,
    FinishPolicy,
    Halt,
    Ingress,
    Permissions,
    Proceed,
    ResumeInput,
    Suspend,
    gate,
)
from .ids import new_run_id
from .runtime import AgentRuntime, run
from .workspace import HostWorkspaceProvider

try:
    __version__ = version("semora")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "Agent",
    "AgentDefinition",
    "AgentRuntime",
    "ChatModel",
    "Continue",
    "ControlPlane",
    "Controls",
    "Ctx",
    "Deny",
    "ExecutionContext",
    "FinishPolicy",
    "Halt",
    "HostWorkspaceProvider",
    "Ingress",
    "MemorySteps",
    "PendingInput",
    "Permissions",
    "Proceed",
    "ResumeInput",
    "Suspend",
    "ToolCall",
    "Tools",
    "__version__",
    "gate",
    "new_run_id",
    "run",
]
