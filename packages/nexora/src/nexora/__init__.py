"""Nexora's public Python package."""

from importlib.metadata import PackageNotFoundError, version

from .background import BackgroundResult, BackgroundTasks
from .contracts import (
    BaseMessage,
    CompactContext,
    ModelErrorKind,
    ModelFailure,
    ModelFailureAction,
    OnModelFailure,
    PendingInput,
    ToolCall,
    Tools,
)
from .engines.plain import react_loop
from .orchestrator import ModelFailurePolicy
from .runtime import AgentRuntime, run
from .subagents import (
    Answering,
    Authority,
    Compiled,
    Declarative,
    Remote,
    Subagent,
    Subagents,
)

try:
    __version__ = version("nexora")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "AgentRuntime",
    "Answering",
    "Authority",
    "BackgroundResult",
    "BackgroundTasks",
    "BaseMessage",
    "CompactContext",
    "Compiled",
    "Declarative",
    "ModelErrorKind",
    "ModelFailure",
    "ModelFailureAction",
    "ModelFailurePolicy",
    "OnModelFailure",
    "PendingInput",
    "Remote",
    "Subagent",
    "Subagents",
    "ToolCall",
    "Tools",
    "__version__",
    "react_loop",
    "run",
]
