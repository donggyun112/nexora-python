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
from .providers import FallbackChatModel, ModelProvider, ProviderErrorKind
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
from .workspace import (
    CommandResult,
    HostWorkspaceProvider,
    ResolvedWorkspacePath,
    SandboxCommand,
    SnapshotBackend,
    TarSnapshotBackend,
    WorkspaceAccessMode,
    WorkspaceProvider,
    WorkspaceSession,
    WorkspaceSnapshot,
    WorkspaceViolation,
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
    "CommandResult",
    "CompactContext",
    "Compiled",
    "Declarative",
    "FallbackChatModel",
    "HostWorkspaceProvider",
    "ModelErrorKind",
    "ModelFailure",
    "ModelFailureAction",
    "ModelFailurePolicy",
    "ModelProvider",
    "OnModelFailure",
    "PendingInput",
    "ProviderErrorKind",
    "Remote",
    "ResolvedWorkspacePath",
    "SandboxCommand",
    "SnapshotBackend",
    "Subagent",
    "Subagents",
    "TarSnapshotBackend",
    "ToolCall",
    "Tools",
    "WorkspaceAccessMode",
    "WorkspaceProvider",
    "WorkspaceSession",
    "WorkspaceSnapshot",
    "WorkspaceViolation",
    "__version__",
    "react_loop",
    "run",
]
