"""Nexora's public Python package."""

from importlib.metadata import PackageNotFoundError, version

from .background import BackgroundResult, BackgroundTasks
from .builtins import (
    BuiltinTools,
    ExecToolOptions,
    UrllibWebFetchTransport,
    WebFetchResponse,
    WebFetchSummarizer,
    WebFetchToolOptions,
    WebFetchTransport,
    builtin_tools,
)
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
from .sandbox_remote import (
    HTTPResponse,
    HTTPTransport,
    RemoteSandboxClient,
    RemoteSandboxError,
    UrllibHTTPTransport,
)
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
    ContextualTools,
    ContinuousWorkspaceProvider,
    HostWorkspaceProvider,
    MemoryWorkspaceStateStore,
    ResolvedWorkspacePath,
    ResumableWorkspaceProvider,
    SandboxCommand,
    SandboxSessionState,
    SnapshotBackend,
    TarSnapshotBackend,
    ToolContext,
    WorkspaceAccessMode,
    WorkspaceDirEntry,
    WorkspaceFileStat,
    WorkspaceFS,
    WorkspaceProvider,
    WorkspaceSeed,
    WorkspaceSession,
    WorkspaceSnapshot,
    WorkspaceStateStore,
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
    "BuiltinTools",
    "CommandResult",
    "CompactContext",
    "Compiled",
    "ContextualTools",
    "ContinuousWorkspaceProvider",
    "Declarative",
    "ExecToolOptions",
    "FallbackChatModel",
    "HTTPResponse",
    "HTTPTransport",
    "HostWorkspaceProvider",
    "MemoryWorkspaceStateStore",
    "ModelErrorKind",
    "ModelFailure",
    "ModelFailureAction",
    "ModelFailurePolicy",
    "ModelProvider",
    "OnModelFailure",
    "PendingInput",
    "ProviderErrorKind",
    "Remote",
    "RemoteSandboxClient",
    "RemoteSandboxError",
    "ResolvedWorkspacePath",
    "ResumableWorkspaceProvider",
    "SandboxCommand",
    "SandboxSessionState",
    "SnapshotBackend",
    "Subagent",
    "Subagents",
    "TarSnapshotBackend",
    "ToolCall",
    "ToolContext",
    "Tools",
    "UrllibHTTPTransport",
    "UrllibWebFetchTransport",
    "WebFetchResponse",
    "WebFetchSummarizer",
    "WebFetchToolOptions",
    "WebFetchTransport",
    "WorkspaceAccessMode",
    "WorkspaceDirEntry",
    "WorkspaceFS",
    "WorkspaceFileStat",
    "WorkspaceProvider",
    "WorkspaceSeed",
    "WorkspaceSession",
    "WorkspaceSnapshot",
    "WorkspaceStateStore",
    "WorkspaceViolation",
    "__version__",
    "builtin_tools",
    "react_loop",
    "run",
]
