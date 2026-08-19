"""Nexora's public Python package."""

from importlib.metadata import PackageNotFoundError, version

from nexora_store import ExecutionContext, MemorySteps

from .builtins import BuiltinTools, ExecToolOptions, builtin_tools
from .contracts import Agent, AgentDefinition, ObservationEventSink, PendingInput, ToolCall, Tools
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
from .engines.plain import react_loop
from .goal import Goal, goal_complete, goal_gate
from .orchestration import DurableRuntimeOrchestrator
from .orchestrator import ModelFailurePolicy
from .plan_mode import PlanMode, plan_mode_exit, plan_mode_gate
from .prompts import SystemPrompt, prompt_section, volatile_prompt_section
from .providers import FallbackChatModel, ModelProvider
from .runtime import AgentRuntime, run
from .sandbox_remote import RemoteSandboxClient
from .skills import DirectorySkillSource, SkillRegistry, SkillTools
from .subagents import (
    Answering,
    Authority,
    FactoryAgent,
    HttpAgent,
    RunnerAgent,
    Subagent,
    Subagents,
)
from .tool_search import DeferredTools
from .workspace import HostWorkspaceProvider, WorkspaceProvider

try:
    __version__ = version("nexora")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "Agent",
    "AgentDefinition",
    "AgentRuntime",
    "Answering",
    "Authority",
    "BuiltinTools",
    "Continue",
    "ControlPlane",
    "Controls",
    "Ctx",
    "DeferredTools",
    "Deny",
    "DirectorySkillSource",
    "DurableRuntimeOrchestrator",
    "ExecToolOptions",
    "ExecutionContext",
    "FactoryAgent",
    "FallbackChatModel",
    "FinishPolicy",
    "Goal",
    "Halt",
    "HostWorkspaceProvider",
    "HttpAgent",
    "Ingress",
    "MemorySteps",
    "ModelFailurePolicy",
    "ModelProvider",
    "ObservationEventSink",
    "PendingInput",
    "Permissions",
    "PlanMode",
    "Proceed",
    "RemoteSandboxClient",
    "ResumeInput",
    "RunnerAgent",
    "SkillRegistry",
    "SkillTools",
    "Subagent",
    "Subagents",
    "Suspend",
    "SystemPrompt",
    "ToolCall",
    "Tools",
    "WorkspaceProvider",
    "__version__",
    "builtin_tools",
    "gate",
    "goal_complete",
    "goal_gate",
    "plan_mode_exit",
    "plan_mode_gate",
    "prompt_section",
    "react_loop",
    "run",
    "volatile_prompt_section",
]
