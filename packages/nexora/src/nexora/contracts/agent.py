"""Agent-definition identity and the executable local definition."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from .types import SystemPromptSource, Tools

__all__ = ["Agent", "AgentDefinition"]


@runtime_checkable
class AgentDefinition(Protocol):
    """Common identity shared by local, declarative, compiled, and remote agents."""

    @property
    def name(self) -> str:
        """Return the stable name used for selection and lifecycle events."""
        ...

    @property
    def description(self) -> str:
        """Describe when another agent should delegate work here."""
        ...


@dataclass(frozen=True, slots=True)
class Agent:
    """Bind the model-visible identity and executable tools of one local agent.

    Prompts submitted by users, conversation history, controls, workspaces, and orchestration are
    attempt policy or state. They deliberately stay on ``AgentRuntime`` rather than this definition.
    """

    name: str
    description: str
    model: BaseChatModel
    tools: Tools
    system_prompt: str | SystemPromptSource | None = None
