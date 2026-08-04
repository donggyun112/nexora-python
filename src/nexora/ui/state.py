"""Process-local state for the test console."""

from dataclasses import dataclass, field

from ..controls import Controls
from ..orchestrator import MemorySteps
from .tools import DemoTools


@dataclass(slots=True)
class Session:
    tools: DemoTools = field(default_factory=DemoTools)
    controls: Controls | None = None


@dataclass(slots=True)
class RuntimeState:
    step_store: MemorySteps = field(default_factory=MemorySteps)
    sessions: dict[str, Session] = field(default_factory=dict)

    def session(self, run_id: str) -> Session:
        return self.sessions.setdefault(run_id, Session())


STATE = RuntimeState()
