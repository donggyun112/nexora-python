"""Process-local state and fault injection for the test console."""

from dataclasses import dataclass, field
from typing import Any

from nexora_contracts import BaseMessage
from nexora_contracts.controls import Controls
from nexora_orchestrator import MemorySteps

from .tools import DemoTools


class SimulatedWorkerCrash(RuntimeError):
    """The effect result is committed, but its worker disappears before returning it."""

    def __init__(self, run_id: str, step: str) -> None:
        super().__init__(f"simulated worker crash after committed step {step!r}")
        self.run_id = run_id
        self.step = step


class FaultInjectingMemorySteps(MemorySteps):
    """A demo ledger that can crash one worker immediately after a tool step commits."""

    def __init__(self) -> None:
        super().__init__()
        self._crash_after_step: set[str] = set()

    def arm(self, run_id: str) -> None:
        self._crash_after_step.add(run_id)

    def disarm(self, run_id: str) -> None:
        self._crash_after_step.discard(run_id)

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        await super().finish(run_id, key, value, token)
        if run_id in self._crash_after_step and not key.startswith(
            ("agent:", "signal:", "suspend:")
        ):
            self._crash_after_step.remove(run_id)
            raise SimulatedWorkerCrash(run_id, key)


@dataclass(slots=True)
class Session:
    tools: DemoTools = field(default_factory=DemoTools)
    controls: Controls | None = None
    recovery_history: list[BaseMessage] | None = None


@dataclass(slots=True)
class RuntimeState:
    step_store: FaultInjectingMemorySteps = field(default_factory=FaultInjectingMemorySteps)
    sessions: dict[str, Session] = field(default_factory=dict)

    def session(self, run_id: str) -> Session:
        return self.sessions.setdefault(run_id, Session())


STATE = RuntimeState()
