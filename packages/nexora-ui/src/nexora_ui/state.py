"""Process-local state and fault injection for the test console."""

import json
from dataclasses import dataclass, field
from typing import Any

from nexora.contracts import BaseMessage
from nexora.controls import Controls
from nexora.orchestrator import MemorySteps

from .tools import DemoTools


class SimulatedWorkerCrash(RuntimeError):
    """Simulate a worker disappearing after an effect result commits."""

    def __init__(self, run_id: str, step: str) -> None:
        """Initialize the failure for a committed run step."""
        super().__init__(f"simulated worker crash after committed step {step!r}")
        self.run_id = run_id
        self.step = step


def _preview(value: Any) -> str:
    """Render a compact single-line value preview."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 90 else f"{text[:87]}…"


class FaultInjectingMemorySteps(MemorySteps):
    """Inject a simulated worker crash after a tool step commits."""

    def __init__(self) -> None:
        """Initialize an unarmed in-memory ledger."""
        super().__init__()
        self._crash_after_step: set[str] = set()

    def arm(self, run_id: str) -> None:
        """Arm fault injection for the next tool step in a run."""
        self._crash_after_step.add(run_id)

    def disarm(self, run_id: str) -> None:
        """Disable fault injection for a run."""
        self._crash_after_step.discard(run_id)

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        """Commit a result and raise the armed simulated crash."""
        await super().finish(run_id, key, value, token)
        if run_id in self._crash_after_step and not key.startswith(
            ("agent:", "signal:", "suspend:")
        ):
            self._crash_after_step.remove(run_id)
            raise SimulatedWorkerCrash(run_id, key)

    def snapshot(self, run_id: str) -> list[dict[str, Any]]:
        """Return compact step records for the console's ledger panel."""
        return [
            {"key": key, "status": step.status, "value": _preview(step.value)}
            for (owner, key), step in self._entries.items()
            if owner == run_id
        ]


@dataclass(slots=True)
class Session:
    """Store process-local dependencies and recovery state for one run."""
    tools: DemoTools = field(default_factory=DemoTools)
    controls: Controls | None = None
    recovery_history: list[BaseMessage] | None = None


@dataclass(slots=True)
class RuntimeState:
    """Store process-local sessions and the shared demo ledger."""
    step_store: FaultInjectingMemorySteps = field(default_factory=FaultInjectingMemorySteps)
    sessions: dict[str, Session] = field(default_factory=dict)

    def session(self, run_id: str) -> Session:
        """Return an existing session or create one for ``run_id``."""
        return self.sessions.setdefault(run_id, Session())


STATE = RuntimeState()
