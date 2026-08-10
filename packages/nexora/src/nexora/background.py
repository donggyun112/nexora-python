"""Process-local registry and result types for agent-launched background work."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "BackgroundResult",
    "BackgroundTask",
    "BackgroundTasks",
    "TaskStatus",
]

TaskStatus = Literal["running", "done", "error", "cancelled"]


@dataclass(frozen=True, slots=True)
class BackgroundResult:
    """Result delivered when managed background work settles."""

    task_id: str
    kind: str
    label: str
    content: str
    is_error: bool = False

    def as_message(self) -> str:
        """Render the result as a model-visible background notification."""
        state = "failed" if self.is_error else "completed"
        header = f'[background {self.kind} "{self.label}" {state}] (task {self.task_id})'
        return f"{header}\n{self.content}"


@dataclass(slots=True)
class BackgroundTask:
    """Registered background task and its observable lifecycle state."""

    task_id: str
    kind: str
    label: str
    started_at: float
    task: asyncio.Task[Any]
    status: TaskStatus = "running"
    settled_at: float | None = None
    read_output: Callable[[], str] | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable task status snapshot."""
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            **({"settled_at": self.settled_at} if self.settled_at is not None else {}),
        }


@dataclass
class BackgroundTasks:
    """Manage process-local background tasks launched by one parent run."""

    max_settled_retained: int = 50

    _tasks: dict[str, BackgroundTask] = field(default_factory=dict, repr=False)
    _listeners: list[Callable[[str, TaskStatus], None]] = field(default_factory=list, repr=False)

    def register(
        self,
        task_id: str,
        kind: str,
        label: str,
        task: asyncio.Task[Any],
        *,
        read_output: Callable[[], str] | None = None,
    ) -> BackgroundTask:
        """Register a launched task without notifying settlement listeners."""
        entry = BackgroundTask(
            task_id=task_id,
            kind=kind,
            label=label,
            started_at=time.time(),
            task=task,
            read_output=read_output,
        )
        self._tasks[task_id] = entry
        return entry

    def settle(self, task_id: str, status: TaskStatus) -> None:
        """Record the first terminal status for a task."""
        entry = self._tasks.get(task_id)
        if entry is None or entry.status != "running":
            return
        entry.status = status
        entry.settled_at = time.time()
        self._prune()
        self._notify(task_id, status)

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task and report whether cancellation was applied."""
        entry = self._tasks.get(task_id)
        if entry is None or entry.status != "running":
            return False
        entry.status = "cancelled"
        entry.settled_at = time.time()
        entry.task.cancel()
        self._prune()
        self._notify(task_id, "cancelled")
        return True

    def cancel_all(self) -> int:
        """Cancel all running tasks and return the number cancelled."""
        return sum(self.cancel(task_id) for task_id in list(self._tasks))

    def get(self, task_id: str) -> BackgroundTask | None:
        """Return a task record by identifier."""
        return self._tasks.get(task_id)

    def status(self, task_id: str) -> TaskStatus | None:
        """Return a task status by identifier."""
        entry = self._tasks.get(task_id)
        return entry.status if entry is not None else None

    def list(self) -> list[dict[str, Any]]:
        """Return task snapshots in launch order."""
        return [
            entry.snapshot()
            for entry in sorted(self._tasks.values(), key=lambda item: item.started_at)
        ]

    def subscribe(self, listener: Callable[[str, TaskStatus], None]) -> Callable[[], None]:
        """Subscribe to task settlements and return an unsubscribe callback."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify(self, task_id: str, status: TaskStatus) -> None:
        for listener in list(self._listeners):
            try:
                listener(task_id, status)
            except Exception:  # one bad listener must not break settling
                continue

    def _prune(self) -> None:
        settled = sorted(
            (item for item in self._tasks.values() if item.status != "running"),
            key=lambda item: item.settled_at or 0.0,
        )
        for entry in settled[: max(0, len(settled) - self.max_settled_retained)]:
            del self._tasks[entry.task_id]
