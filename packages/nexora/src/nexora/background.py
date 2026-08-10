"""Detached work an agent launched, and the leash it keeps on it.

Tool-neutral, the way `contracts/background-task.ts` is in the TypeScript reference: nothing here
knows what a subagent is. Any tool that starts work outliving its own call registers it, and the
agent observes it (`check_tasks`), reads it (`read_task_output`), cancels it (`cancel_task`) or
waits on it (`watch_task`) through the same four tools.

Python's `asyncio.Task` already is most of a job record — it cancels, it reports done, it holds the
exception. What it cannot say is whether the *work* succeeded, because an agent that finishes by
reporting a failure is a task that completed normally. So the outcome is recorded by whoever pumped
the task (`settle`) and everything else is read off the `Task`.
"""

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
    """What settled work reports back, once the thing that launched it is gone.

    `kind` names the launching tool family (`subagent`, and whatever comes after it); `label` is
    what a person reads in `check_tasks`. Neither is interpreted here.
    """

    task_id: str
    kind: str
    label: str
    content: str
    is_error: bool = False

    def as_message(self) -> str:
        """The one rendering of a settled task, so every delivery path reads alike."""
        state = "failed" if self.is_error else "completed"
        header = f'[background {self.kind} "{self.label}" {state}] (task {self.task_id})'
        return f"{header}\n{self.content}"


@dataclass(slots=True)
class BackgroundTask:
    """One registered job. `status` is the outcome; the `Task` is the live handle."""

    task_id: str
    kind: str
    label: str
    started_at: float
    task: asyncio.Task[Any]
    status: TaskStatus = "running"
    settled_at: float | None = None
    read_output: Callable[[], str] | None = None
    """Live view of whatever the job is producing, for jobs that stream. Absent for jobs that
    only have a final answer, which is most of them."""

    def snapshot(self) -> dict[str, Any]:
        """The task as `check_tasks` reports it — no handles, nothing live."""
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
    """Every job one parent run launched, oldest first.

    Not durable, and deliberately so. A background job is an in-flight coroutine in this process;
    a record of it surviving a crash the coroutine did not would describe something that is no
    longer running. What *is* durable is the result, once it settles and reaches the input queue.
    """

    max_settled_retained: int = 50
    """Settled jobs are pruned oldest-first past this, so a long run's `check_tasks` stays
    readable and the dict stays bounded. Running jobs are never evicted."""

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
        """Take a launched job under management. Registering fires no listener."""
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
        """Record an outcome, unless the job already has one.

        A no-op on an already-settled job, which is what makes `cancel` and a pump racing to
        finish safe: cancellation wins because it landed first, and the pump's later `settle`
        does not overwrite it with `done`.
        """
        entry = self._tasks.get(task_id)
        if entry is None or entry.status != "running":
            return
        entry.status = status
        entry.settled_at = time.time()
        self._prune()
        self._notify(task_id, status)

    def cancel(self, task_id: str) -> bool:
        """Cut a running job off. False when there is nothing running under that id."""
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
        """Cut off everything still running, and report how many that was.

        For a host tearing a run down: an abandoned child keeps burning tokens exactly as long as
        nobody stops it, and `asyncio` will only complain about the orphan after the fact.
        """
        return sum(self.cancel(task_id) for task_id in list(self._tasks))

    def get(self, task_id: str) -> BackgroundTask | None:
        """The live record, or None."""
        return self._tasks.get(task_id)

    def status(self, task_id: str) -> TaskStatus | None:
        """The outcome of one job, or None when no such job is known."""
        entry = self._tasks.get(task_id)
        return entry.status if entry is not None else None

    def list(self) -> list[dict[str, Any]]:
        """Every known job as a snapshot, in launch order."""
        return [
            entry.snapshot()
            for entry in sorted(self._tasks.values(), key=lambda item: item.started_at)
        ]

    def subscribe(self, listener: Callable[[str, TaskStatus], None]) -> Callable[[], None]:
        """Hear about settlements. Returns the unsubscribe.

        Fires after the status is already updated, so a listener that asks the registry a question
        gets the answer that caused the notification rather than the one before it.
        """
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
