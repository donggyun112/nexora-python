"""The event contract.

Two kinds, and the difference is structural rather than cosmetic:

* **Blocking** — the handler's answer changes what happens next. These are not published; they
  are hooks the loop awaits. `PreToolUse` is the one the loop uses today, and it arrives as
  `before_tool_call` rather than as a subscription, because a return value that nobody reads
  is a permission check that silently passes.
* **Observing** — the handler is told something happened and cannot change it. These go
  through `emit`.

`EventEnvelope.event_id` is derived, never random. A run that crashes and resumes re-emits the
events of rounds it had already finished; a derived id lets an outbox drop the duplicates.
"""

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger


class EventType(StrEnum):
    # Tools
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"
    POST_TOOL_BATCH = "post_tool_batch"

    # Permission
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_DENIED = "permission_denied"

    # Prompt
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    USER_PROMPT_EXPANSION = "user_prompt_expansion"

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SETUP = "setup"
    CONFIG_CHANGE = "config_change"
    INSTRUCTIONS_LOADED = "instructions_loaded"
    CWD_CHANGED = "cwd_changed"
    DIRECTORY_ADDED = "directory_added"
    FILE_CHANGED = "file_changed"
    MESSAGE_DISPLAY = "message_display"

    # Agent stop / idle
    STOP = "stop"
    STOP_FAILURE = "stop_failure"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"
    TEAMMATE_IDLE = "teammate_idle"

    # Compaction
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"

    # Tasks
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"

    # Elicitation (MCP)
    ELICITATION = "elicitation"
    ELICITATION_RESULT = "elicitation_result"

    # Worktree
    WORKTREE_CREATE = "worktree_create"
    WORKTREE_REMOVE = "worktree_remove"

    # Notification
    NOTIFICATION = "notification"


BLOCKING: frozenset[EventType] = frozenset(
    {
        EventType.PRE_TOOL_USE,
        EventType.PERMISSION_REQUEST,
        EventType.ELICITATION,
        EventType.USER_PROMPT_SUBMIT,
        EventType.PRE_COMPACT,
    }
)
"""Events whose handler answer feeds back into the run, rather than merely observing it."""


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_type: EventType
    session_id: str
    thread_id: str
    run_id: str
    sequence: int
    payload: dict[str, Any]
    turn_id: str | None = None
    event_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", self._derive_id())

    def _derive_id(self) -> str:
        """Stable across a crash-and-resume: same coordinates, same id."""
        parts = (
            self.run_id,
            self.turn_id or "",
            str(self.sequence),
            str(self.event_type),
            str(self.payload.get("call_id", "")),
        )
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


Sink = Callable[[EventEnvelope], Awaitable[None]]


class EventStream:
    """Numbers a run's events and hands them to a sink.

    One per run — `sequence` is what makes `event_id` unique without randomness.

    A failing sink is logged and skipped, never raised. These events are observations: an
    audit log or a UI socket having a bad moment is not a reason to kill an agent mid-run. The
    cost is that a dropped event is only visible in the log, which is why `event_id` is
    derived — a durable sink can replay and dedupe rather than rely on delivery here.
    """

    def __init__(self, sink: Sink, *, session_id: str, thread_id: str, run_id: str) -> None:
        self._sink = sink
        self._session_id = session_id
        self._thread_id = thread_id
        self._run_id = run_id
        self._sequence = 0

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        envelope = EventEnvelope(
            event_type=EventType(event_type),
            session_id=self._session_id,
            thread_id=self._thread_id,
            run_id=self._run_id,
            sequence=self._sequence,
            payload=payload,
            turn_id=str(payload["turn"]) if "turn" in payload else None,
        )
        self._sequence += 1
        try:
            await self._sink(envelope)
        except Exception:
            logger.exception(
                "event sink failed: {} ({})", envelope.event_type, envelope.event_id
            )
