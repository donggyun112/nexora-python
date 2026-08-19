"""Define the runtime event vocabulary and observation-only event stream.

Events report what happened but never make control-flow decisions. Delivery is at most once;
durable, load-bearing records belong at control points instead.
"""

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from loguru import logger


class EventType(StrEnum):
    """Enumerate runtime lifecycle and observation event names."""
    # Tools
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"

    # Permission
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_DENIED = "permission_denied"
    TOOL_REQUEST_CANCELLED = "tool_request_cancelled"

    # Prompt
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    USER_PROMPT_EXPANSION = "user_prompt_expansion"
    CONTEXT_INJECTED = "context_injected"

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


BLOCKING: Mapping[EventType, str] = MappingProxyType(
    {
        EventType.USER_PROMPT_SUBMIT: "on_inputs",
        EventType.PRE_TOOL_USE: "pre_tool_use",
        EventType.PERMISSION_REQUEST: "on_resume",
    }
)
"""Map events that announce decisions to their corresponding control points."""


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Carry one event and its stream-local identity coordinates."""
    event_type: EventType
    session_id: str
    thread_id: str
    run_id: str
    sequence: int
    payload: dict[str, Any]
    turn_id: str | None = None
    event_id: str = field(default="")

    def __post_init__(self) -> None:
        """Derive an event identifier when the caller did not provide one."""
        if not self.event_id:
            object.__setattr__(self, "event_id", self._derive_id())

    def _derive_id(self) -> str:
        """Derive a stable identifier from coordinates within this event stream."""
        parts = (
            self.run_id,
            self.turn_id or "",
            str(self.sequence),
            str(self.event_type),
            str(self.payload.get("call_id", "")),
        )
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


Sink = Callable[[EventEnvelope], Awaitable[None]]
"""Observe an event without changing runtime control flow."""

ObservationEventSink = Sink
"""Public name for a best-effort UI, logging, or monitoring destination."""

Publisher = Callable[[str, dict[str, Any]], Awaitable[Any]]


class RuntimeEvents:
    """Publish typed runtime and host lifecycle observations.

    Runtime-owned events are emitted by the loop and orchestrator. Host-owned lifecycle methods
    are exposed for integrations through ``AgentRuntime.events`` and ``Orchestrator.events``.
    """

    def __init__(self, emit: Publisher | None) -> None:
        """Initialize the facade with an optional publisher."""
        self._emit = emit

    async def publish(self, event_type: EventType, **payload: Any) -> None:
        """Publish an event when a publisher is configured."""
        if self._emit is not None:
            await self._emit(event_type, payload)

    async def session_start(self, source: str) -> None:
        """Publish a session-start event."""
        await self.publish(EventType.SESSION_START, source=source)

    async def session_end(self, reason: str) -> None:
        """Publish a session-end event."""
        await self.publish(EventType.SESSION_END, reason=reason)

    async def setup(self, **details: Any) -> None:
        """Publish runtime setup details."""
        await self.publish(EventType.SETUP, **details)

    async def config_change(self, changes: dict[str, Any]) -> None:
        """Publish a configuration change."""
        await self.publish(EventType.CONFIG_CHANGE, changes=changes)

    async def cwd_changed(self, previous: str, current: str) -> None:
        """Publish a working-directory change."""
        await self.publish(EventType.CWD_CHANGED, previous=previous, current=current)

    async def instructions_loaded(self, paths: list[str]) -> None:
        """Publish the loaded instruction paths."""
        await self.publish(EventType.INSTRUCTIONS_LOADED, paths=paths)

    async def pre_compact(self, reason: str) -> None:
        """Publish the start of context compaction."""
        await self.publish(EventType.PRE_COMPACT, reason=reason)

    async def post_compact(self, reason: str) -> None:
        """Publish the completion of context compaction."""
        await self.publish(EventType.POST_COMPACT, reason=reason)

    async def subagent_start(self, agent_id: str, task: str = "") -> None:
        """Publish a subagent-start event."""
        await self.publish(EventType.SUBAGENT_START, agent_id=agent_id, task=task)

    async def subagent_stop(self, agent_id: str, reason: str) -> None:
        """Publish a subagent-stop event."""
        await self.publish(EventType.SUBAGENT_STOP, agent_id=agent_id, reason=reason)

    async def task_created(self, task_id: str, description: str = "") -> None:
        """Publish a task-created event."""
        await self.publish(EventType.TASK_CREATED, task_id=task_id, description=description)

    async def task_completed(self, task_id: str, outcome: Any = None) -> None:
        """Publish a task-completed event."""
        await self.publish(EventType.TASK_COMPLETED, task_id=task_id, outcome=outcome)

    async def teammate_idle(self, teammate_id: str) -> None:
        """Publish a teammate-idle event."""
        await self.publish(EventType.TEAMMATE_IDLE, teammate_id=teammate_id)

    async def elicitation(self, request_id: str, request: Any) -> None:
        """Publish an elicitation request."""
        await self.publish(EventType.ELICITATION, request_id=request_id, request=request)

    async def elicitation_result(self, request_id: str, result: Any) -> None:
        """Publish an elicitation result."""
        await self.publish(EventType.ELICITATION_RESULT, request_id=request_id, result=result)

    async def notification(self, message: str, *, level: str = "info") -> None:
        """Publish a user-facing notification."""
        await self.publish(EventType.NOTIFICATION, message=message, level=level)


class EventStream:
    """Number a run's events and deliver them to an observation sink."""

    def __init__(self, sink: Sink, *, session_id: str, thread_id: str, run_id: str) -> None:
        """Initialize stream identity and its sequence counter."""
        self._sink = sink
        self._session_id = session_id
        self._thread_id = thread_id
        self._run_id = run_id
        self._sequence = 0

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        """Envelope and deliver one event, logging sink failures."""
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
            logger.exception("event sink failed: {} ({})", envelope.event_type, envelope.event_id)
