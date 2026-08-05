"""The event contract. Observation, and only observation.

Nothing here decides anything. A decision has to be a value the caller holds so it can pick the
next step, and the order of those steps is the whole design — "the bypass entry sits *after* the
deny rules" is an invariant you can only state in a call chain. Published events arrive whenever
a subscriber gets around to them, so a permission answer cannot live on this channel; it lives in
`nexora.controls.Permissions`, which is an ordered chain of calls.

What this channel is for: telling a UI what happened, and writing an audit trail. A failing sink
is logged and skipped — an audit socket having a bad moment is not a reason to kill an agent
mid-run. That rule is only safe *because* nothing load-bearing rides here.

`BLOCKING` maps each event that announces a decision point to the `Controls` method that makes
that decision. The decision is made by that call, never by publishing the event; the mapping
exists so a subscriber can tell "you are being told about a gate" from "you are being told about
a result" — and so a declared gate that no control point implements fails a test instead of
reading as a guarantee.

`EventEnvelope.event_id` is derived, never random. A run that crashes and resumes re-emits the
events of rounds it had already finished; a derived id lets an outbox drop the duplicates.
"""

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from loguru import logger


class EventType(StrEnum):
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
"""Decision points: the announcing event, mapped to the `Controls` method that decides it.

The event is published as a courtesy — after the decision, or beside it when the decision is a
park waiting on a person. Answering one of these on the event channel does nothing, on purpose: an
ordered short-circuit cannot be expressed by subscribers, and a permission model whose precedence
depends on dispatch order is not a permission model.

A mapping and not a set, because membership was the only claim a set could make and it made a
false one: `PRE_COMPACT` and `ELICITATION` were listed here while `Controls` had no method for
either, so the contract advertised a gate that did not exist. Naming the control point makes the
pairing checkable — `test_blocking_events_each_name_a_control_point_that_exists` resolves every
value against the protocol. Compaction and elicitation come back to this table on the day their
control point does.
"""


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
"""An observer. Told what happened; cannot change it."""

Publisher = Callable[[str, dict[str, Any]], Awaitable[Any]]


class RuntimeEvents:
    """Typed emission points owned by the session/orchestration host.

    The agent engines emit prompt, tool, permission, and stop events themselves. They cannot know
    that a configuration file changed, a compaction began, or a background teammate became idle.
    The component performing those actions calls this facade at the real boundary instead of
    pretending every agent loop invocation is a new session.

    These methods publish observations, and that is all they do. A load-bearing decision uses
    `Controls`, so where no `Controls` method exists there is no decision point: `pre_compact` and
    `elicitation` announce work this codebase has no gate for. Publishing them does not create one
    — the control point and the `BLOCKING` entry do, together.
    """

    def __init__(self, emit: Publisher | None) -> None:
        self._emit = emit

    async def publish(self, event_type: EventType, **payload: Any) -> None:
        if self._emit is not None:
            await self._emit(event_type, payload)

    async def session_start(self, source: str) -> None:
        await self.publish(EventType.SESSION_START, source=source)

    async def session_end(self, reason: str) -> None:
        await self.publish(EventType.SESSION_END, reason=reason)

    async def setup(self, **details: Any) -> None:
        await self.publish(EventType.SETUP, **details)

    async def config_change(self, changes: dict[str, Any]) -> None:
        await self.publish(EventType.CONFIG_CHANGE, changes=changes)

    async def cwd_changed(self, previous: str, current: str) -> None:
        await self.publish(EventType.CWD_CHANGED, previous=previous, current=current)

    async def instructions_loaded(self, paths: list[str]) -> None:
        await self.publish(EventType.INSTRUCTIONS_LOADED, paths=paths)

    async def pre_compact(self, reason: str) -> None:
        await self.publish(EventType.PRE_COMPACT, reason=reason)

    async def post_compact(self, reason: str) -> None:
        await self.publish(EventType.POST_COMPACT, reason=reason)

    async def subagent_start(self, agent_id: str, task: str = "") -> None:
        await self.publish(EventType.SUBAGENT_START, agent_id=agent_id, task=task)

    async def subagent_stop(self, agent_id: str, reason: str) -> None:
        await self.publish(EventType.SUBAGENT_STOP, agent_id=agent_id, reason=reason)

    async def task_created(self, task_id: str, description: str = "") -> None:
        await self.publish(EventType.TASK_CREATED, task_id=task_id, description=description)

    async def task_completed(self, task_id: str, outcome: Any = None) -> None:
        await self.publish(EventType.TASK_COMPLETED, task_id=task_id, outcome=outcome)

    async def teammate_idle(self, teammate_id: str) -> None:
        await self.publish(EventType.TEAMMATE_IDLE, teammate_id=teammate_id)

    async def elicitation(self, request_id: str, request: Any) -> None:
        await self.publish(EventType.ELICITATION, request_id=request_id, request=request)

    async def elicitation_result(self, request_id: str, result: Any) -> None:
        await self.publish(EventType.ELICITATION_RESULT, request_id=request_id, result=result)

    async def notification(self, message: str, *, level: str = "info") -> None:
        await self.publish(EventType.NOTIFICATION, message=message, level=level)


class EventStream:
    """Numbers a run's events and hands them to a sink.

    One per run — `sequence` is what makes `event_id` unique without randomness.

    A failing sink is logged and skipped, never raised. These events are observations: an audit
    log or a UI socket having a bad moment is not a reason to kill an agent mid-run. The cost is
    that a dropped event is only visible in the log, which is why `event_id` is derived — a
    durable sink can replay and dedupe rather than rely on delivery here.

    Anything that must not be silently dropped is therefore not an event. The durable record of a
    resolved tool call is `Permissions.record`, a call that raises through.
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
            logger.exception("event sink failed: {} ({})", envelope.event_type, envelope.event_id)
