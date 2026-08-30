"""Fork a conversation from just before one injected input.

The core keeps the pieces apart on purpose: the ledger holds each input's pre-screen
original, the transcript rewinds by preserved branch, and controls arrive as call
arguments. Forking is therefore composition, not a runtime feature — this package walks
those seams and adds no authority of its own. The source run's ledger is never touched:
what actually went out stays the record.
"""

from datetime import UTC, datetime
from typing import Any, NamedTuple

from langchain_core.language_models import BaseChatModel
from semora import AgentRuntime
from semora.contracts import Agent, Tools
from semora.controls import Controls
from semora.history import decode_pending_input
from semora.transcript import SCHEMA_VERSION, entry_id, messages_at
from semora_store import ExecutionStore, Transcript

__all__ = [
    "EventCheckpoint",
    "ForkCoordinate",
    "fork_event",
    "fork_run",
    "read_event_checkpoint",
    "record_event_checkpoint",
]


class ForkCoordinate(NamedTuple):
    """One replayable position attached to an observation edge."""

    from_run_id: str
    origin_id: str | None
    leaf_uuid: str | None


class EventCheckpoint(NamedTuple):
    """Durable before/after transcript coordinates for one observation."""

    event_id: str
    conversation_id: str
    before: ForkCoordinate
    after: ForkCoordinate


def _coordinate_payload(coordinate: ForkCoordinate) -> dict[str, str | None]:
    return {
        "from_run_id": coordinate.from_run_id,
        "origin_id": coordinate.origin_id,
        "leaf_uuid": coordinate.leaf_uuid,
    }


def _decode_coordinate(payload: object) -> ForkCoordinate:
    if not isinstance(payload, dict):
        raise TypeError("fork checkpoint coordinate must be a mapping")
    from_run_id = payload.get("from_run_id")
    origin_id = payload.get("origin_id")
    leaf_uuid = payload.get("leaf_uuid")
    if not isinstance(from_run_id, str) or not from_run_id:
        raise TypeError("fork checkpoint coordinate has no source run")
    if origin_id is not None and not isinstance(origin_id, str):
        raise TypeError("fork checkpoint origin must be a string or None")
    if leaf_uuid is not None and not isinstance(leaf_uuid, str):
        raise TypeError("fork checkpoint leaf must be a string or None")
    return ForkCoordinate(from_run_id, origin_id, leaf_uuid)


async def record_event_checkpoint(
    transcript: Transcript,
    checkpoint: EventCheckpoint,
) -> bool:
    """Append one immutable event-to-state mapping to the conversation transcript."""
    body = {
        "type": "fork_checkpoint",
        "event_id": checkpoint.event_id,
        "before": _coordinate_payload(checkpoint.before),
        "after": _coordinate_payload(checkpoint.after),
    }
    return await transcript.append(
        {
            "uuid": entry_id(None, body),
            "conversation_id": checkpoint.conversation_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "schema_version": SCHEMA_VERSION,
            **body,
        }
    )


async def read_event_checkpoint(
    transcript: Transcript,
    conversation_id: str,
    event_id: str,
) -> EventCheckpoint:
    """Read the newest durable mapping for one event identity."""
    entries = await transcript.read(conversation_id)
    entry = next(
        (
            item
            for item in reversed(entries)
            if item.get("type") == "fork_checkpoint" and item.get("event_id") == event_id
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"conversation {conversation_id!r} has no fork checkpoint {event_id!r}")
    return EventCheckpoint(
        event_id,
        conversation_id,
        _decode_coordinate(entry.get("before")),
        _decode_coordinate(entry.get("after")),
    )


async def fork_run(
    runtime: AgentRuntime,
    store: ExecutionStore,
    *,
    from_run_id: str,
    origin_id: str,
    run_id: str,
    model: BaseChatModel | Agent,
    tools: Tools | str | None = None,
    controls: Controls | None = None,
    conversation_id: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Re-run the conversation from just before ``origin_id`` entered model context.

    The pre-screen original is read from ``from_run_id``'s ledger and enqueued for the new
    ``run_id``, so it passes through whatever ``controls`` this call supplies — the fork
    screens, records, and announces it like any live input. The rewind is
    ``run(history=...)``'s own branch-preserving replace: the source branch stays
    observable, the conversation head moves to the fork.

    Raises ``ValueError`` when the ledger has no such input or it never reached model
    context — either way there is no injection point to fork from.
    """
    records = await store.list_inputs(from_run_id)
    record = next((r for r in records if r.input_id == origin_id), None)
    if record is None:
        raise ValueError(f"run {from_run_id!r} has no ledger record for input {origin_id!r}")
    original = decode_pending_input(record.value)

    conversation = conversation_id or from_run_id
    history = await runtime.committed_history(from_run_id, conversation)
    cut = next((i for i, message in enumerate(history) if message.id == origin_id), None)
    if cut is None:
        raise ValueError(f"input {origin_id!r} never entered the model context of {conversation!r}")

    await runtime.submit(run_id, original, conversation_id=conversation)
    return await runtime.run(
        run_id,
        model,
        tools,
        controls=controls,
        conversation_id=conversation,
        history=list(history[:cut]),
        **options,
    )


async def fork_event(
    runtime: AgentRuntime,
    store: ExecutionStore,
    transcript: Transcript,
    *,
    event_id: str,
    edge: str,
    run_id: str,
    model: BaseChatModel | Agent,
    tools: Tools | str | None = None,
    controls: Controls | None = None,
    conversation_id: str,
    **options: Any,
) -> dict[str, Any]:
    """Fork from the durable coordinate attached to one observation edge.

    A before edge with an input origin reuses :func:`fork_run`, so the source ledger's
    pre-screen original crosses the new controls. Other coordinates continue from their
    explicit transcript leaf without copying source-run effect records.
    """
    if edge not in {"before", "after"}:
        raise ValueError("edge must be 'before' or 'after'")
    checkpoint = await read_event_checkpoint(transcript, conversation_id, event_id)
    coordinate = checkpoint.before if edge == "before" else checkpoint.after
    if edge == "before" and coordinate.origin_id is not None:
        return await fork_run(
            runtime,
            store,
            from_run_id=coordinate.from_run_id,
            origin_id=coordinate.origin_id,
            run_id=run_id,
            model=model,
            tools=tools,
            controls=controls,
            conversation_id=checkpoint.conversation_id,
            **options,
        )

    entries = await transcript.read(checkpoint.conversation_id)
    history = messages_at(entries, coordinate.leaf_uuid)
    return await runtime.run(
        run_id,
        model,
        tools,
        controls=controls,
        conversation_id=checkpoint.conversation_id,
        history=history,
        **options,
    )
