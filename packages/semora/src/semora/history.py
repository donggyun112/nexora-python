"""Serialization and message reconstruction for suspended agent runs.

The execution ledger stores these payloads opaquely; this module owns their LangChain message
semantics.
"""

from typing import Any, NamedTuple, cast

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from .contracts.types import BaseMessage, PendingInput, ToolCall
from .tools import render_for_model


def resume_after_suspend(
    snapshot: list[BaseMessage],
    call_id: str,
    result: dict[str, Any],
    *,
    name: str = "",
) -> list[BaseMessage]:
    """Append a suspended call's answer to its saved model history.

    Args:
        snapshot: History produced by ``suspend_history_snapshot``.
        call_id: Suspended tool-call identifier.
        result: Resumption result encoded as a tool result.
        name: Optional tool name.

    Returns:
        History ready to pass to the planner for continuation.
    """
    return [*snapshot, suspension_result_message(call_id, result, name=name)]


def suspension_result_message(
    call_id: str,
    result: dict[str, Any],
    *,
    name: str = "",
) -> ToolMessage:
    """The single queueable answer to the call that parked the run."""
    return ToolMessage(
        content=render_for_model(result),
        tool_call_id=call_id,
        name=name or None,
        status="error" if result.get("type") == "error" else "success",
    )


def cancelled_tool_inputs(
    snapshot: list[BaseMessage],
    *,
    reason: str = "cancelled by a newer user request",
) -> list[PendingInput]:
    """Close every unanswered tool call before a replacing HumanMessage may be admitted.

    The cancellation is a protocol answer, not an execution result. Completed calls already have
    ToolMessages in the snapshot and are preserved; only calls with no answer are closed.
    """
    answered = {
        message.tool_call_id for message in snapshot if isinstance(message, ToolMessage)
    }
    pending: list[ToolCall] = []
    for message in snapshot:
        if isinstance(message, AIMessage):
            pending.extend(call for call in message.tool_calls if call["id"] not in answered)
    return [
        PendingInput(
            "cancelled_tool_result",
            suspension_result_message(
                call["id"] or "",
                {"type": "error", "message": reason, "code": "cancelled"},
                name=call.get("name", ""),
            ),
            f"cancel:{call['id']}",
        )
        for call in pending
    ]


def encode_pending_input(item: PendingInput) -> dict[str, Any]:
    """Serialize one queue item without teaching the execution ledger about messages."""
    return {
        "kind": item.kind,
        "message": messages_to_dict([item.message])[0],
        "origin_id": item.origin_id,
    }


def decode_pending_input(payload: dict[str, Any]) -> PendingInput:
    """Restore the typed input stored by `encode_pending_input`."""
    messages = messages_from_dict([payload["message"]])
    return PendingInput(
        kind=str(payload["kind"]),
        message=messages[0],
        origin_id=payload.get("origin_id"),
    )


def suspend_history_snapshot(
    messages: list[BaseMessage],
    suspended_call_ids: list[str],
    completed_call_ids: list[str],
) -> list[BaseMessage]:
    """Build model history for a suspended tool round.

    Unexecuted calls are removed from the suspending assistant message. The suspended calls and
    already-completed calls remain so each can receive a matching tool result.

    Args:
        messages: Model history at suspension time.
        suspended_call_ids: Tool calls awaiting an external answer, in model order.
        completed_call_ids: Calls from the same round that already completed.

    Returns:
        A detached history snapshot suitable for resumption.
    """
    retained = {*suspended_call_ids, *completed_call_ids}
    suspending = _index_of_message_calling(messages, next(iter(suspended_call_ids), ""))
    if suspending < 0:
        return list(messages)

    message = messages[suspending]
    assert isinstance(message, AIMessage)
    snapshot: list[BaseMessage] = list(messages)
    snapshot[suspending] = AIMessage(
        content=message.content,
        tool_calls=[call for call in message.tool_calls if call["id"] in retained],
        id=message.id,
    )
    return snapshot


class Suspension(NamedTuple):
    """Persisted continuation for a run awaiting an external decision.

    Attributes:
        call: First suspended tool call, in model order.
        request: Its complete external approval request.
        messages: Model history snapshot.
        completed: Results completed before suspension.
        rules_version: Effective policy identity at suspension time.
        turn: Original planner turn number.
        subject: Audit subject associated with the run.
        parked: Every suspended ``(call, request)`` pair of the round, in model order.
            ``call``/``request`` above are its first entry.
    """

    call: ToolCall
    request: dict[str, Any]
    messages: list[BaseMessage]
    completed: list[dict[str, Any]]
    rules_version: str
    turn: int
    subject: str = ""
    parked: tuple[tuple[ToolCall, dict[str, Any]], ...] = ()


class TranscriptCursor(NamedTuple):
    """Coordinates of a suspension's canonical history in the transcript."""

    conversation_id: str
    leaf_uuid: str | None


def encode_continuation(
    call: ToolCall,
    request: dict[str, Any],
    messages: list[BaseMessage],
    completed: list[dict[str, Any]],
    rules_version: str = "",
    *,
    turn: int = 0,
    subject: str = "",
    parked: list[tuple[ToolCall, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Encode a suspension as an opaque ledger payload.

    This legacy/self-contained form remains available to orchestrator-only callers. AgentRuntime
    replaces the reconstructible fields with a transcript cursor after the transcript is durable.
    ``parked`` carries every suspended call of the round; ``call``/``request`` must be its first
    entry, which is also what a single-call round encodes without it.
    """
    payload = {
        "origin": "pre_tool_use",
        "call": call,
        "request": request,
        "messages": messages_to_dict(messages),
        "completed": completed,
        "rules_version": rules_version,
        "turn": turn,
        "subject": subject,
    }
    if parked is not None and len(parked) > 1:
        payload["calls"] = [{"call": c, "request": r} for c, r in parked]
    return payload


def decode_continuation(payload: dict[str, Any] | None) -> Suspension | None:
    """The other half. None passes through, so a caller can hand `suspension()` straight in."""
    if payload is None:
        return None
    origin = payload.get("origin")
    legacy_kind = payload.get("kind")
    if origin != "pre_tool_use" and legacy_kind != "effect_approval":
        raise ValueError(
            "cannot resume an ambiguous or tool-originated legacy suspension; "
            "only pre_tool_use permission continuations are executable"
        )
    if "messages" not in payload:
        raise ValueError("cursor continuation requires transcript messages")
    return _decode_continuation(payload, messages_from_dict(payload["messages"]))


def decode_cursor_continuation(
    payload: dict[str, Any], messages: list[BaseMessage]
) -> Suspension:
    """Decode a compact continuation after its transcript cursor has been resolved."""
    if "messages" in payload:
        decoded = decode_continuation(payload)
        assert decoded is not None
        return decoded
    if continuation_cursor(payload) is None:
        raise ValueError("continuation has neither messages nor a transcript cursor")
    return _decode_continuation(payload, messages)


def continuation_cursor(payload: dict[str, Any]) -> TranscriptCursor | None:
    """Return the transcript coordinates carried by a compact continuation."""
    raw = payload.get("transcript")
    if not isinstance(raw, dict) or not isinstance(raw.get("conversation_id"), str):
        return None
    leaf = raw.get("leaf_uuid")
    if leaf is not None and not isinstance(leaf, str):
        raise ValueError("continuation transcript leaf_uuid must be a string or null")
    return TranscriptCursor(raw["conversation_id"], leaf)


def compact_continuation(
    payload: dict[str, Any], *, conversation_id: str, leaf_uuid: str | None
) -> dict[str, Any]:
    """Replace reconstructible history with the cursor of its durable transcript branch."""
    compact = {
        name: value
        for name, value in payload.items()
        if name not in {"messages", "completed", "transcript"}
    }
    compact["transcript"] = {
        "conversation_id": conversation_id,
        "leaf_uuid": leaf_uuid,
    }
    return compact


def _decode_continuation(
    payload: dict[str, Any], messages: list[BaseMessage]
) -> Suspension:
    origin = payload.get("origin")
    legacy_kind = payload.get("kind")
    if origin != "pre_tool_use" and legacy_kind != "effect_approval":
        raise ValueError(
            "cannot resume an ambiguous or tool-originated legacy suspension; "
            "only pre_tool_use permission continuations are executable"
        )
    first = cast(ToolCall, payload["call"])
    parked = tuple(
        (cast(ToolCall, entry["call"]), entry["request"])
        for entry in payload.get("calls", [])
    ) or ((first, payload["request"]),)
    return Suspension(
        call=first,
        request=payload["request"],
        messages=messages,
        completed=payload.get("completed", []),
        rules_version=payload.get("rules_version", ""),
        turn=int(payload.get("turn", 0)),
        subject=str(payload.get("subject", "")),
        parked=parked,
    )


def _index_of_message_calling(messages: list[BaseMessage], call_id: str) -> int:
    """Index of the last assistant message that issued `call_id`, or -1."""
    for index in reversed(range(len(messages))):
        message = messages[index]
        if isinstance(message, AIMessage) and any(
            call["id"] == call_id for call in message.tool_calls
        ):
            return index
    return -1
