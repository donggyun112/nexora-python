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
    suspended_call_id: str,
    completed_call_ids: list[str],
) -> list[BaseMessage]:
    """Build model history for a suspended tool round.

    Unexecuted calls are removed from the suspending assistant message. The suspended call and
    already-completed calls remain so each can receive a matching tool result.

    Args:
        messages: Model history at suspension time.
        suspended_call_id: Tool call awaiting an external answer.
        completed_call_ids: Calls from the same round that already completed.

    Returns:
        A detached history snapshot suitable for resumption.
    """
    retained = {suspended_call_id, *completed_call_ids}
    suspending = _index_of_message_calling(messages, suspended_call_id)
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
        call: Suspended tool call.
        request: Complete external approval request.
        messages: Model history snapshot.
        completed: Results completed before suspension.
        rules_version: Effective policy identity at suspension time.
        turn: Original planner turn number.
        subject: Audit subject associated with the run.
    """

    call: ToolCall
    request: dict[str, Any]
    messages: list[BaseMessage]
    completed: list[dict[str, Any]]
    rules_version: str
    turn: int
    subject: str = ""


def encode_continuation(
    call: ToolCall,
    request: dict[str, Any],
    messages: list[BaseMessage],
    completed: list[dict[str, Any]],
    rules_version: str = "",
    *,
    turn: int = 0,
    subject: str = "",
) -> dict[str, Any]:
    """Encode a suspension as an opaque ledger payload.

    The payload currently includes the full conversation snapshot. Once the append-only transcript
    store is wired into the runtime, this becomes a cursor into it — the snapshot is only here
    because nothing else yet holds the conversation durably.
    """
    return {
        "origin": "pre_tool_use",
        "call": call,
        "request": request,
        "messages": messages_to_dict(messages),
        "completed": completed,
        "rules_version": rules_version,
        "turn": turn,
        "subject": subject,
    }


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
    return Suspension(
        call=cast(ToolCall, payload["call"]),
        request=payload["request"],
        messages=messages_from_dict(payload["messages"]),
        completed=payload["completed"],
        rules_version=payload.get("rules_version", ""),
        turn=int(payload.get("turn", 0)),
        subject=str(payload.get("subject", "")),
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
