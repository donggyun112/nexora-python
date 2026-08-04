"""Messages across a suspension: pruning a stopped turn, reattaching the answer, parking the record.

**Where the line is.** The ledger (`nexora.orchestrator`) owns the continuation *state* — resuming a
run is its job — and stores it as an opaque payload. It does not know what a message is, and it must
not: a ledger that knows about messages grows a second opinion about what a run is.

This module is the codec on the other side of that line. It owns message shapes, the provider's rule
that every tool call needs an answer, and what a suspension has to carry — and it turns those into
a payload the ledger can hold. It does not persist anything itself; `Orchestrator.suspend` does.

Ported from `loop-helpers.ts:96-125`.
"""

from typing import Any, Literal, NamedTuple, cast

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
    """The history a run continues from, once the suspended call has an answer.

    The other half of `suspend_history_snapshot`, and the reason that function keeps the
    suspending call in place: the answer arrives as that call's `ToolMessage`, so the assistant
    turn asking for it must still be there to answer. Pass the result to `react_loop` as
    `history` with an empty `prompt` — the supervisor is not instructing the agent again, it is
    handing over what it was waiting for.

    Calls the model issued but that never ran are already gone from the snapshot; the model
    re-issues them. Only the suspended call is answered here, which is what makes a resumed run
    a continuation rather than a replay.
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
    """History as it should look when the turn resumes.

    The batch that suspended may have dispatched calls that never ran. Left in place, those
    tool calls are unanswerable — a provider rejects an assistant turn whose tool calls have
    no matching results — so they are dropped from the suspending message. Only the suspended
    call (whose result arrives on resume) and the ones that already completed survive; the
    model re-issues the rest.

    Why snapshot at all, when a transcript can be replayed? Because replay is only faithful
    for what the pause cannot change. The tools that already ran are external facts — they
    happened, and nothing in the history says so until their results are written down. The
    same test decides what else belongs in a suspension record: reconstructible from the
    conversation, leave it out; changed by the outside world while stopped, put it in.

    The history itself is the documented exception — reproducible in principle, snapshotted
    anyway so a resumed turn sees exactly the messages the suspended one did.
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


SuspensionKind = Literal["effect_approval", "elicitation"]
"""Why the run parked: an unexecuted effect gate, or an already-executed question tool."""


class Suspension(NamedTuple):
    """A run stopped waiting for a person, written down so another process can continue it.

    What is stored is deliberately **not the decision**. A suspension's window is days, not
    milliseconds, and rules change inside it — so an effect approval resume passes the answer and
    `rules_version` window through the current `on_resume` control. Storing "approved" as authority
    would let a rule added on Tuesday be ignored by an approval granted on Monday.
    """

    call: ToolCall
    request: dict[str, Any]
    """The suspend result the gate or the tool returned, whole — it carries the external handle."""
    messages: list[BaseMessage]
    completed: list[dict[str, Any]]
    rules_version: str
    """What the rules looked like when the call was made. Compare, do not trust."""
    kind: SuspensionKind
    """Whether resume must execute the call or use the human answer as the call result."""
    turn: int
    """The original model turn, retained so resumed control and observation events stay accurate."""


def encode_continuation(
    call: ToolCall,
    request: dict[str, Any],
    messages: list[BaseMessage],
    completed: list[dict[str, Any]],
    rules_version: str = "",
    *,
    kind: SuspensionKind = "elicitation",
    turn: int = 0,
) -> dict[str, Any]:
    """A suspension as something the ledger can hold without knowing what a message is.

    Half of the codec. `Orchestrator.suspend` stores what this returns; it never looks inside.

    ponytail: the whole conversation goes in, every time. It is reconstructible in principle, and
    the ceiling is that there is nowhere to reconstruct it from — no transcript exists yet
    (ADR-003 §5). When one does, this shrinks to a position in it plus the one pruned message.
    """
    return {
        "call": call,
        "request": request,
        "messages": messages_to_dict(messages),
        "completed": completed,
        "rules_version": rules_version,
        "kind": kind,
        "turn": turn,
    }


def decode_continuation(payload: dict[str, Any] | None) -> Suspension | None:
    """The other half. None passes through, so a caller can hand `suspension()` straight in."""
    if payload is None:
        return None
    return Suspension(
        call=cast(ToolCall, payload["call"]),
        request=payload["request"],
        messages=messages_from_dict(payload["messages"]),
        completed=payload["completed"],
        rules_version=payload.get("rules_version", ""),
        # Old records did not preserve the origin. Treating them as elicitation is the safe
        # migration: it cannot unexpectedly execute an external effect after an upgrade.
        kind=cast(SuspensionKind, payload.get("kind", "elicitation")),
        turn=int(payload.get("turn", 0)),
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
