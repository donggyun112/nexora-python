"""Message-history surgery.

Only what the loop needs today: pruning a suspended turn down to something that can be
replayed. Ported from `loop-helpers.ts:96-125`.
"""

from langchain_core.messages import AIMessage

from .contracts.types import BaseMessage


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


def _index_of_message_calling(messages: list[BaseMessage], call_id: str) -> int:
    """Index of the last assistant message that issued `call_id`, or -1."""
    for index in reversed(range(len(messages))):
        message = messages[index]
        if isinstance(message, AIMessage) and any(
            call["id"] == call_id for call in message.tool_calls
        ):
            return index
    return -1
