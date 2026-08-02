"""Message-history surgery.

Only what the loop needs today: pruning a suspended turn down to something that can be
replayed. Ported from `loop-helpers.ts:96-125`.
"""

from typing import Any

from .types import LLMMessage


def suspend_history_snapshot(
    messages: list[LLMMessage],
    suspended_call_id: str,
    completed_call_ids: list[str],
) -> list[LLMMessage]:
    """History as it should look when the turn resumes.

    The batch that suspended may have dispatched calls that never ran. Left in place, those
    `tool_call` blocks are unanswerable — a provider rejects an assistant turn whose tool
    calls have no matching results — so they are dropped from the suspending message. Only
    the suspended call (whose result arrives on resume) and the ones that already completed
    survive; the model re-issues the rest.

    Why snapshot at all, when a transcript can be replayed? Because replay is only faithful
    for what the pause cannot change. The tools that already ran are external facts — they
    happened, and nothing in the history says so until their results are written down. The
    same test decides what else belongs in a suspension record: reconstructible from the
    conversation, leave it out; changed by the outside world while stopped, put it in.

    The history itself is the documented exception — reproducible in principle, snapshotted
    anyway so a resumed turn sees exactly the messages the suspended one did.
    """
    retained = {suspended_call_id, *completed_call_ids}
    suspending_index = _index_of_message_calling(messages, suspended_call_id)

    snapshot: list[LLMMessage] = []
    for index, message in enumerate(messages):
        content = message["content"]
        if not isinstance(content, list):
            snapshot.append({**message})
            continue
        if index == suspending_index:
            content = [block for block in content if _survives(block, retained)]
        else:
            content = list(content)
        snapshot.append({**message, "content": content})
    return snapshot


def _index_of_message_calling(messages: list[LLMMessage], call_id: str) -> int:
    """Index of the last assistant message that issued `call_id`, or -1."""
    for index in reversed(range(len(messages))):
        message = messages[index]
        content = message["content"]
        if message["role"] != "assistant" or not isinstance(content, list):
            continue
        if any(b.get("type") == "tool_call" and b.get("id") == call_id for b in content):
            return index
    return -1


def _survives(block: dict[str, Any], retained: set[str]) -> bool:
    return block.get("type") != "tool_call" or block.get("id") in retained
