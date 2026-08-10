"""Keep a run's conversation, so talking to an agent twice is one conversation.

`AgentRuntime` does not persist a transcript yet — the README says so, and the console proved it:
an independent agent opened with `wait="none"` was reachable by run id and answered its second
turn with "I do not remember what you said". Reachable is not continuous.

`nexora.transcript` is the piece that closes it, and until the runtime writes one itself the
documented path is to hand the committed history back explicitly. So the console records what it
watches and hands it back on the next turn. What it can observe is what it can restore: the
prompt, the assistant turns, the tool calls and their results — the model-visible conversation and
nothing else. Streamed text is joined per turn rather than per delta, because a transcript of
tokens is not a transcript.

This is a console-level fix and stops at the console's edge. When the runtime writes its own
transcript this module is what should be deleted, not extended. Two limits it does not paper over:
the history lives as long as this process does, and a *retried* run appends its turn again rather
than recognising it — `TranscriptWriter` deduplicates against the same parent, and a retry that
reopens at a moved tip no longer has one.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from nexora.contracts import BaseMessage, ToolCall
from nexora.transcript import TranscriptWriter, active_branch, messages_of
from nexora_store import Transcript


class Recorder:
    """Assemble the messages of one run from the events the console sees, and store them."""

    def __init__(
        self,
        store: Transcript,
        run_id: str,
        prompt: str = "",
        *,
        parent_uuid: str | None = None,
    ) -> None:
        """Start a recording for one run, opening with the prompt that began it.

        Prefer `Recorder.open`, which finds `parent_uuid` for you. Constructing directly with the
        default starts a new branch, which is only right for a conversation that has none.
        """
        self._writer = TranscriptWriter(
            store, conversation_id=run_id, run_id=run_id, parent_uuid=parent_uuid
        )
        self._pending: list[BaseMessage] = [HumanMessage(prompt)] if prompt else []
        self._text: list[str] = []
        self._calls: list[ToolCall] = []

    @classmethod
    async def open(cls, store: Transcript, run_id: str, prompt: str = "") -> Recorder:
        """Continue this run's conversation rather than starting a second one beside it.

        The transcript is a tree, and a writer with no parent begins a new branch. A recorder per
        turn therefore forked the conversation once per turn: every earlier message stayed on disk
        as a sibling branch, `active_branch` returned only the newest fork, and an agent reached a
        second time answered as a stranger while its own history sat one link away. So the tip is
        read first and the chain continues from it.
        """
        branch = active_branch(await store.read(run_id))
        recorder = cls(
            store, run_id, prompt, parent_uuid=branch[-1]["uuid"] if branch else None
        )
        await recorder._writer.opened()
        await recorder._flush()
        return recorder

    async def observe(self, event: dict[str, Any]) -> None:
        """Fold one planner event into the conversation being assembled.

        A turn is closed by whatever ends it — the tool round it asked for, or the run itself.
        Until then its text is still arriving, and writing a message per delta would store a
        conversation no model could read back.
        """
        match event:
            case {"type": "text", "text": str(delta)}:
                self._text.append(delta)
            case {"type": "tool_call", "id": str(call_id), "name": str(name)}:
                self._calls.append(
                    {
                        "id": call_id,
                        "name": name,
                        "args": event.get("input") or {},
                        "type": "tool_call",
                    }
                )
            case {"type": "tool_result", "id": str(call_id)}:
                self._close_turn()
                self._pending.append(
                    ToolMessage(
                        content=_rendered(event.get("result")),
                        tool_call_id=call_id,
                        status="error" if event.get("is_error") else "success",
                    )
                )
            case {"type": "done"} | {"type": "error"}:
                self._close_turn()
        await self._flush()

    async def closed(self) -> None:
        """Commit whatever the last turn left unwritten."""
        self._close_turn()
        await self._flush()

    def _close_turn(self) -> None:
        """Seal the assistant turn being assembled, if there is one."""
        if not self._text and not self._calls:
            return
        # Assembled once and in this order because a `ToolMessage` may only answer a call the
        # assistant message before it actually made; splitting them would leave orphans.
        self._pending.insert(
            len(self._pending) - _trailing_answers(self._pending),
            AIMessage(content="".join(self._text), tool_calls=list(self._calls)),
        )
        self._text.clear()
        self._calls.clear()

    async def _flush(self) -> None:
        for message in self._pending:
            await self._writer.record(message)
        self._pending.clear()


def _trailing_answers(pending: list[BaseMessage]) -> int:
    """How many tool answers are already queued behind the turn being sealed."""
    answers = 0
    for message in reversed(pending):
        if not isinstance(message, ToolMessage):
            break
        answers += 1
    return answers


def _rendered(result: Any) -> str:
    """The text a tool result showed the model, as the loop renders it."""
    match result:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "error", "message": str(message)}:
            return f"[ERROR] {message}"
        case _:
            return str(result)


async def history_of(store: Transcript, run_id: str) -> list[BaseMessage]:
    """What this run already said, ready to hand back as `history`."""
    return messages_of(await store.read(run_id))
