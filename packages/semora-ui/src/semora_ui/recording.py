"""Console-level recording and restoration of model-visible run history.

The recorder assembles streamed planner events into LangChain messages and persists them through
the transcript store. Subsequent console attempts restore that history explicitly.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from semora.contracts import BaseMessage, ToolCall
from semora.transcript import TranscriptWriter, active_branch, messages_of
from semora_store import Transcript


class Recorder:
    """Assemble and persist model-visible messages from planner events."""

    def __init__(
        self,
        store: Transcript,
        run_id: str,
        prompt: str = "",
        *,
        parent_uuid: str | None = None,
    ) -> None:
        """Initialize a recorder at a known transcript position."""
        self._writer = TranscriptWriter(
            store, conversation_id=run_id, run_id=run_id, parent_uuid=parent_uuid
        )
        self._pending: list[BaseMessage] = [HumanMessage(prompt)] if prompt else []
        self._text: list[str] = []
        self._calls: list[ToolCall] = []

    @classmethod
    async def open(cls, store: Transcript, run_id: str, prompt: str = "") -> Recorder:
        """Open a recorder at the current tip of a run's active transcript branch."""
        branch = active_branch(await store.read(run_id))
        recorder = cls(
            store, run_id, prompt, parent_uuid=branch[-1]["uuid"] if branch else None
        )
        await recorder._writer.opened()
        await recorder._flush()
        return recorder

    async def observe(self, event: dict[str, Any]) -> None:
        """Fold one planner event into the pending conversation turn."""
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
    """Count tool answers queued after the assistant turn being assembled."""
    answers = 0
    for message in reversed(pending):
        if not isinstance(message, ToolMessage):
            break
        answers += 1
    return answers


def _rendered(result: Any) -> str:
    """Render a tool result using the planner's model-visible format."""
    match result:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "error", "message": str(message)}:
            return f"[ERROR] {message}"
        case _:
            return str(result)


async def history_of(store: Transcript, run_id: str) -> list[BaseMessage]:
    """Return the active transcript branch as model history."""
    return messages_of(await store.read(run_id))
