"""Permission policy used by the local UI's pre-tool suspension scenario."""

from typing import Any

from nexora.contracts.types import ToolCall
from nexora.controls import ControlPlane, Permissions, gate


async def require_note_write_approval(call: ToolCall) -> dict[str, Any] | None:
    """Park `remember_note` before its effect crosses the execution boundary."""
    if call["name"] != "remember_note":
        return None
    return {
        "type": "suspend",
        "pending_id": call["id"],
        "reason": "remember_note requires operator permission before execution",
        "source": "pre_tool_use",
    }


def permission_controls() -> ControlPlane:
    return ControlPlane(pre_tool_use=Permissions(gate(require_note_write_approval)))
