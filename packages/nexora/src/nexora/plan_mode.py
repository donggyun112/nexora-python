"""Plan mode, as a permission gate rather than an architecture.

Claude Code holds plan mode in the permission context: `hasPermissionsToUseTool` reads the mode and
refuses every call that the tool's own `Tool.isReadOnly(input)` does not vouch for. Ported that way
here, it needs no second planner and no phase — one `Permissions` stage, one `Journal` writer, and a
flag the caller can turn back on to replan.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .contracts.types import ToolCall, Tools
from .controls import Continue, Ctx, Deny, ToolDecision
from .tools import is_read_only

__all__ = ["PlanMode", "plan_mode_exit", "plan_mode_gate"]


@dataclass(slots=True)
class PlanMode:
    """Whether the session is planning.

    Claude Code's `toolPermissionContext.mode` is the reference: a value that flips both ways, so a
    later turn can re-enter planning.
    """

    active: bool = False


def plan_mode_gate(
    tools: Tools, mode: PlanMode, *, exit_tool: str
) -> Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]:
    """Build a `Permissions` stage denying every call that is not read-only while planning."""

    async def stage(ctx: Ctx, call: ToolCall) -> ToolDecision:
        # Read `mode.active` per call and cache nothing: the caller turns it back on to replan.
        if not mode.active:
            return Continue()
        # `exit_tool` passes even while planning, for the reason `ExitPlanModeV2Tool.isReadOnly()`
        # returns true and `EnterPlanModeTool.isEnabled()` guards the same way: without this, plan
        # mode is a trap the model can enter but never leave.
        if call["name"] == exit_tool or is_read_only(tools, call):
            return Continue()
        return Deny(
            {
                "type": "error",
                "message": (
                    f"{call['name']} was not run: plan mode is active and this call is not "
                    f"read-only. Keep researching with read-only tools, then call {exit_tool} "
                    "to submit the plan and leave plan mode."
                ),
            }
        )

    return stage


def plan_mode_exit(
    mode: PlanMode, *, exit_tool: str
) -> Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]:
    """Build a `Journal` writer that leaves plan mode when `exit_tool` succeeds.

    Only on success, and a parked call is not one: a failed or suspended exit submitted no plan, so
    planning holds. Success is read the way `absorb_round` reads it for `terminates_loop`.
    """

    async def write(ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        if call["name"] == exit_tool and result.get("type") not in ("error", "suspend"):
            mode.active = False

    return write
