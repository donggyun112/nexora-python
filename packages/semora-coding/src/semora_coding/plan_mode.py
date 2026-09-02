"""Plan mode, as a permission gate rather than an architecture.

Claude Code holds plan mode in the permission context: `hasPermissionsToUseTool` reads the mode and
refuses every call that the tool's own `Tool.isReadOnly(input)` does not vouch for. Ported that way
here, it needs no second planner and no phase — one `Permissions` stage, one `Journal` writer, and a
flag the caller can turn back on to replan.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from semora.contracts.types import ToolCall, Tools
from semora.controls import Continue, Ctx, Deny, ToolDecision
from semora.tools import is_read_only

from .prompts import PromptSection, volatile_prompt_section

__all__ = ["PlanMode", "plan_mode_enter", "plan_mode_exit", "plan_mode_gate", "plan_mode_prompt"]


@dataclass(slots=True)
class PlanMode:
    """Whether the session is planning.

    Claude Code's `toolPermissionContext.mode` is the reference: a value that flips both ways, so a
    later turn can re-enter planning.
    """

    active: bool = False
    approved: list[Any] = field(default_factory=list)
    """Pre-approved calls the model submitted with the last accepted plan (`allowedPrompts`).

    Carried verbatim from the exit tool's `allowed_prompts` argument. What one entry covers —
    exact match, prefix, shell parse — widens permissions, so that judgment stays with the host
    gate that reads this list.
    """


def plan_mode_gate(
    tools: Tools,
    mode: PlanMode,
    *,
    exit_tool: str,
    allow: Callable[[ToolCall], bool] | None = None,
) -> Callable[[Ctx, ToolCall], Awaitable[ToolDecision]]:
    """Build a `Permissions` stage denying every call that is not read-only while planning.

    `allow` names the host's exceptions — Claude Code's "the only file you are allowed to edit"
    is a plan-file write clearing the gate this way.
    """

    async def stage(ctx: Ctx, call: ToolCall) -> ToolDecision:
        # Read `mode.active` per call and cache nothing: the caller turns it back on to replan.
        if not mode.active:
            return Continue()
        # `exit_tool` passes even while planning, for the reason `ExitPlanModeV2Tool.isReadOnly()`
        # returns true and `EnterPlanModeTool.isEnabled()` guards the same way: without this, plan
        # mode is a trap the model can enter but never leave.
        if call["name"] == exit_tool or is_read_only(tools, call):
            return Continue()
        if allow is not None and allow(call):
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


def plan_mode_enter(
    mode: PlanMode, *, enter_tool: str
) -> Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]:
    """Build a `Journal` writer that starts plan mode when `enter_tool` succeeds.

    `EnterPlanModeTool` is the reference: entering is a tool call the host may still gate. Entering
    drops the approvals of the last plan — a new plan pre-approves nothing until it is accepted.
    """

    async def write(ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        if call["name"] == enter_tool and result.get("type") not in ("error", "suspend"):
            mode.active = True
            mode.approved = []

    return write


def plan_mode_exit(
    mode: PlanMode, *, exit_tool: str
) -> Callable[[Ctx, ToolCall, dict[str, Any]], Awaitable[None]]:
    """Build a `Journal` writer that leaves plan mode when `exit_tool` succeeds.

    Only on success, and a parked call is not one: a failed or suspended exit submitted no plan, so
    planning holds. Success is read the way `absorb_round` reads it for `terminates_loop`. Success
    also lands the call's `allowed_prompts` on `mode.approved`: the plan's acceptance is what
    turned them from a request into an approval.
    """

    async def write(ctx: Ctx, call: ToolCall, result: dict[str, Any]) -> None:
        if call["name"] == exit_tool and result.get("type") not in ("error", "suspend"):
            mode.active = False
            mode.approved = list((call["args"] or {}).get("allowed_prompts") or [])

    return write


def plan_mode_prompt(mode: PlanMode, *, exit_tool: str, name: str = "plan_mode") -> PromptSection:
    """Build a volatile system-prompt section announcing plan mode while it is on.

    The gate alone teaches by denial; the reference injects the constraint up front and the model
    plans instead of colliding with the gate. Volatile because the section must appear and vanish
    with `mode.active` inside one run.
    """

    def compute() -> str | None:
        if not mode.active:
            return None
        return (
            "Plan mode is active. The user indicated that they do not want you to execute yet -- "
            "you MUST NOT make any edits, run any non-read-only tools, or otherwise make changes "
            "to the system. This supercedes any other instructions you have received. Research "
            f"with read-only tools, then call {exit_tool} to submit the plan and leave plan mode."
        )

    return volatile_prompt_section(
        name, compute, reason="the reminder must appear and vanish with PlanMode.active"
    )
