"""Plan mode as one permission gate over a mutable mode. Fakes, not mocks."""

from typing import Any

from semora.controls import Continue, Ctx, Deny, Journal, Permissions
from semora_coding.builtins._exec import exec_is_read_only
from semora_coding.plan_mode import (
    PlanMode,
    plan_mode_enter,
    plan_mode_exit,
    plan_mode_gate,
    plan_mode_prompt,
)
from semora_coding.prompts import SystemPrompt

from tests.test_loop import Tools, a_call

CTX = Ctx(turn=0)
ENTER = "start_planning"
EXIT = "submit_plan"
OK: dict[str, Any] = {"type": "text", "text": "plan recorded"}
FAILED: dict[str, Any] = {"type": "error", "message": "plan rejected"}


def planning_tools() -> Tools:
    """Definitions where the exit tool is not read-only, so only the exit rule can pass it."""
    return Tools(
        defs={
            "read": {"is_read_only": True},
            "write": {"is_read_only": False},
            EXIT: {"is_read_only": False},
        }
    )


async def test_a_write_is_denied_while_planning() -> None:
    """The mode is read at the gate (hasPermissionsToUseTool), so a write must not reach the tool.

    The denial result is the model's only explanation, so it has to name the way out.
    """
    gate = plan_mode_gate(planning_tools(), PlanMode(active=True), exit_tool=EXIT)

    decision = await gate(CTX, a_call("c1", "write"))

    assert isinstance(decision, Deny)
    assert decision.result["type"] == "error"
    assert EXIT in decision.result["message"]


async def test_a_read_only_call_passes_while_planning() -> None:
    """Planning is not a freeze: a call the definition marks read-only still runs."""
    gate = plan_mode_gate(planning_tools(), PlanMode(active=True), exit_tool=EXIT)

    assert await gate(CTX, a_call("c1", "read")) == Continue()


async def test_a_predicate_flag_splits_one_tool_by_its_arguments() -> None:
    """Tool.isReadOnly(input) — one exec tool, `ls` reads and `rm` writes.

    Reading `is_read_only` as a plain bool collapses both arguments to one answer.
    """
    tools = Tools(defs={"exec": {"is_read_only": exec_is_read_only}})
    gate = plan_mode_gate(tools, PlanMode(active=True), exit_tool=EXIT)

    allowed = await gate(CTX, a_call("c1", "exec", {"argv": ["ls"]}))
    denied = await gate(CTX, a_call("c2", "exec", {"argv": ["rm", "-rf", "build"]}))

    assert allowed == Continue()
    assert isinstance(denied, Deny)


async def test_the_exit_tool_passes_while_planning() -> None:
    """The exit tool is the way out; gating it traps the model in a mode it cannot leave.

    These definitions mark it not read-only, so passing can only come from the exit rule.
    """
    gate = plan_mode_gate(planning_tools(), PlanMode(active=True), exit_tool=EXIT)

    assert await gate(CTX, a_call("c1", EXIT)) == Continue()


async def test_a_successful_exit_lets_the_next_write_through() -> None:
    """The exit tool's result ends planning, so the gate has to see the mode go false."""
    mode = PlanMode(active=True)
    gate = plan_mode_gate(planning_tools(), mode, exit_tool=EXIT)

    await plan_mode_exit(mode, exit_tool=EXIT)(CTX, a_call("c1", EXIT), OK)

    assert await gate(CTX, a_call("c2", "write")) == Continue()


async def test_a_failed_exit_keeps_planning_on() -> None:
    """Only on success: an errored exit earns a recovery round, not write access."""
    mode = PlanMode(active=True)
    gate = plan_mode_gate(planning_tools(), mode, exit_tool=EXIT)

    await plan_mode_exit(mode, exit_tool=EXIT)(CTX, a_call("c1", EXIT), FAILED)

    assert isinstance(await gate(CTX, a_call("c2", "write")), Deny)


async def test_another_tool_succeeding_does_not_end_planning() -> None:
    """Only the named exit tool leaves the mode; otherwise any successful read would open it."""
    mode = PlanMode(active=True)
    gate = plan_mode_gate(planning_tools(), mode, exit_tool=EXIT)

    await plan_mode_exit(mode, exit_tool=EXIT)(CTX, a_call("c1", "read"), OK)

    assert isinstance(await gate(CTX, a_call("c2", "write")), Deny)


async def test_a_write_passes_when_the_mode_is_off() -> None:
    """An inactive mode is inert, which is what lets the gate stay installed for a whole run."""
    gate = plan_mode_gate(planning_tools(), PlanMode(), exit_tool=EXIT)

    assert await gate(CTX, a_call("c1", "write")) == Continue()


async def test_re_entering_plan_mode_denies_again() -> None:
    """Replanning needs the mode to close after it opened; a one-way phase cannot do this."""
    mode = PlanMode(active=True)
    gate = plan_mode_gate(planning_tools(), mode, exit_tool=EXIT)
    await plan_mode_exit(mode, exit_tool=EXIT)(CTX, a_call("c1", EXIT), OK)
    assert await gate(CTX, a_call("c2", "write")) == Continue()

    mode.active = True

    assert isinstance(await gate(CTX, a_call("c3", "write")), Deny)


async def test_the_enter_tool_turns_planning_on() -> None:
    """EnterPlanModeTool sets the mode by succeeding, so the writer must flip the same flag."""
    mode = PlanMode()
    gate = plan_mode_gate(planning_tools(), mode, exit_tool=EXIT)

    await plan_mode_enter(mode, enter_tool=ENTER)(CTX, a_call("c1", ENTER), OK)

    assert mode.active
    assert isinstance(await gate(CTX, a_call("c2", "write")), Deny)


async def test_a_failed_enter_does_not_start_planning() -> None:
    """Only on success, like the exit: an errored enter announced no mode change to the user."""
    mode = PlanMode()

    await plan_mode_enter(mode, enter_tool=ENTER)(CTX, a_call("c1", ENTER), FAILED)

    assert not mode.active


async def test_entering_clears_the_approvals_of_the_last_plan() -> None:
    """A new plan starts from zero: approvals submitted with the old plan must not survive it."""
    mode = PlanMode(approved=[{"tool": "exec", "prompt": "pytest"}])

    await plan_mode_enter(mode, enter_tool=ENTER)(CTX, a_call("c1", ENTER), OK)

    assert mode.approved == []


async def test_a_successful_exit_carries_the_approved_prompts() -> None:
    """Plan approval doubles as pre-approval (allowedPrompts), so the list must survive the exit.

    Carried verbatim: what an entry covers is the host gate's judgment, not this module's.
    """
    mode = PlanMode(active=True)
    approvals = [{"tool": "exec", "prompt": "pytest"}]

    await plan_mode_exit(mode, exit_tool=EXIT)(
        CTX, a_call("c1", EXIT, {"allowed_prompts": approvals}), OK
    )

    assert not mode.active
    assert mode.approved == approvals


async def test_a_failed_exit_carries_no_approvals() -> None:
    """A rejected plan approved nothing; keeping its list would pre-approve unapproved calls."""
    mode = PlanMode(active=True)

    await plan_mode_exit(mode, exit_tool=EXIT)(
        CTX, a_call("c1", EXIT, {"allowed_prompts": [{"tool": "exec", "prompt": "rm"}]}), FAILED
    )

    assert mode.approved == []


async def test_an_allowed_call_passes_while_planning() -> None:
    """The plan-file exception: the one write the reminder promises must clear the gate too."""
    is_plan_file = lambda call: call["args"].get("path") == "PLAN.md"  # noqa: E731
    gate = plan_mode_gate(
        planning_tools(), PlanMode(active=True), exit_tool=EXIT, allow=is_plan_file
    )

    allowed = await gate(CTX, a_call("c1", "write", {"path": "PLAN.md"}))
    denied = await gate(CTX, a_call("c2", "write", {"path": "src/app.py"}))

    assert allowed == Continue()
    assert isinstance(denied, Deny)


async def test_the_prompt_section_tracks_the_mode_each_round() -> None:
    """The reminder is injected while planning and gone after: a cached section would lie."""
    mode = PlanMode(active=True)
    prompt = SystemPrompt([plan_mode_prompt(mode, exit_tool=EXIT)])

    while_planning = await prompt.render()
    mode.active = False
    after_exit = await prompt.render()

    assert EXIT in while_planning
    assert after_exit == ""


async def test_the_composers_carry_the_gate_and_the_exit() -> None:
    """The assembled path: a stage or writer lost inside Permissions/Journal is a silent bypass."""
    mode = PlanMode(active=True)
    permissions = Permissions(plan_mode_gate(planning_tools(), mode, exit_tool=EXIT))
    journal = Journal(plan_mode_exit(mode, exit_tool=EXIT))

    denied = await permissions(CTX, a_call("c1", "write"))
    await journal(CTX, a_call("c2", EXIT), OK)

    assert isinstance(denied, Deny)
    assert await permissions(CTX, a_call("c3", "write")) == Continue()
