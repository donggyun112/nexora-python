"""Plan mode as one permission gate over a mutable mode. Fakes, not mocks."""

from typing import Any

from nexora.builtins._exec import exec_is_read_only
from nexora.controls import Continue, Ctx, Deny, Journal, Permissions
from nexora.plan_mode import PlanMode, plan_mode_exit, plan_mode_gate

from tests.test_loop import Tools, a_call

CTX = Ctx(turn=0)
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


async def test_the_composers_carry_the_gate_and_the_exit() -> None:
    """The assembled path: a stage or writer lost inside Permissions/Journal is a silent bypass."""
    mode = PlanMode(active=True)
    permissions = Permissions(plan_mode_gate(planning_tools(), mode, exit_tool=EXIT))
    journal = Journal(plan_mode_exit(mode, exit_tool=EXIT))

    denied = await permissions(CTX, a_call("c1", "write"))
    await journal(CTX, a_call("c2", EXIT), OK)

    assert isinstance(denied, Deny)
    assert await permissions(CTX, a_call("c3", "write")) == Continue()
