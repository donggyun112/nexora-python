"""Typed control points with point-specific composition rules."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from nexora.contracts import PendingInput
from nexora.controls import (
    Continue,
    ControlPlane,
    Controls,
    Ctx,
    Deny,
    Halt,
    Ingress,
    Journal,
    Permissions,
    Proceed,
    ResumeInput,
    Steering,
    Suspend,
    Suspending,
)
from tests.test_loop import a_call

CTX = Ctx(turn=0)
CALL = a_call("c1", "deploy")


def gate(answer: Any, seen: list[str], name: str) -> Any:
    async def stage(ctx: Ctx, call: Any) -> Any:
        seen.append(name)
        return answer

    return stage


# ── on_inputs: screens chain, any may halt ───────────────────────────────

INPUT = PendingInput("user_prompt", HumanMessage("ssn is 123"))


async def test_screens_chain_each_seeing_the_previous_rewrite() -> None:
    """A pipeline, not a vote: masking only works if later screens never see the original."""

    def stamp(tag: str) -> Any:
        async def screen(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput]:
            return [
                PendingInput(i.kind, HumanMessage(f"{i.message.content}|{tag}"), i.origin_id)
                for i in inputs
            ]

        return screen

    admitted = await Ingress(stamp("a"), stamp("b"))(CTX, [INPUT])

    assert isinstance(admitted, list)
    assert [str(i.message.content) for i in admitted] == ["ssn is 123|a|b"]


async def test_a_halting_screen_is_the_last_one_asked() -> None:
    seen: list[str] = []

    def screen(outcome: Any, name: str) -> Any:
        async def stage(ctx: Ctx, inputs: Any) -> Any:
            seen.append(name)
            return outcome

        return stage

    chain = Ingress(screen(Halt("policy"), "blocks"), screen([], "never"))

    assert await chain(CTX, [INPUT]) == Halt("policy")
    assert seen == ["blocks"]


async def test_nothing_registered_admits_inputs_unchanged() -> None:
    inputs = [INPUT]

    assert await Ingress()(CTX, inputs) == inputs
    assert await ControlPlane().on_inputs(CTX, inputs) == inputs


# ── pre_tool_use: deny wins, allow decides nothing ───────────────────────


async def test_a_deny_beats_a_suspend_whatever_the_order() -> None:
    seen: list[str] = []
    chain = Permissions(
        gate(Suspend({"pending_id": "c1"}), seen, "asks"),
        gate(Deny({"type": "error", "message": "no"}), seen, "denies"),
    )

    assert await chain(CTX, CALL) == Deny({"type": "error", "message": "no"})
    assert seen == ["asks", "denies"]


async def test_an_allow_does_not_short_circuit() -> None:
    """The load-bearing half: a permissive stage is an opinion, and the rules below still run."""
    seen: list[str] = []
    chain = Permissions(
        gate(Continue(), seen, "hook"),
        gate(Deny({"type": "error", "message": "rule"}), seen, "rules"),
        gate(Continue(), seen, "after"),
    )

    assert await chain(CTX, CALL) == Deny({"type": "error", "message": "rule"})
    assert seen == ["hook", "rules"]  # the deny stopped it, the allow did not


async def test_a_remembered_suspend_is_the_answer_when_nothing_denies() -> None:
    seen: list[str] = []
    chain = Permissions(
        gate(Suspend({"pending_id": "c1"}), seen, "asks"),
        gate(Continue(), seen, "allows"),
    )

    assert await chain(CTX, CALL) == Suspend({"pending_id": "c1"})
    assert seen == ["asks", "allows"]


async def test_nothing_registered_lets_the_call_through() -> None:
    assert await Permissions()(CTX, CALL) == Continue()


# ── after_tool_call: all run, a raise stops the run ──────────────────────────


async def test_every_writer_runs_and_a_raise_is_fail_closed() -> None:
    written: list[str] = []

    async def ok(ctx: Ctx, call: Any, result: Any) -> None:
        written.append("ok")

    async def broken(ctx: Ctx, call: Any, result: Any) -> None:
        raise RuntimeError("disk full")

    async def never(ctx: Ctx, call: Any, result: Any) -> None:
        written.append("never")

    with pytest.raises(RuntimeError, match="disk full"):
        await Journal(ok, broken, never)(CTX, CALL, {"type": "text", "text": "x"})

    assert written == ["ok"]


# ── before_model: steers accumulate in order ─────────────────────────────────


async def test_steers_accumulate_in_arrival_order() -> None:
    a, b = HumanMessage("first"), HumanMessage("second")

    async def one(ctx: Ctx) -> Any:
        return Proceed([a])

    async def two(ctx: Ctx) -> Any:
        return Proceed([b])

    action = await Steering(one, two)(CTX)

    assert isinstance(action, Proceed)
    assert action.steers == [a, b]


async def test_a_halt_stops_before_the_later_sources() -> None:
    seen: list[str] = []

    async def stops(ctx: Ctx) -> Any:
        seen.append("stops")
        return Halt("policy")

    async def never(ctx: Ctx) -> Any:
        seen.append("never")
        return Proceed()

    assert await Steering(stops, never)(CTX) == Halt("policy")
    assert seen == ["stops"]


# ── on_suspend: all must succeed before it is announced ──────────────────────


async def test_a_failed_persist_prevents_the_announcement() -> None:
    async def broken(*args: Any) -> None:
        raise RuntimeError("no database")

    with pytest.raises(RuntimeError, match="no database"):
        await Suspending(broken)(CTX, CALL, {"pending_id": "c1"}, [], [])


# ── the plane ────────────────────────────────────────────────────────────────


async def test_an_empty_plane_lets_everything_through() -> None:
    plane = ControlPlane()

    assert isinstance(plane, Controls)
    assert await plane.before_model(CTX) == Proceed()
    assert await plane.pre_tool_use(CTX, CALL) == Continue()
    resume = ResumeInput(
        answer={"type": "text", "text": "approved"},
        request={"type": "suspend", "pending_id": "c1"},
        suspended_rules_version="v1",
        current_rules_version="v1",
    )
    assert await plane.on_resume(CTX, CALL, resume) == Continue()
    await plane.after_tool_call(CTX, CALL, {"type": "text", "text": "x"})
    await plane.on_suspend(CTX, CALL, {"pending_id": "c1"}, [], [])


async def test_a_denied_human_answer_cannot_be_lifted_by_the_resume_handler() -> None:
    async def allow(ctx: Ctx, call: Any, resume: ResumeInput) -> Any:
        return Continue()

    answer = {"type": "error", "message": "rejected"}
    resume = ResumeInput(answer, {"pending_id": "c1"}, "v1", "v2")

    assert await ControlPlane(on_resume=allow).on_resume(CTX, CALL, resume) == Deny(answer)
