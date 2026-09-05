"""The composition rules, one per control point. Pure: no model, no store."""

from collections.abc import Sequence

import pytest
from pydantic_ai.messages import ModelRequestPart, ToolCallPart, UserPromptPart
from semora.contracts import PendingInput
from semora.controls import (
    Continue,
    ControlPlane,
    Ctx,
    Deny,
    FinishPolicy,
    Halt,
    Ingress,
    Journal,
    Permissions,
    Proceed,
    ResumeInput,
    Steering,
    Suspend,
)

CTX = Ctx(turn=1)
CALL = ToolCallPart("write", {"path": "a"}, tool_call_id="c1")


def text(item: PendingInput) -> str:
    assert isinstance(item.part, UserPromptPart)
    return str(item.part.content)


def texts(parts: Sequence[ModelRequestPart]) -> list[str]:
    return [str(p.content) for p in parts if isinstance(p, UserPromptPart)]


async def test_screens_chain_each_seeing_the_previous_rewrite() -> None:
    async def upper(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput]:
        return [PendingInput(i.kind, UserPromptPart(text(i).upper())) for i in inputs]

    async def exclaim(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput]:
        return [PendingInput(i.kind, UserPromptPart(text(i) + "!")) for i in inputs]

    out = await Ingress(upper, exclaim)(CTX, [PendingInput("user", UserPromptPart("hi"))])
    assert not isinstance(out, Halt)
    assert text(out[0]) == "HI!"


async def test_a_halting_screen_is_the_last_one_asked() -> None:
    asked: list[str] = []

    async def halt(ctx: Ctx, inputs: list[PendingInput]) -> Halt:
        asked.append("halt")
        return Halt("policy")

    async def later(ctx: Ctx, inputs: list[PendingInput]) -> list[PendingInput]:
        asked.append("later")
        return inputs

    assert await Ingress(halt, later)(CTX, []) == Halt("policy")
    assert asked == ["halt"]


async def test_a_deny_beats_a_suspend_whatever_the_order() -> None:
    async def deny(ctx: Ctx, call: ToolCallPart) -> Deny:
        return Deny({"type": "error", "message": "no"})

    async def ask(ctx: Ctx, call: ToolCallPart) -> Suspend:
        return Suspend({"pending_id": "p1"})

    assert isinstance(await Permissions(ask, deny)(CTX, CALL), Deny)
    assert isinstance(await Permissions(deny, ask)(CTX, CALL), Deny)


async def test_an_allow_does_not_short_circuit() -> None:
    asked: list[str] = []

    async def allow(ctx: Ctx, call: ToolCallPart) -> Continue:
        asked.append("allow")
        return Continue()

    async def deny(ctx: Ctx, call: ToolCallPart) -> Deny:
        asked.append("deny")
        return Deny("no")

    assert isinstance(await Permissions(allow, deny)(CTX, CALL), Deny)
    assert asked == ["allow", "deny"]


async def test_a_remembered_suspend_is_the_answer_when_nothing_denies() -> None:
    async def ask(ctx: Ctx, call: ToolCallPart) -> Suspend:
        return Suspend({"pending_id": "p1"})

    async def allow(ctx: Ctx, call: ToolCallPart) -> Continue:
        return Continue()

    assert await Permissions(ask, allow)(CTX, CALL) == Suspend({"pending_id": "p1"})


async def test_every_writer_runs_and_a_raise_is_fail_closed() -> None:
    seen: list[str] = []

    async def first(ctx: Ctx, call: ToolCallPart, result: object) -> None:
        seen.append("first")

    async def broken(ctx: Ctx, call: ToolCallPart, result: object) -> None:
        raise OSError("journal is full")

    async def third(ctx: Ctx, call: ToolCallPart, result: object) -> None:
        seen.append("third")

    with pytest.raises(OSError):
        await Journal(first, broken, third)(CTX, CALL, "ok")
    assert seen == ["first"]


async def test_steers_accumulate_in_arrival_order() -> None:
    async def one(ctx: Ctx) -> Proceed:
        return Proceed([UserPromptPart("one")])

    async def two(ctx: Ctx) -> Proceed:
        return Proceed([UserPromptPart("two")])

    decision = await Steering(one, two)(CTX)
    assert isinstance(decision, Proceed)
    assert texts(decision.steers) == ["one", "two"]


async def test_a_halt_stops_before_the_later_sources() -> None:
    asked: list[str] = []

    async def stop(ctx: Ctx) -> Halt:
        asked.append("stop")
        return Halt("policy")

    async def later(ctx: Ctx) -> Proceed:
        asked.append("later")
        return Proceed()

    assert await Steering(stop, later)(CTX) == Halt("policy")
    assert asked == ["stop"]


async def test_a_gate_cannot_relabel_an_ending_it_did_not_object_to() -> None:
    async def fine(ctx: Ctx, reason: str) -> Halt:
        return Halt("policy")

    assert await FinishPolicy(fine)(CTX, "completed") == Halt("completed")


async def test_vetoes_accumulate_their_steering() -> None:
    async def cite(ctx: Ctx, reason: str) -> Proceed:
        return Proceed([UserPromptPart("cite")])

    async def test(ctx: Ctx, reason: str) -> Proceed:
        return Proceed([UserPromptPart("test")])

    decision = await FinishPolicy(cite, test)(CTX, "completed")
    assert isinstance(decision, Proceed)
    assert texts(decision.steers) == ["cite", "test"]


async def test_an_empty_plane_lets_everything_through() -> None:
    plane = ControlPlane()
    inputs = [PendingInput("user", UserPromptPart("hi"))]
    assert await plane.on_inputs(CTX, inputs) == inputs
    assert await plane.before_model(CTX) == Proceed()
    assert await plane.pre_tool_use(CTX, CALL) == Continue()
    assert await plane.before_finish(CTX, "completed") == Halt("completed")


async def test_a_denied_human_answer_cannot_be_lifted_by_the_resume_handler() -> None:
    async def lift(ctx: Ctx, call: ToolCallPart, resume: ResumeInput) -> Continue:
        return Continue()

    plane = ControlPlane(on_resume=lift)
    resume = ResumeInput({"type": "error", "message": "no"}, {"pending_id": "p1"}, "v1", "v1")
    assert await plane.on_resume(CTX, CALL, resume) == Deny({"type": "error", "message": "no"})
