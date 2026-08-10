"""`is_concurrency_safe` — the opt-in half of ADR-002, all-or-nothing per batch."""

import asyncio
from typing import Any

import pytest

from nexora import AgentRuntime
from nexora.contracts import BatchTools
from nexora.engines.plain import react_loop
from nexora.tools import Concurrent, InvalidToolResult
from tests.test_loop import Tools, a_call, says, scripted

SAFE = {"is_concurrency_safe": True}


class Traced(Tools):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.trace: list[str] = []

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        self.trace.append(f"start:{call_id}")
        await asyncio.sleep(0)
        self.trace.append(f"end:{call_id}")
        return {"type": "text", "text": name}


async def run(tools: Traced, *names: str) -> list[dict[str, Any]]:
    calls = [a_call(chr(ord("a") + i), n) for i, n in enumerate(names)]
    llm = scripted(says("", *calls), says("fin"))
    return [e async for e in react_loop(llm, Concurrent(tools))]


async def test_a_batch_that_all_declared_itself_safe_runs_together() -> None:
    tools = Traced(defs={"read": SAFE, "grep": SAFE}, names=["read", "grep"])

    events = await run(tools, "read", "grep", "read")

    assert tools.trace == ["start:a", "start:b", "start:c", "end:a", "end:b", "end:c"]
    assert [e["id"] for e in events if e["type"] == "tool_result"] == ["a", "b", "c"]


async def test_one_undeclared_call_makes_the_whole_round_sequential() -> None:
    """One undeclared call makes the entire batch execute sequentially."""
    tools = Traced(defs={"read": SAFE, "write": {}}, names=["read", "write"])

    await run(tools, "read", "read", "write")

    assert tools.trace == [
        "start:a", "end:a",
        "start:b", "end:b",
        "start:c", "end:c",
    ]


async def test_a_single_call_is_not_worth_a_gather() -> None:
    tools = Traced(defs={"read": SAFE}, names=["read"])

    await run(tools, "read")

    assert tools.trace == ["start:a", "end:a"]


async def test_a_predicate_flag_is_evaluated_against_the_arguments() -> None:
    """`_flag` allows a callable, so "safe unless it writes" is expressible per call."""
    reads_only = {"is_concurrency_safe": lambda args: args.get("mode") == "read"}
    tools = Traced(defs={"open": reads_only}, names=["open"])

    llm = scripted(
        says(
            "",
            a_call("a", "open", {"mode": "read"}),
            a_call("b", "open", {"mode": "write"}),
        ),
        says("fin"),
    )
    async for _ in react_loop(llm, Concurrent(tools)):
        pass

    assert tools.trace == ["start:a", "end:a", "start:b", "end:b"]  # `b` spoils the batch


async def test_wrapping_keeps_the_batch_capability_declared() -> None:
    assert isinstance(Concurrent(Traced()), BatchTools)


async def test_a_concurrent_tool_cannot_return_a_suspension() -> None:

    class Asking(Traced):
        async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
            if name == "ask":
                self.trace.append(f"ask:{call_id}")
                return {"type": "suspend", "pending_id": call_id}
            # Slow enough to still be in flight when `ask` fails, so an abandoned sibling is
            # visibly abandoned rather than a race the test happens to win.
            self.trace.append(f"start:{call_id}")
            await asyncio.sleep(0.05)
            self.trace.append(f"end:{call_id}")
            return {"type": "text", "text": name}

    tools = Asking(defs={"read": SAFE, "ask": SAFE}, names=["read", "ask"])
    calls = [a_call("a", "read"), a_call("b", "ask"), a_call("c", "read")]
    with pytest.raises(InvalidToolResult, match="pre_tool_use"):
        await AgentRuntime().run(
            "concurrent-invalid-suspend",
            scripted(says("", *calls)),
            tools,
            "hi",
        )
    # The raise must not outrun the calls beside it. `gather` leaves siblings running when it
    # propagates, so without `return_exceptions` these two finish after the round is over — an
    # effect landing with nothing awaiting it, and no step to show for it.
    assert tools.trace.count("end:a") == 1
    assert tools.trace.count("end:c") == 1


async def test_a_raising_tool_does_not_take_the_batch_down_with_it() -> None:
    """react.ts catches per call, so one thrown tool costs one result and not the round."""

    class Breaking(Traced):
        async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
            if name == "boom":
                raise RuntimeError("disk on fire")
            return await super().execute(name, call_id, arguments)

    tools = Breaking(defs={"read": SAFE, "boom": SAFE}, names=["read", "boom"])
    calls = [a_call("a", "read"), a_call("b", "boom"), a_call("c", "read")]
    llm = scripted(says("", *calls), says("fin"))

    results = {
        event["id"]: event["result"]
        async for event in react_loop(llm, Concurrent(tools))
        if event["type"] == "tool_result"
    }

    assert results["b"] == {"type": "error", "message": "RuntimeError: disk on fire"}
    assert results["a"]["type"] == "text" and results["c"]["type"] == "text"
