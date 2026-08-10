import asyncio
from typing import Any

from nexora.engines.langgraph import langgraph_loop

from tests.test_loop import Tools, a_call, says, scripted


async def policy_suspend() -> None:
    llm = scripted(
        says("", a_call("c1", "deploy"), a_call("c2", "read")),
        says("model was called again"),
    )
    tools = Tools(names=["deploy", "read"])
    captured: list[Any] = []

    async def gate(call: dict[str, Any]) -> dict[str, Any] | None:
        if call["name"] == "deploy":
            return {"type": "suspend", "pending_id": "approval-1"}
        return None

    async def on_suspend(*args: Any) -> None:
        captured.append(args)

    events = [
        event
        async for event in langgraph_loop(
            llm,
            tools,
            "hi",
            before_tool_call=gate,
            on_suspend=on_suspend,
        )
    ]
    print("policy suspend")
    print("tools ran:", tools.ran)
    print("model calls:", len(llm.seen))
    print("on_suspend calls:", len(captured))
    print("event types:", [event["type"] for event in events])


async def tool_return_suspend() -> None:
    llm = scripted(
        says("", a_call("c1", "read"), a_call("c2", "ask")),
        says("model was called again"),
    )
    tools = Tools(
        results={"ask": {"type": "suspend", "pending_id": "handraise-1"}},
        defs={"ask": {"is_exclusive": True}},
        names=["read", "ask"],
    )
    captured: list[Any] = []

    async def on_suspend(*args: Any) -> None:
        captured.append(args)

    events = [
        event
        async for event in langgraph_loop(
            llm,
            tools,
            "hi",
            on_suspend=on_suspend,
        )
    ]
    print("tool-return suspend")
    print("tools ran:", tools.ran)
    print("model calls:", len(llm.seen))
    print("on_suspend calls:", len(captured))
    print("event types:", [event["type"] for event in events])


async def main() -> None:
    await policy_suspend()
    await tool_return_suspend()


asyncio.run(main())
