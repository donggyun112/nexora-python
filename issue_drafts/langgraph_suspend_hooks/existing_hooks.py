import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime

from tests.test_loop import a_call, says, scripted


class StopBeforeTools(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        del runtime
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        self.calls = list(last.tool_calls)
        if any(call["name"] == "deploy" for call in self.calls):
            return {"jump_to": "end"}
        return None


class StopAfterSuspendResult(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        del runtime
        trailing = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        if not trailing:
            return None
        self.results = [
            message.artifact for message in trailing if isinstance(message.artifact, dict)
        ]
        if any(result.get("type") == "suspend" for result in self.results):
            return {"jump_to": "end"}
        return None


def make_tool(name: str, result: dict[str, Any], ran: list[str]) -> StructuredTool:
    async def run() -> tuple[str, dict[str, Any]]:
        ran.append(name)
        return "ok", result

    return StructuredTool.from_function(
        coroutine=run,
        name=name,
        description=name,
        response_format="content_and_artifact",
    )


async def policy_suspend_with_after_model() -> None:
    ran: list[str] = []
    middleware = StopBeforeTools()
    model = scripted(says("", a_call("c1", "deploy"), a_call("c2", "read")))
    agent = create_agent(
        model,
        [
            make_tool("deploy", {"type": "text", "text": "deployed"}, ran),
            make_tool("read", {"type": "text", "text": "read"}, ran),
        ],
        middleware=[middleware],
    )

    async for _ in agent.astream({"messages": [("user", "hi")]}):
        pass
    print("after_model policy gate")
    print("tools ran:", ran)
    print("stopped before requesting a second model response: yes")
    print("calls visible to hook:", [call["name"] for call in middleware.calls])


async def tool_suspend_with_before_model() -> None:
    ran: list[str] = []
    middleware = StopAfterSuspendResult()
    model = scripted(says("", a_call("c1", "read"), a_call("c2", "ask")))
    agent = create_agent(
        model,
        [
            make_tool("read", {"type": "text", "text": "read"}, ran),
            make_tool("ask", {"type": "suspend", "pending_id": "p1"}, ran),
        ],
        middleware=[middleware],
    )

    async for _ in agent.astream({"messages": [("user", "hi")]}):
        pass
    print("before_model tool-result detector")
    print("tools ran:", ran)
    print("stopped before requesting a second model response: yes")
    print("results visible to hook:", middleware.results)


async def main() -> None:
    await policy_suspend_with_after_model()
    await tool_suspend_with_before_model()


asyncio.run(main())
