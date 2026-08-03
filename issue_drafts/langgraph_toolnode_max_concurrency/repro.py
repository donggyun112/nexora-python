import asyncio
import threading
import time

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def graph_input() -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-1", "name": "slow", "args": {}},
                    {"id": "call-2", "name": "fast", "args": {}},
                ],
            )
        ]
    }


def compile_graph(tools: list[StructuredTool]):
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


class SyncProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.trace: list[str] = []
        self.lock = threading.Lock()

    def run(self, name: str, delay: float) -> str:
        with self.lock:
            self.trace.append(f"{name}:start")
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(delay)
        with self.lock:
            self.active -= 1
            self.trace.append(f"{name}:end")
        return "ok"


class AsyncProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.trace: list[str] = []

    async def run(self, name: str, delay: float) -> str:
        self.trace.append(f"{name}:start")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(delay)
        self.active -= 1
        self.trace.append(f"{name}:end")
        return "ok"


def run_sync_control() -> SyncProbe:
    probe = SyncProbe()

    def slow() -> str:
        return probe.run("slow", 0.05)

    def fast() -> str:
        return probe.run("fast", 0.01)

    graph = compile_graph(
        [
            StructuredTool.from_function(
                func=slow, name="slow", description="Slow sync tool"
            ),
            StructuredTool.from_function(
                func=fast, name="fast", description="Fast sync tool"
            ),
        ]
    )
    graph.invoke(graph_input(), config={"max_concurrency": 1})
    return probe


async def run_async_reproduction() -> AsyncProbe:
    probe = AsyncProbe()

    async def slow() -> str:
        return await probe.run("slow", 0.05)

    async def fast() -> str:
        return await probe.run("fast", 0.01)

    graph = compile_graph(
        [
            StructuredTool.from_function(
                coroutine=slow, name="slow", description="Slow async tool"
            ),
            StructuredTool.from_function(
                coroutine=fast, name="fast", description="Fast async tool"
            ),
        ]
    )
    await graph.ainvoke(graph_input(), config={"max_concurrency": 1})
    return probe


sync_probe = run_sync_control()
async_probe = asyncio.run(run_async_reproduction())

print("sync max_active:", sync_probe.max_active)
print("sync trace:", sync_probe.trace)
print("async max_active:", async_probe.max_active)
print("async trace:", async_probe.trace)

assert sync_probe.max_active == 1
assert async_probe.max_active == 1, (
    "RunnableConfig.max_concurrency=1 was ignored by ToolNode.ainvoke"
)
