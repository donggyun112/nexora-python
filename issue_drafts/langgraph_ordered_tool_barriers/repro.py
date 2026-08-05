import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

trace: list[str] = []


def make_tool(name: str, execution_mode: str) -> StructuredTool:
    async def run() -> str:
        trace.append(f"{name}:start")
        await asyncio.sleep(0.02)
        trace.append(f"{name}:end")
        return "ok"

    return StructuredTool.from_function(
        coroutine=run,
        name=name,
        description=f"{execution_mode} tool",
        metadata={"execution_mode": execution_mode},
    )


tools = [
    make_tool("read_a", "shared"),
    make_tool("read_b", "shared"),
    make_tool("write_a", "exclusive"),
    make_tool("write_b", "exclusive"),
    make_tool("read_c", "shared"),
]

builder = StateGraph(MessagesState)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
graph = builder.compile()

message = AIMessage(
    content="",
    tool_calls=[
        {"id": f"call-{index}", "name": tool.name, "args": {}}
        for index, tool in enumerate(tools)
    ],
)

asyncio.run(graph.ainvoke({"messages": [message]}))

first_end = next(index for index, event in enumerate(trace) if event.endswith(":end"))
started_before_first_end = [event for event in trace[:first_end] if event.endswith(":start")]

print("current ToolNode trace:", trace)
print("calls started before the first completion:", started_before_first_end)
print(
    "desired phases:",
    [["read_a", "read_b"], ["write_a"], ["write_b"], ["read_c"]],
)

assert len(started_before_first_end) == 5
