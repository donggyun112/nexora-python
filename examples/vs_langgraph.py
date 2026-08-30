"""The same crash, in a graph framework and in this runtime.

    uv pip install langgraph          # the only example with a dependency of its own
    uv run python examples/vs_langgraph.py

A model asks for three charges in one turn. The worker dies after the first one has
gone through. Both sides then resume and finish the round.

LangGraph writes a checkpoint when a node finishes, and a ReAct agent runs the whole
tool batch inside one node. The node never finished, so nothing about it was recorded,
and resuming runs it again from the top. The first customer is charged twice. The
message history is intact the whole time; what is missing is any record that money
moved.

Nexora keys the ledger by tool call id and writes the intent before the call, so a
resumed round can tell `done` from `absent` from a call that started and never
reported. The committed charge is restored, the two missing ones run, and the model is
never asked again.

Nothing here is a LangGraph bug, and nothing stops you from writing the same
bookkeeping by hand on top of it. The point is that you have to, and that the framework
does not tell you so — the graph looks correct, the tests pass, and the second charge
only shows up in production.
"""

import asyncio
import operator
from contextlib import suppress
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from nexora import AgentRuntime, MemorySteps
from nexora.orchestrator import Orchestrator

CALLS = [
    {"id": "c1", "name": "charge_card", "args": {"customer": "c-001"}, "type": "tool_call"},
    {"id": "c2", "name": "charge_card", "args": {"customer": "c-002"}, "type": "tool_call"},
    {"id": "c3", "name": "charge_card", "args": {"customer": "c-003"}, "type": "tool_call"},
]


def langgraph_run() -> list[str]:
    """The standard agent shape: one node decides, one node runs the whole batch."""
    charged: list[str] = []
    crash_armed = [True]

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    def agent(state: State) -> dict[str, Any]:
        answered = any(isinstance(m, ToolMessage) for m in state["messages"])
        return {"messages": [AIMessage(content="done") if answered
                             else AIMessage(content="", tool_calls=CALLS)]}

    def tools(state: State) -> dict[str, Any]:
        done = []
        for call in state["messages"][-1].tool_calls:
            if call["id"] == "c2" and crash_armed[0]:
                crash_armed[0] = False
                raise RuntimeError("the worker died")
            charged.append(call["args"]["customer"])
            done.append(ToolMessage(content="charged", tool_call_id=call["id"]))
        return {"messages": done}

    graph = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("tools", tools)
        .add_edge(START, "agent")
        .add_conditional_edges(
            "agent",
            lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
            ["tools", END],
        )
        .add_edge("tools", "agent")
        .compile(checkpointer=InMemorySaver())
    )

    config = {"configurable": {"thread_id": "billing"}}
    with suppress(RuntimeError):
        graph.invoke({"messages": [HumanMessage("charge all three")]}, config)
    print(f"  langgraph  crashed holding {charged}")
    graph.invoke(None, config)
    return charged


async def nexora_run() -> list[str]:
    """The same batch, the same crash point, against the durable step ledger."""
    charged: list[str] = []

    class Billing:
        async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
            charged.append(args["customer"])
            return {"type": "text", "text": "charged"}

        def get(self, name: str) -> dict[str, Any]:
            # A charge is not safe to reorder, so the batch runs one at a time.
            return {"name": name, "is_concurrency_safe": False}

        def list(self) -> list[dict[str, Any]]:
            return [{"name": "charge_card", "description": "charge", "parameters": {}}]

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        def _stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=str(next(self.messages).content))
            )

    store, billing = MemorySteps(), Billing()
    transcript = [HumanMessage("charge all three"), AIMessage(content="", tool_calls=CALLS)]

    # The worker dies once the first charge is committed.
    await Orchestrator("billing", store).execute_round(
        billing, CALLS, lambda: len(charged) == 1
    )
    print(f"  nexora     crashed holding {charged}")

    # Recovery does not ask the model anything; this one would answer differently.
    never_asked = Scripted(messages=iter([AIMessage(content="no idea what happened")]))
    await AgentRuntime(store=store).recover(
        "billing", transcript, never_asked, billing, retry_running=False
    )
    return charged


def main() -> None:
    print("three charges in one model turn, worker dies after the first\n")
    graph_charges = langgraph_run()
    nexora_charges = asyncio.run(nexora_run())

    print(f"\n  langgraph  finished with {graph_charges}")
    print(f"  nexora     finished with {nexora_charges}")
    print(f"\n  c-001 was charged {graph_charges.count('c-001')} times by the graph, "
          f"{nexora_charges.count('c-001')} time by the runtime")

    assert graph_charges.count("c-001") == 2, graph_charges
    assert nexora_charges == ["c-001", "c-002", "c-003"], nexora_charges


if __name__ == "__main__":
    main()
