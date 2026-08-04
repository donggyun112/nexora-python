### Checked other resources

- [x] This is a bug, not a usage question.
- [x] I added a clear and descriptive title that summarizes this issue.
- [x] I used the GitHub search to find a similar question and didn't find it.
- [x] I am sure that this is a bug in LangGraph rather than my code.
- [x] The bug is not resolved by updating to the latest stable version of LangGraph.
- [x] This is not related to the langchain-community package.
- [x] I posted a self-contained, minimal, reproducible example. A maintainer can copy it and run it AS IS.

### Related Issues / PRs

I did not find an existing Python LangGraph issue or pull request covering `ToolNode` ignoring `RunnableConfig.max_concurrency` in its async execution path.

This is distinct from requests for a new deterministic tool-ordering option. `max_concurrency` is already a public `RunnableConfig` field. The issue is that the async `ToolNode` path does not enforce that existing concurrency limit.

### Reproduction Steps / Example Code (Python)

```python
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
```

### Error Message and Stack Trace

```text
sync max_active: 1
sync trace: ['slow:start', 'slow:end', 'fast:start', 'fast:end']
async max_active: 2
async trace: ['slow:start', 'fast:start', 'fast:end', 'slow:end']
Traceback (most recent call last):
  File "repro.py", line 123, in <module>
    assert async_probe.max_active == 1, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: RunnableConfig.max_concurrency=1 was ignored by ToolNode.ainvoke
```

### Description

I am using a `ToolNode` in a custom `StateGraph`. The input contains two tool calls, and I invoke the graph with `config={"max_concurrency": 1}`.

[`RunnableConfig.max_concurrency`](https://reference.langchain.com/python/langchain-core/runnables/config/RunnableConfig/max_concurrency) is documented as the maximum number of parallel calls to make. I therefore expect both `graph.invoke(...)` and `graph.ainvoke(...)` to limit the number of concurrently executing tools to one.

The synchronous path behaves as expected: the measured maximum is one. The asynchronous path starts both tools concurrently and the measured maximum is two. No exception or warning indicates that the config was ignored.

This matters for async applications that use the public `ToolNode` API to protect external rate-limited services, database or connection pools, non-concurrency-safe tools, or other bounded resources. Setting the documented concurrency limit currently provides no protection on this path.

The behavior appears to originate in `ToolNode._func` and `ToolNode._afunc`. The synchronous implementation uses the configured executor:

```python
with get_executor_for_config(config) as executor:
    outputs = list(
        executor.map(self._run_one, tool_calls, input_types, tool_runtimes)
    )
```

The asynchronous implementation instead gathers every coroutine without applying `config["max_concurrency"]`:

```python
coros = []
for call, tool_runtime in zip(tool_calls, tool_runtimes, strict=False):
    coros.append(self._arun_one(call, input_type, tool_runtime))
outputs = await asyncio.gather(*coros)
```

Source: https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py#L793-L860

A possible minimal fix would be to use LangChain Core's existing `gather_with_concurrency` helper with `config.get("max_concurrency")`, together with sync/async parity tests. I am mentioning this only as a possible direction; the main report is the observable public-API contract mismatch.

Scope note: this reproduction directly places one `ToolNode` in a custom `StateGraph` and passes multiple calls to that node. In a separate control on the tested versions, `create_agent`'s `Send`-based path honored `max_concurrency=1`. This report is limited to direct multi-call `ToolNode` async execution.

### System Info

```text
System Information
------------------
> OS:  Darwin
> OS Version:  Darwin Kernel Version 25.5.0: Tue Jun  9 22:28:34 PDT 2026; root:xnu-12377.121.10~1/RELEASE_ARM64_T6041
> Python Version:  3.12.10 (main, May 30 2025, 05:53:56) [Clang 20.1.4 ]

Package Information
-------------------
> langchain_core: 1.5.3
> langchain: 1.3.14
> langsmith: 0.10.15
> langchain_openai: 0.3.34
> langchain_protocol: 0.0.18
> langgraph_sdk: 0.4.2

Optional packages not installed
-------------------------------
> deepagents
> deepagents-cli

Other Dependencies
------------------
> anyio: 4.14.2
> distro: 1.9.0
> httpx: 0.28.1
> jsonpatch: 1.33
> langgraph: 1.2.10
> langgraph-prebuilt: 1.1.0
> openai: 2.52.0
> orjson: 3.11.9
> packaging: 26.2
> pydantic: 2.13.4
> pytest: 9.1.1
> pyyaml: 6.0.3
> requests: 2.34.2
> requests-toolbelt: 1.0.0
> sniffio: 1.3.1
> tenacity: 9.1.4
> tiktoken: 0.13.0
> typing-extensions: 4.16.0
> uuid-utils: 0.17.0
> websockets: 15.0.1
> xxhash: 3.8.1
> zstandard: 0.25.0
```
