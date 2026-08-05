# Verification record

Date: 2026-08-03 (Asia/Seoul)

Canonical issue: https://github.com/langchain-ai/langgraph/issues/8517

Verified GitHub metadata:

- state: `open`
- issue type: `Bug`
- labels: `bug`, `external`

## Classification

- Target repository: `langchain-ai/langgraph`
- Package path: `langgraph-prebuilt` / `langgraph.prebuilt.ToolNode`
- Classification: correctness bug in an existing public configuration contract
- Not claimed: deterministic ordering of arbitrary tool calls
- Exact claim: direct multi-call `ToolNode` async execution exceeds `RunnableConfig.max_concurrency`

## Version check

- PyPI latest stable `langgraph` queried on 2026-08-03: `1.2.10`
- Reproduced with `langgraph==1.2.10`
- Installed `langgraph-prebuilt==1.1.0`
- `langchain-core==1.5.3`
- Python 3.12.10 on Darwin ARM64

## Reproduction results

The standalone reproducer was run four times. Every run produced the same measured values and traces:

```text
sync max_active: 1
sync trace: ['slow:start', 'slow:end', 'fast:start', 'fast:end']
async max_active: 2
async trace: ['slow:start', 'fast:start', 'fast:end', 'slow:end']
AssertionError: RunnableConfig.max_concurrency=1 was ignored by ToolNode.ainvoke
```

The non-zero exit is intentional: the final assertion encodes the documented expected value.

`ruff check issue_drafts/langgraph_toolnode_max_concurrency/repro.py` also passed.

## Source confirmation

The installed stable package and the downloaded upstream source both show the same asymmetry:

- sync `ToolNode._func` uses `get_executor_for_config(config)` before mapping tool calls;
- async `ToolNode._afunc` builds every coroutine and calls unrestricted `asyncio.gather(*coros)`;
- `langchain_core.runnables.utils.gather_with_concurrency` already provides a semaphore-based helper that accepts an integer limit or `None`.

Upstream source: https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py#L793-L860

Official config contract: https://reference.langchain.com/python/langchain-core/runnables/config/RunnableConfig/max_concurrency

## Duplicate search

GitHub issue searches on 2026-08-03:

- `ToolNode max_concurrency`: no exact result
- `max_concurrency`: eight results; none concern `ToolNode` async tool execution
- `asyncio.gather ToolNode`: no result
- `ToolNode concurrency`: no result
- `ToolNode sequential`: no result

GitHub pull request searches:

- `max_concurrency ToolNode`: no result
- `asyncio.gather ToolNode`: no result
- `ToolNode concurrency`: no result

An adjacent LangGraph.js request for a new sequential execution option is not a duplicate. This report concerns Python and a documented config field that is accepted but ignored on one execution path.

## Scope controls

- Sync direct `ToolNode`: honors the limit.
- Async direct `ToolNode`: ignores the limit.
- Async `create_agent` control on the tested versions: honors the limit through its `Send`-based task path.

The report is intentionally limited to direct multi-call `ToolNode.ainvoke` behavior.
