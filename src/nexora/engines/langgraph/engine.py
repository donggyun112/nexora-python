"""The same agent loop, driven by LangChain's `create_agent` instead of a `while`.

This exists to keep ADR-001 falsifiable. The claim there is that the reference's semantics do
not all fit LangChain's six middleware hooks; the way to find out is to write it and run both
engines against one conformance suite (`tests/test_engine_conformance.py`).

Where a semantic has no hook to live in, that is recorded here as a `ponytail:` comment rather
than worked around silently — the gaps are the point.
"""

from collections import Counter
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ...contracts.events import EventType
from ...contracts.types import (
    Aborted,
    BeforeToolCall,
    DrainSteers,
    Emit,
    LLMMessage,
    OnSuspend,
    ShouldStopAfterTurn,
    ToolCall,
    Tools,
)
from ...tools import render_for_model, select_for_execution, terminates_loop


async def langgraph_loop(
    model: Any,
    tools: Tools,
    prompt: str,
    *,
    system_prompt: str | None = None,
    history: list[LLMMessage] | None = None,
    aborted: Aborted = lambda: False,
    before_tool_call: BeforeToolCall | None = None,
    emit: Emit | None = None,
    drain_steers: DrainSteers | None = None,
    should_stop_after_turn: ShouldStopAfterTurn | None = None,
    on_suspend: OnSuspend | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Same inputs and same event stream as `nexora.loop.react_loop`.

    `model` is a LangChain chat model here rather than a `nexora.types.LLM`: `create_agent`
    owns the provider call, so the engines cannot share that one argument.

    ponytail: `on_suspend` is accepted and ignored — the gate reports a suspension as an
    event, but handing the caller a history snapshot to persist is not written. Everything
    else is implemented; `tests/test_engine_conformance.py` says which.
    """
    outcome = _Outcome()
    agent = create_agent(
        model,
        _as_langchain_tools(tools),
        system_prompt=system_prompt,
        middleware=[
            _Steering(drain_steers),
            _RoundEnd(tools, emit, should_stop_after_turn, outcome, aborted),
            _ReportFailures(outcome),
            _Gate(before_tool_call, emit, outcome),
        ],
    )

    state: dict[str, Any] = {"messages": _to_langchain(history or [], prompt)}
    seen: set[str] = set()
    calls_made: list[dict[str, Any]] = []
    spent: Counter[str] = Counter()
    text = ""

    if aborted():
        yield _done(text, calls_made, "aborted", spent)
        return

    # Both modes at once: `messages` carries the provider's deltas so text streams token by
    # token, `values` carries whole messages so tool calls and results can be read structurally.
    async for mode, payload in agent.astream(  # type: ignore[call-overload]
        state, stream_mode=["values", "messages"]
    ):
        if mode == "messages":
            delta = getattr(payload[0], "content", "")
            if isinstance(delta, str) and delta:
                yield {"type": "text", "text": delta}
            continue
        for message in payload["messages"]:
            key = str(getattr(message, "id", None) or id(message))
            if key in seen:
                continue
            seen.add(key)
            # The final answer is the last assistant turn, not every delta ever streamed —
            # accumulating the deltas would concatenate turns that a steer split apart.
            if isinstance(message, AIMessage) and isinstance(message.content, str):
                text = message.content
            for event in _translate(message, tools, calls_made, spent):
                yield event

    # A hook cannot say "the run failed" through its return value — it must return something
    # the graph accepts — so the middlewares park the reason here and the engine reads it once
    # the graph is done. The event therefore arrives after the round rather than during it.
    if outcome.error is not None:
        yield {"type": "error", "message": outcome.error}
        return
    if outcome.suspended is not None:
        call, result = outcome.suspended
        yield {"type": "suspended", "pending_id": result["pending_id"], "tool_call_id": call.id}
        return

    if emit is not None:
        await emit(EventType.STOP, {"reason": outcome.stop_reason, "content": text})
    yield _done(text, calls_made, outcome.stop_reason, spent)


@dataclass
class _Outcome:
    """Side channel out of the middleware.

    ponytail: a workaround, not a hook. `wrap_model_call` must return a `ModelResponse` and
    `wrap_tool_call` a `ToolMessage`, so neither can report "this run ended badly" in-band.
    """

    error: str | None = None
    suspended: tuple[ToolCall, dict[str, Any]] | None = None
    stop_reason: str = "completed"
    """Why the run ended. `jump_to: "end"` stops the graph but carries no reason with it."""


class _ReportFailures(AgentMiddleware):
    """Provider failure ends the run as a reported error — react.ts L131."""

    def __init__(self, outcome: _Outcome) -> None:
        super().__init__()
        self._outcome = outcome

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except Exception as failure:
            self._outcome.error = str(failure)
            # Ending the graph means handing back a turn with no tool calls; the engine
            # replaces the resulting `done` with the error it parked above.
            return AIMessage(content="")


class _Gate(AgentMiddleware):
    """The policy gate — allow / deny / ask — plus the events recording what it decided."""

    def __init__(
        self,
        before_tool_call: BeforeToolCall | None,
        emit: Emit | None,
        outcome: _Outcome,
    ) -> None:
        super().__init__()
        self._before = before_tool_call
        self._emit = emit
        self._outcome = outcome

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        raw = request.tool_call
        call = ToolCall(raw.get("id") or "", raw["name"], raw.get("args", {}))
        await self._publish(EventType.PRE_TOOL_USE, call, {})

        decision = await self._before(call) if self._before else None
        if decision is not None:
            return await self._refuse(call, decision)

        result = await handler(request)
        await self._publish(
            EventType.POST_TOOL_USE,
            call,
            {"result": {"type": "text", "text": str(getattr(result, "content", ""))}},
        )
        return result

    async def _refuse(self, call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        if decision.get("type") == "suspend":
            await self._publish(EventType.PERMISSION_REQUEST, call, {"request": decision})
            self._outcome.suspended = (call, decision)
            return ToolMessage(
                content="suspended pending approval", tool_call_id=call.id, name=call.name
            )
        await self._publish(EventType.PERMISSION_DENIED, call, {"reason": decision})
        await self._publish(EventType.POST_TOOL_USE_FAILURE, call, {"result": decision})
        return ToolMessage(
            content=render_for_model(decision),
            tool_call_id=call.id,
            name=call.name,
            status="error",
        )

    async def _publish(self, event: EventType, call: ToolCall, extra: dict[str, Any]) -> None:
        if self._emit is None:
            return
        await self._emit(
            event, {"call_id": call.id, "name": call.name, "input": call.arguments, **extra}
        )


class _RoundEnd(AgentMiddleware):
    """Everything that has to happen *after tools ran* — `post_tool_batch` and the stop policy.

    ponytail: there is no after-tools hook, so this runs in `before_model` and works out that
    tools just ran by looking for `ToolMessage`s at the tail of the message list. That is state
    re-derivation: the position the loop is at is not available, so it is reconstructed. The
    `while` engine knows because it is on that line.

    The re-derivation is also lossy. It fires on the next model call, so a round whose tools
    ran but whose run then ended never reports its batch.
    """

    def __init__(
        self,
        tools: Tools,
        emit: Emit | None,
        should_stop: ShouldStopAfterTurn | None,
        outcome: _Outcome,
        aborted: Aborted,
    ) -> None:
        super().__init__()
        self._aborted = aborted
        self._tools = tools
        self._emit = emit
        self._should_stop = should_stop
        self._outcome = outcome
        self._turn = -1
        self._reported = 0

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._turn += 1
        if self._aborted():
            self._outcome.stop_reason = "aborted"
            return {"jump_to": "end"}
        finished = _trailing_tool_messages(state["messages"])
        if not finished or len(finished) <= self._reported:
            return None
        self._reported = len(finished)

        if self._emit is not None:
            await self._emit(
                EventType.POST_TOOL_BATCH,
                {
                    "turn": self._turn - 1,
                    "calls": [{"call_id": m.tool_call_id, "name": m.name or ""} for m in finished],
                },
            )

        # A successful terminating tool ends the run; a failed one gets a recovery round.
        ended_by_tool = any(
            m.status != "error"
            and terminates_loop(self._tools, ToolCall(m.tool_call_id, m.name or "", {}))
            for m in finished
        )
        stopped_by_policy = self._should_stop is not None and await self._should_stop(
            self._turn - 1, "", []
        )
        if ended_by_tool or stopped_by_policy:
            self._outcome.stop_reason = "tool" if ended_by_tool else "policy"
            return {"jump_to": "end"}
        return None


def _trailing_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    """The `ToolMessage`s closing the most recent round, oldest first."""
    tail: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            tail.append(message)
        elif tail:
            break
    return list(reversed(tail))


class _Steering(AgentMiddleware):
    """Steers absorbed at both points react.ts uses — L116 and L152."""

    def __init__(self, drain_steers: DrainSteers | None) -> None:
        super().__init__()
        self._drain = drain_steers

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._drain is None:
            return None
        steers = self._drain()
        if not steers:
            return None
        return {"messages": _to_langchain(steers, None)}

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """A steer that landed as the turn finished cancels the stop — react.ts L152."""
        if self._drain is None:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        steers = self._drain()
        if not steers:
            return None
        return {"messages": _to_langchain(steers, None), "jump_to": "model"}



def _as_langchain_tools(tools: Tools) -> list[StructuredTool]:
    """Wrap our executor's tools so `create_agent`'s ToolNode can call them."""
    wrapped: list[StructuredTool] = []
    for summary in tools.list():
        name = summary["name"]

        async def run(_name: str = name, **kwargs: Any) -> tuple[str, dict[str, Any]]:
            # `content_and_artifact` keeps the result's real shape alongside the text the
            # model sees; without it a multimodal or suspend result is flattened to a string
            # on the way through `ToolMessage` and cannot be recovered.
            result = await tools.execute(_name, _name, kwargs)
            return render_for_model(result), result

        wrapped.append(
            StructuredTool.from_function(
                coroutine=run,
                name=name,
                description=summary["description"],
                args_schema=summary["parameters"],
                response_format="content_and_artifact",
            )
        )
    return wrapped


def _to_langchain(messages: list[LLMMessage], prompt: str | None) -> list[BaseMessage]:
    from langchain_core.messages import HumanMessage

    out: list[BaseMessage] = [
        HumanMessage(content=m["content"] if isinstance(m["content"], str) else "")
        for m in messages
    ]
    if prompt is not None:
        out.append(HumanMessage(content=prompt))
    return out


def _translate(
    message: BaseMessage,
    tools: Tools,
    calls_made: list[dict[str, Any]],
    spent: Counter[str],
) -> Iterator[dict[str, Any]]:
    """One LangChain message into zero or more Nexora events."""
    if isinstance(message, AIMessage):
        if message.usage_metadata:
            spent.update(
                {
                    "prompt_tokens": message.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": message.usage_metadata.get("output_tokens", 0),
                }
            )
        requested = [ToolCall(c["id"] or "", c["name"], c["args"]) for c in message.tool_calls]
        for call in select_for_execution(tools, requested):
            calls_made.append({"name": call.name, "input": call.arguments})
            yield {
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
    elif isinstance(message, ToolMessage):
        yield {
            "type": "tool_result",
            "id": message.tool_call_id,
            "name": message.name or "",
            "result": message.artifact
            if isinstance(message.artifact, dict)
            else {"type": "text", "text": str(message.content)},
            "is_error": message.status == "error",
        }


def _done(
    content: str,
    calls_made: list[dict[str, Any]],
    reason: str,
    spent: Counter[str],
) -> dict[str, Any]:
    done: dict[str, Any] = {
        "type": "done",
        "content": content,
        "tool_calls": calls_made,
        "stop_reason": reason,
    }
    if spent:
        done["usage"] = dict(spent)
    return done
