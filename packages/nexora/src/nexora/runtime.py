"""Public runtime facade for durable Nexora agent execution."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage

from .background import BackgroundResult
from .contracts import (
    BaseMessage,
    CompactContext,
    EventType,
    OnModelFailure,
    PendingInput,
    RuntimeEvents,
    Tools,
)
from .contracts.types import Aborted, Emit, OnSuspend
from .controls import Controls, Ctx
from .driver import drive
from .engines.plain import react_loop
from .history import (
    cancelled_tool_inputs,
    decode_continuation,
    suspension_result_message,
)
from .orchestrator import AgentSuspended, MemorySteps, Orchestrator, StepLog
from .subagents import Deliver

__all__ = ["AgentRuntime", "run"]

OnAgentEvent = Callable[[dict[str, Any]], Awaitable[None]]
InputMode = Literal["interactive", "headless"]

_CANCELLED = {
    "type": "error",
    "message": "cancelled by a newer user request",
    "code": "cancelled",
}


class AgentRuntime:
    """Bind the plain planner to durable execution, suspension, recovery, and events."""

    def __init__(
        self,
        *,
        store: StepLog | None = None,
        emit: Emit | None = None,
        owner: str = "local",
        lease_ttl: float = 60.0,
        model_failure_policy: OnModelFailure | None = None,
        compact_context: CompactContext | None = None,
    ) -> None:
        """Initialize persistence, events, leases, and bounded model-failure recovery."""
        self._store = store if store is not None else MemorySteps()
        self._emit = emit
        self._owner = owner
        self._lease_ttl = lease_ttl
        self._model_failure_policy = model_failure_policy
        self._compact_context = compact_context
        self.events = RuntimeEvents(emit)

    def background_sink(self, run_id: str) -> Deliver:
        """Return a callback that submits background results to a run's input queue."""

        async def deliver(result: BackgroundResult) -> None:
            arrival = PendingInput(
                "background_result", HumanMessage(result.as_message()), result.task_id
            )
            await self.submit(run_id, arrival)

        return deliver

    async def run(
        self,
        run_id: str,
        model: Any,
        tools: Tools,
        prompt: str = "",
        *,
        controls: Controls | None = None,
        on_event: OnAgentEvent | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        prompt_id: str | None = None,
        input_mode: InputMode = "interactive",
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Start or continue a turn; interactive input cancels a parked tool request."""
        async with self._orchestrator(run_id, on_suspend, rules_version, on_event) as orchestrator:
            history = engine_options.pop("history", None)
            incoming = (
                PendingInput("user_prompt", HumanMessage(prompt), prompt_id) if prompt else None
            )
            completing: str | None = None
            # Decoded once, before the branch, because every parked state needs the same answer
            # and three copies of "decode, then check for None" was three chances to disagree
            # about what a corrupt record means.
            active = await orchestrator.active_continuation()
            waiting = (
                decode_continuation(active.get("continuation")) if active is not None else None
            )

            # A run is parked in one of three states, and a new prompt means something different
            # in each. `waiting` is the only one a prompt can change: interactively it cancels the
            # request the run stopped on, headlessly it queues behind it. `switching`/`resuming`
            # are a transition already committed, so a prompt just joins the queue.
            if active is None:
                if incoming is not None:
                    await orchestrator.enqueue_input(incoming)
            elif waiting is None:
                raise RuntimeError("active continuation is corrupt")
            elif (
                active.get("state") == "waiting"
                and incoming is not None
                and input_mode == "interactive"
            ):
                await self._cancel_and_switch(orchestrator, active, waiting, incoming)
                # Not `if history is None`: the suspension's own transcript is the only one that
                # answers the call this just cancelled.
                history = list(waiting.messages)
                completing = waiting.call["id"] or ""
            else:
                if incoming is not None:
                    await orchestrator.enqueue_input(incoming)
                if active.get("state") == "waiting":
                    raise AgentSuspended(
                        str(waiting.request["pending_id"]), waiting.call["id"] or ""
                    )
                if active.get("state") in {"switching", "resuming"}:
                    history = list(waiting.messages) if history is None else history
                    completing = waiting.call["id"] or ""

            outcome = await self._drive(
                orchestrator,
                model,
                tools,
                history=history,
                controls=controls,
                on_event=on_event,
                **engine_options,
            )
            if completing is not None:
                await orchestrator.complete_continuation(completing)
            return outcome

    async def submit(
        self,
        run_id: str,
        item: PendingInput,
        *,
        input_mode: InputMode = "interactive",
    ) -> PendingInput:
        """Durably route input; user input cancels a parked request in interactive mode."""
        async with self._orchestrator(run_id) as orchestrator:
            active = await orchestrator.active_continuation()
            if (
                active is not None
                and active.get("state") == "waiting"
                and item.kind in {"user_prompt", "user_steer"}
                and input_mode == "interactive"
            ):
                waiting = decode_continuation(active.get("continuation"))
                if waiting is None:
                    raise RuntimeError("active suspension has no continuation")
                return await self._cancel_and_switch(orchestrator, active, waiting, item)
            return await orchestrator.enqueue_input(item)

    async def resume(
        self,
        run_id: str,
        pending_id: str,
        answer: dict[str, Any],
        model: Any,
        tools: Tools,
        *,
        controls: Controls | None = None,
        on_event: OnAgentEvent | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Route an answer by the suspension's external `pending_id` and resume its call."""
        async with self._orchestrator(run_id, on_suspend, rules_version, on_event) as orchestrator:
            active = await orchestrator.active_continuation()
            waiting = decode_continuation(active.get("continuation")) if active else None
            if (
                waiting is None
                or active is None
                or active.get("state") not in {"waiting", "resuming"}
                or str(waiting.request.get("pending_id")) != pending_id
            ):
                raise LookupError(f"no active suspension for pending id {pending_id!r}")
            tool_call_id = waiting.call["id"] or ""
            result = await orchestrator.resume_effect(
                tools,
                waiting.call,
                answer,
                waiting.request,
                waiting.rules_version,
                aborted=engine_options.get("aborted", lambda: False),
                turn=waiting.turn,
                controls=controls,
                # The parked subject, unless the caller names one. A different person approving
                # does not change who the run acts for, and the effect is about to run under that
                # authority — so the record's subject is the default, not the resumer's.
                ctx=Ctx(
                    turn=waiting.turn,
                    messages=list(waiting.messages),
                    subject=str(engine_options.get("subject") or waiting.subject),
                ),
            )
            answer_message = suspension_result_message(
                waiting.call["id"] or "",
                result,
                name=waiting.call.get("name", ""),
            )
            await orchestrator.continue_with_input(
                tool_call_id,
                active["continuation"],
                PendingInput("resume_result", answer_message, f"resume:{pending_id}")
            )
            outcome = await self._drive(
                orchestrator,
                model,
                tools,
                history=waiting.messages,
                controls=controls,
                on_event=on_event,
                **engine_options,
            )
            await orchestrator.complete_continuation(tool_call_id)
            return outcome

    async def recover(
        self,
        run_id: str,
        history: list[BaseMessage],
        model: Any,
        tools: Tools,
        *,
        controls: Controls | None = None,
        aborted: Aborted = lambda: False,
        retry_running: bool = True,
        on_event: OnAgentEvent | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Recover an interrupted tool round and continue without replaying its model turn."""
        async with self._orchestrator(run_id, on_suspend, rules_version, on_event) as orchestrator:
            recovered = await orchestrator.recover_pending(
                history,
                tools,
                controls=controls,
                aborted=aborted,
                retry_running=retry_running,
            )
            for message in recovered.history[len(history) :]:
                if isinstance(message, ToolMessage):
                    await orchestrator.enqueue_input(
                        PendingInput(
                            "tool_result",
                            message,
                            f"tool:{message.tool_call_id}:result",
                        )
                    )
            return await self._drive(
                orchestrator,
                model,
                tools,
                history=history,
                controls=controls,
                on_event=on_event,
                aborted=aborted,
                **engine_options,
            )

    async def _drive(
        self,
        orchestrator: Orchestrator,
        model: Any,
        tools: Tools,
        *,
        history: list[BaseMessage] | None,
        controls: Controls | None,
        on_event: OnAgentEvent | None,
        **engine_options: Any,
    ) -> dict[str, Any]:
        if "drain_inputs" in engine_options or "admit_inputs" in engine_options:
            raise TypeError("AgentRuntime owns drain_inputs and admit_inputs")
        engine_options.setdefault("on_model_failure", self._model_failure_policy)
        engine_options.setdefault("compact_context", self._compact_context)

        async def drain_inputs() -> list[PendingInput]:
            # Rebuilt per drain, not hoisted: `history` belongs to the caller, and a set computed
            # once would answer for the transcript as it looked before the run rather than now.
            return await orchestrator.claim_inputs(
                {message.id for message in history or [] if message.id is not None}
            )

        return await drive(
            react_loop,
            model,
            tools,
            history=history,
            controls=controls,
            emit=orchestrator.emit,
            drain_inputs=drain_inputs,
            admit_inputs=orchestrator.admit_inputs,
            execute_round=orchestrator.execute_round,
            on_event=on_event,
            **engine_options,
        )

    def _orchestrator(
        self,
        run_id: str,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        on_event: OnAgentEvent | None = None,
    ) -> Orchestrator:
        return Orchestrator(
            run_id,
            self._store,
            owner=self._owner,
            ttl=self._lease_ttl,
            emit=self._emit,
            on_suspend=on_suspend,
            on_agent_event=on_event,
            rules_version=rules_version,
        )

    async def _cancel_and_switch(
        self,
        orchestrator: Orchestrator,
        active: dict[str, Any],
        waiting: Any,
        incoming: PendingInput,
    ) -> PendingInput:
        cancellations = cancelled_tool_inputs(waiting.messages, reason=_CANCELLED["message"])
        normalized = await orchestrator.cancel_and_switch(
            waiting.call,
            active["continuation"],
            dict(_CANCELLED),
            cancellations,
            incoming,
        )
        await orchestrator.emit(
            EventType.TOOL_REQUEST_CANCELLED,
            {
                "turn": waiting.turn,
                "call_id": waiting.call["id"],
                "name": waiting.call["name"],
                "input": waiting.call["args"],
                "reason": dict(_CANCELLED),
                "replacement_input_id": normalized[-1].origin_id,
            },
        )
        return normalized[-1]


async def run(
    model: Any,
    tools: Tools,
    prompt: str = "",
    *,
    run_id: str = "default",
    runtime: AgentRuntime | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Convenience entry point for one Nexora agent turn."""
    active = runtime if runtime is not None else AgentRuntime()
    return await active.run(run_id, model, tools, prompt, **options)
