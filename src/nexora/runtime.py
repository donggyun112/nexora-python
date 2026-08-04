"""The public Agent Runtime facade.

Nexora is not a general-purpose workflow engine. This facade binds the plain agent planner to its
durable execution boundary: `StepLog`, permission suspension, recovery, and events.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, ToolMessage

from .contracts import BaseMessage, EventType, PendingInput, RuntimeEvents, Tools
from .contracts.types import Aborted, Emit, OnSuspend
from .controls import Ctx
from .driver import drive
from .engines.plain import react_loop
from .history import cancelled_tool_inputs, decode_continuation, suspension_result_message
from .orchestrator import AgentSuspended, MemorySteps, Orchestrator, StepLog

__all__ = ["AgentRuntime", "run"]

OnAgentEvent = Callable[[dict[str, Any]], Awaitable[None]]
InputMode = Literal["interactive", "headless"]

_CANCELLED = {
    "type": "error",
    "message": "cancelled by a newer user request",
    "code": "cancelled",
}


class AgentRuntime:
    """Run Nexora agents without exposing their orchestration wiring.

    `StepLog` persists effect intent/results and the input queue. Agent transcript persistence is a
    separate responsibility; until a transcript store lands, crash recovery accepts durable
    history explicitly. Queue admission therefore never pretends to be a transcript checkpoint.
    """

    def __init__(
        self,
        *,
        store: StepLog | None = None,
        emit: Emit | None = None,
        owner: str = "local",
        lease_ttl: float = 60.0,
    ) -> None:
        self._store = store if store is not None else MemorySteps()
        self._emit = emit
        self._owner = owner
        self._lease_ttl = lease_ttl
        self.events = RuntimeEvents(emit)
        """Lifecycle events emitted by the host at the boundary where they actually occur."""

    async def run(
        self,
        run_id: str,
        model: Any,
        tools: Tools,
        prompt: str = "",
        *,
        controls: Any = None,
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
            active = await orchestrator.active_continuation()
            completing: str | None = None
            if prompt:
                incoming = PendingInput("user_prompt", HumanMessage(prompt), prompt_id)
                if active is not None and active.get("state") == "waiting":
                    waiting = decode_continuation(active.get("continuation"))
                    if waiting is None:
                        raise RuntimeError("active suspension has no continuation")
                    if input_mode == "headless":
                        await orchestrator.enqueue_input(incoming)
                        raise AgentSuspended(
                            str(waiting.request["pending_id"]), waiting.call["id"] or ""
                        )
                    await self._cancel_and_switch(orchestrator, active, waiting, incoming)
                    history = list(waiting.messages)
                    completing = waiting.call["id"] or ""
                elif active is not None and active.get("state") in {"switching", "resuming"}:
                    waiting = decode_continuation(active.get("continuation"))
                    if waiting is None:
                        raise RuntimeError("active continuation is corrupt")
                    await orchestrator.enqueue_input(incoming)
                    history = list(waiting.messages) if history is None else history
                    completing = waiting.call["id"] or ""
                else:
                    await orchestrator.enqueue_input(incoming)
            elif active is not None:
                waiting = decode_continuation(active.get("continuation"))
                if waiting is None:
                    raise RuntimeError("active continuation is corrupt")
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
        controls: Any = None,
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
                ctx=Ctx(turn=waiting.turn, messages=list(waiting.messages)),
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
        controls: Any = None,
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
        controls: Any,
        on_event: OnAgentEvent | None,
        **engine_options: Any,
    ) -> dict[str, Any]:
        if "drain_inputs" in engine_options or "admit_inputs" in engine_options:
            raise TypeError("AgentRuntime owns drain_inputs and admit_inputs")

        async def drain_inputs() -> list[PendingInput]:
            return await orchestrator.claim_inputs(history)

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
