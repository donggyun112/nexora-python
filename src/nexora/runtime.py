"""The public Agent Runtime facade.

Nexora is not a general-purpose workflow engine. This facade binds the plain agent planner to its
durable execution boundary: `StepLog`, permission suspension, recovery, and events.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from .contracts import BaseMessage, PendingInput, RuntimeEvents, Tools
from .contracts.types import Aborted, Emit, OnSuspend
from .controls import Ctx
from .driver import drive
from .engines.plain import react_loop
from .history import decode_continuation, suspension_result_message
from .orchestrator import MemorySteps, Orchestrator, StepLog
from .tools import a_tool_result

__all__ = ["AgentRuntime", "run"]

OnAgentEvent = Callable[[dict[str, Any]], Awaitable[None]]


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
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Start or continue one agent turn on the configured engine."""
        async with self._orchestrator(run_id, on_suspend, rules_version, on_event) as orchestrator:
            if prompt:
                await orchestrator.enqueue_input(
                    PendingInput("user_prompt", HumanMessage(prompt), prompt_id)
                )
            history = engine_options.pop("history", None)
            return await self._drive(
                orchestrator,
                model,
                tools,
                history=history,
                controls=controls,
                on_event=on_event,
                **engine_options,
            )

    async def submit(self, run_id: str, item: PendingInput) -> PendingInput:
        """Durably enqueue an asynchronous input without starting or holding a worker."""
        return await self._orchestrator(run_id).enqueue_input(item)

    async def resume(
        self,
        run_id: str,
        tool_call_id: str,
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
        """Resume a policy/human suspension without holding the original worker."""
        async with self._orchestrator(run_id, on_suspend, rules_version, on_event) as orchestrator:
            waiting = decode_continuation(await orchestrator.suspension(tool_call_id))
            if waiting is None:
                raise LookupError(f"no suspension for tool call {tool_call_id!r}")
            result = answer
            if waiting.kind == "effect_approval":
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
            elif on_event is not None:
                event = a_tool_result(waiting.call, result)
                event["executed"] = True
                await on_event(event)
            answer_message = suspension_result_message(
                waiting.call["id"] or "",
                result,
                name=waiting.call.get("name", ""),
            )
            await orchestrator.enqueue_input(
                PendingInput("resume_result", answer_message, f"resume:{tool_call_id}")
            )
            return await self._drive(
                orchestrator,
                model,
                tools,
                history=waiting.messages,
                controls=controls,
                on_event=on_event,
                **engine_options,
            )

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
