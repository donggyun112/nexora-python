"""Optional runtime-orchestration ports and the durable implementation."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from semora_store import ExecutionContext, ExecutionStore

from .contracts import Emit, InvokeModel, OnSuspend, PendingInput
from .orchestrator import Orchestrator
from .tools import ExecuteRound, execute_calls

__all__ = [
    "DurableRuntimeOrchestrator",
    "RuntimeInputSession",
    "RuntimeOrchestrationContext",
    "RuntimeOrchestrationSession",
    "RuntimeOrchestrator",
]

AgentEventSink = Callable[[dict[str, Any]], Awaitable[None]]
_execute_direct: ExecuteRound = execute_calls


@dataclass(frozen=True, slots=True)
class RuntimeOrchestrationContext:
    """Describe one runtime attempt to a detachable orchestrator."""

    execution: ExecutionContext
    emit: Emit | None = None
    on_suspend: OnSuspend | None = None
    on_agent_event: AgentEventSink | None = None
    rules_version: str = ""

    @property
    def run_id(self) -> str:
        """Return the framework-owned execution coordinate."""
        return self.execution.run_id


class RuntimeInputSession(Protocol):
    """Admit external inputs through an orchestrated attempt."""

    async def submit(self, item: PendingInput) -> PendingInput:
        """Queue one input and return its stable normalized envelope."""
        ...

    async def claim(self, represented: set[str] | None = None) -> list[PendingInput]:
        """Claim inputs not already represented in model-visible history."""
        ...

    async def admit(self, items: list[PendingInput]) -> None:
        """Commit inputs appended to model-visible history."""
        ...

    async def discard(self, items: list[PendingInput]) -> None:
        """Commit inputs permanently removed by ingress controls."""
        ...


class RuntimeOrchestrationSession(Protocol):
    """Wrap execution boundaries for the lifetime of one runtime attempt."""

    @property
    def inputs(self) -> RuntimeInputSession | None:
        """Return input admission when the orchestrator owns it."""
        ...

    def wrap_model(self, inner: InvokeModel) -> InvokeModel:
        """Wrap the model invocation boundary."""
        ...

    def wrap_tools(self, inner: ExecuteRound) -> ExecuteRound:
        """Wrap the tool-round execution boundary."""
        ...


class RuntimeOrchestrator(Protocol):
    """Open an optional orchestration session around the plain planner."""

    def open(
        self, context: RuntimeOrchestrationContext
    ) -> AbstractAsyncContextManager[RuntimeOrchestrationSession]:
        """Open one attempt and close all attached orchestration resources afterward."""
        ...


@dataclass(slots=True)
class _DurableRuntimeSession:
    """Adapt the durable orchestrator to the small runtime-orchestration port."""

    orchestrator: Orchestrator

    @property
    def inputs(self) -> RuntimeInputSession:
        """Expose this session's durable input queue."""
        return self

    async def submit(self, item: PendingInput) -> PendingInput:
        """Queue one durable input."""
        return await self.orchestrator.enqueue_input(item)

    async def claim(self, represented: set[str] | None = None) -> list[PendingInput]:
        """Claim durable inputs for this attempt."""
        return await self.orchestrator.claim_inputs(represented)

    async def admit(self, items: list[PendingInput]) -> None:
        """Commit admitted durable inputs."""
        await self.orchestrator.admit_inputs(items)

    async def discard(self, items: list[PendingInput]) -> None:
        """Commit discarded durable inputs."""
        await self.orchestrator.discard_inputs(items)

    def wrap_model(self, inner: InvokeModel) -> InvokeModel:
        """Replace direct model invocation with durable execute-or-replay."""
        del inner
        return self.orchestrator.invoke_model

    def wrap_tools(self, inner: ExecuteRound) -> ExecuteRound:
        """Replace direct tool execution with the durable effect boundary."""
        del inner
        return self.orchestrator.execute_round


class DurableRuntimeOrchestrator:
    """Attach the existing ledger, suspension, and recovery policy to ``AgentRuntime``."""

    def __init__(
        self,
        execution_store: ExecutionStore,
        *,
        owner: str = "local",
        lease_ttl: float = 60.0,
    ) -> None:
        """Configure the durable resources shared by runtime attempts."""
        self._store = execution_store
        self._owner = owner
        self._lease_ttl = lease_ttl

    def session(self, context: RuntimeOrchestrationContext) -> Orchestrator:
        """Build the durable session used by recovery-specific runtime operations."""
        return Orchestrator(
            context.execution,
            self._store,
            owner=self._owner,
            ttl=self._lease_ttl,
            emit=context.emit,
            on_suspend=context.on_suspend,
            on_agent_event=context.on_agent_event,
            rules_version=context.rules_version,
        )

    @asynccontextmanager
    async def open(
        self, context: RuntimeOrchestrationContext
    ) -> AsyncGenerator[RuntimeOrchestrationSession, None]:
        """Acquire the run lease and expose durable model, tool, and input boundaries."""
        orchestrator = self.session(context)
        async with orchestrator:
            yield _DurableRuntimeSession(orchestrator)
