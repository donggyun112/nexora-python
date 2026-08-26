"""Public facade over the plain planner with optional runtime orchestration."""

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage, messages_to_dict
from nexora_store import (
    ExecutionContext,
    ExecutionStore,
    Transcript,
)

from .background import BackgroundResult
from .contracts import (
    Agent,
    BaseMessage,
    CompactContext,
    EventStream,
    EventType,
    InvokeModel,
    ModelStreamFactory,
    ObservationEventSink,
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
    continuation_cursor,
    decode_continuation,
    decode_cursor_continuation,
    suspension_result_message,
)
from .orchestration import (
    DurableRuntimeOrchestrator,
    RuntimeOrchestrationContext,
    RuntimeOrchestrationSession,
    RuntimeOrchestrator,
    _DurableRuntimeSession,
    _execute_direct,
)
from .orchestrator import AgentSuspended, Orchestrator, StepLog, require_pending_ids
from .subagents import Deliver
from .transcript import TranscriptWriter, active_branch, messages_at, messages_of
from .workspace import ContextualTools, ToolContext, WorkspaceProvider, WorkspaceSeed

__all__ = ["AgentRuntime", "run"]

OnAgentEvent = Callable[[dict[str, Any]], Awaitable[None]]
InputMode = Literal["interactive", "headless"]

_CANCELLED = {
    "type": "error",
    "message": "cancelled by a newer user request",
    "code": "cancelled",
}


async def _invoke_direct(
    step: str, factory: ModelStreamFactory
) -> AsyncIterator[Any]:
    """Expose the loop's direct provider call as a wrappable invocation boundary."""
    del step
    async for chunk in factory():
        yield chunk


@dataclass(slots=True)
class _RuntimeTranscript:
    """Keep one runtime attempt aligned with an append-only conversation branch."""

    writer: TranscriptWriter
    messages: list[BaseMessage]
    uuids: list[str]

    @classmethod
    async def open(
        cls,
        store: Transcript,
        conversation_id: str,
        run_id: str,
        *,
        context: ExecutionContext,
    ) -> "_RuntimeTranscript":
        """Restore a run's active branch and continue writing from its tip."""
        store = store.for_execution(context)
        entries = await store.read(conversation_id)
        branch = active_branch(entries)
        message_entries = [entry for entry in branch if isinstance(entry.get("message"), dict)]
        restored = messages_of(entries)
        writer = TranscriptWriter(
            store,
            conversation_id=conversation_id,
            run_id=run_id,
            parent_uuid=branch[-1]["uuid"] if branch else None,
        )
        await writer.opened()
        return cls(writer, restored, [str(entry["uuid"]) for entry in message_entries])

    async def replace(self, desired: list[BaseMessage]) -> None:
        """Move to the common prefix and append the desired model-visible history."""
        common = 0
        current = messages_to_dict(self.messages)
        wanted = messages_to_dict(desired)
        while common < min(len(current), len(wanted)) and current[common] == wanted[common]:
            common += 1
        if common < len(current):
            await self.writer.rewind(self.uuids[common - 1] if common else None)
            del self.messages[common:]
            del self.uuids[common:]
        await self.append(desired[common:])

    async def append(self, messages: list[BaseMessage]) -> None:
        """Append exact LangChain messages and retain their resulting branch coordinates."""
        for message in messages:
            await self.writer.record(message)
            self.messages.append(message)
            assert self.writer.parent_uuid is not None
            self.uuids.append(self.writer.parent_uuid)


class AgentRuntime:
    """Run the plain planner directly or through an attached orchestrator."""

    def __init__(
        self,
        *,
        execution_store: ExecutionStore | None = None,
        store: StepLog | None = None,
        orchestrator: RuntimeOrchestrator | None = None,
        emit: Emit | None = None,
        event_sink: ObservationEventSink | None = None,
        owner: str = "local",
        lease_ttl: float = 60.0,
        transcript: Transcript | None = None,
        workspace_provider: WorkspaceProvider | None = None,
        workspace_manifest: Mapping[str, Any] | None = None,
        workspace_seed_dirs: Sequence[WorkspaceSeed] = (),
        model_failure_policy: OnModelFailure | None = None,
        compact_context: CompactContext | None = None,
    ) -> None:
        """Initialize runtime collaborators and optional durable compatibility wiring."""
        if execution_store is not None and store is not None:
            raise TypeError("pass either execution_store or the compatibility store argument")
        configured_execution_store = execution_store if execution_store is not None else store
        if configured_execution_store is not None and orchestrator is not None:
            raise TypeError("pass either execution_store or orchestrator, not both")
        if emit is not None and event_sink is not None:
            raise TypeError("pass either emit or event_sink, not both")
        self._runtime_orchestrator = (
            orchestrator
            if orchestrator is not None
            else (
                DurableRuntimeOrchestrator(
                    configured_execution_store, owner=owner, lease_ttl=lease_ttl
                )
                if configured_execution_store is not None
                else None
            )
        )
        self._emit = emit
        self._event_sink = event_sink
        self._transcript = transcript
        self._workspace_provider = workspace_provider
        self._workspace_manifest = workspace_manifest
        self._workspace_seed_dirs = tuple(workspace_seed_dirs)
        self._model_failure_policy = model_failure_policy
        self._compact_context = compact_context
        self.events = RuntimeEvents(emit)

    def background_sink(
        self, run_id: str | ExecutionContext, *, conversation_id: str | None = None
    ) -> Deliver:
        """Return a callback that submits background results to a run's input queue."""

        async def deliver(result: BackgroundResult) -> None:
            arrival = PendingInput(
                "background_result", HumanMessage(result.as_message()), result.task_id
            )
            await self.submit(run_id, arrival, conversation_id=conversation_id)

        return deliver

    async def run(
        self,
        run_id: str | ExecutionContext,
        model: BaseChatModel | Agent,
        tools: Tools | str | None = None,
        prompt: str = "",
        *,
        controls: Controls | None = None,
        on_event: OnAgentEvent | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        prompt_id: str | None = None,
        input_mode: InputMode = "interactive",
        model_identity: str | None = None,
        conversation_id: str | None = None,
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Run one turn directly, or attach the configured orchestration session."""
        model, tools, prompt = _resolve_agent(model, tools, prompt, engine_options)
        execution = _execution_context(run_id)
        if execution.subject is not None:
            supplied_subject = engine_options.get("subject")
            if supplied_subject not in {None, "", execution.subject}:
                raise ValueError("subject cannot differ from the trusted execution context")
            engine_options["subject"] = execution.subject
        context = self._orchestration_context(
            execution,
            conversation_id=conversation_id,
            on_suspend=on_suspend,
            rules_version=rules_version,
            on_event=on_event,
        )
        if self._runtime_orchestrator is None:
            return await self._run_direct(
                execution.run_id,
                model,
                tools,
                prompt,
                controls=controls,
                on_event=on_event,
                prompt_id=prompt_id,
                model_identity=model_identity,
                conversation_id=conversation_id,
                session=None,
                emit=context.emit,
                execution=execution,
                **engine_options,
            )
        async with self._runtime_orchestrator.open(context) as session:
            if not isinstance(session, _DurableRuntimeSession):
                return await self._run_direct(
                    execution.run_id,
                    model,
                    tools,
                    prompt,
                    controls=controls,
                    on_event=on_event,
                    prompt_id=prompt_id,
                    model_identity=model_identity,
                    conversation_id=conversation_id,
                    session=session,
                    emit=context.emit,
                    execution=execution,
                    **engine_options,
                )
            return await self._run_durable(
                session.orchestrator,
                model,
                tools,
                prompt,
                controls=controls,
                on_event=on_event,
                prompt_id=prompt_id,
                input_mode=input_mode,
                model_identity=model_identity,
                conversation_id=conversation_id,
                **engine_options,
            )

    async def _run_durable(
        self,
        orchestrator: Orchestrator,
        model: Any,
        tools: Tools,
        prompt: str,
        *,
        controls: Controls | None,
        on_event: OnAgentEvent | None,
        prompt_id: str | None,
        input_mode: InputMode,
        model_identity: str | None,
        conversation_id: str | None,
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Apply durable continuation semantics around the shared plain planner."""
        run_id = orchestrator.run_id
        conversation = conversation_id or run_id
        history = engine_options.pop("history", None)
        recover_pending = self._transcript is not None
        incoming = PendingInput("user_prompt", HumanMessage(prompt), prompt_id) if prompt else None
        completing: str | None = None
        # Decoded once, before the branch, because every parked state needs the same answer
        # and three copies of "decode, then check for None" was three chances to disagree
        # about what a corrupt record means.
        active = await orchestrator.active_continuation()
        waiting = await self._decode_waiting(active, conversation, orchestrator.context)

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
            recover_pending = False
        else:
            if incoming is not None:
                await orchestrator.enqueue_input(incoming)
            if active.get("state") == "waiting":
                answered = active.get("answers") or {}
                undecided = [
                    (str(request.get("pending_id")), call["id"] or "")
                    for call, request in waiting.parked
                    if (call["id"] or "") not in answered
                ] or [(str(waiting.request["pending_id"]), waiting.call["id"] or "")]
                raise AgentSuspended(undecided[0][0], undecided[0][1], pending=undecided)
            if active.get("state") in {"switching", "resuming"}:
                history = list(waiting.messages) if history is None else history
                completing = waiting.call["id"] or ""
                recover_pending = False

        outcome = await self._drive(
            orchestrator,
            model,
            tools,
            history=history,
            controls=controls,
            on_event=on_event,
            recover_pending=recover_pending,
            conversation_id=conversation,
            model_identity=model_identity,
            **engine_options,
        )
        if completing is not None:
            await orchestrator.complete_continuation(completing)
        return outcome

    async def _run_direct(
        self,
        run_id: str | ExecutionContext,
        model: Any,
        tools: Tools,
        prompt: str,
        *,
        controls: Controls | None,
        on_event: OnAgentEvent | None,
        prompt_id: str | None,
        model_identity: str | None,
        conversation_id: str | None,
        session: RuntimeOrchestrationSession | None,
        emit: Emit | None,
        execution: ExecutionContext,
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Drive the shared planner without durable continuation or step state."""
        run_id = _execution_context(run_id).run_id
        owned = {
            "drain_inputs",
            "discard_inputs",
            "admit_inputs",
            "invoke_model",
            "execute_round",
            "record_messages",
        }.intersection(engine_options)
        if owned:
            names = ", ".join(sorted(owned))
            raise TypeError(f"AgentRuntime owns {names}; call react_loop directly to replace them")

        history = engine_options.pop("history", None)
        conversation = conversation_id or run_id
        transcript = (
            await _RuntimeTranscript.open(
                self._transcript,
                conversation,
                run_id,
                context=execution,
            )
            if self._transcript is not None
            else None
        )
        if transcript is not None:
            if history is None:
                history = list(transcript.messages)
            else:
                await transcript.replace(history)

        queued = [PendingInput("user_prompt", HumanMessage(prompt), prompt_id)] if prompt else []
        input_session = session.inputs if session is not None else None
        if input_session is not None and queued:
            await input_session.submit(queued[0])
            queued.clear()

        async def drain_inputs() -> list[PendingInput]:
            nonlocal queued
            drained, queued = queued, []
            if input_session is not None:
                represented = {message.id for message in history or [] if message.id is not None}
                drained.extend(await input_session.claim(represented))
            return drained

        engine_options.setdefault("on_model_failure", self._model_failure_policy)
        engine_options.setdefault("compact_context", self._compact_context)
        invoke_model: InvokeModel | None = (
            session.wrap_model(_invoke_direct) if session is not None else None
        )
        execute_round = (
            session.wrap_tools(_execute_direct) if session is not None else _execute_direct
        )

        async with self._workspace_tools(run_id, tools) as active_tools:
            outcome = await drive(
                react_loop,
                model,
                active_tools,
                history=history,
                controls=controls,
                emit=emit,
                drain_inputs=drain_inputs,
                discard_inputs=input_session.discard if input_session is not None else None,
                admit_inputs=input_session.admit if input_session is not None else None,
                record_messages=transcript.append if transcript is not None else None,
                invoke_model=invoke_model,
                execute_round=execute_round,
                on_event=on_event,
                model_identity=model_identity,
                **engine_options,
            )
        if transcript is not None:
            await transcript.writer.closed(outcome)
        return outcome

    async def submit(
        self,
        run_id: str | ExecutionContext,
        item: PendingInput,
        *,
        input_mode: InputMode = "interactive",
        conversation_id: str | None = None,
    ) -> PendingInput:
        """Durably route input; user input cancels a parked request in interactive mode."""
        execution = _execution_context(run_id)
        orchestrator = self._orchestrator(execution)
        active = await orchestrator.active_continuation()
        if (
            active is not None
            and active.get("state") == "waiting"
            and item.kind in {"user_prompt", "user_steer"}
            and input_mode == "interactive"
        ):
            waiting = await self._decode_waiting(
                active, conversation_id or execution.run_id, execution
            )
            if waiting is None:
                raise RuntimeError("active suspension has no continuation")
            # Only this branch rewrites the transcript, and only a parked run can reach it — the
            # attempt ended when it suspended, so the lease is free to take.
            async with orchestrator:
                return await self._cancel_and_switch(orchestrator, active, waiting, item)
        # Unleased on purpose. Enqueueing is append-only and keyed by the input's own id, so it
        # needs no exclusion — while a live run holds the lease itself, and taking it here (then
        # dropping it on the way out) moved the token out from under the attempt still writing
        # with it. A background child settling mid-run fenced its own parent.
        return await orchestrator.enqueue_input(item)

    async def resume(
        self,
        run_id: str | ExecutionContext,
        pending_id: str,
        answer: dict[str, Any],
        model: BaseChatModel | Agent,
        tools: Tools | None = None,
        *,
        controls: Controls | None = None,
        on_event: OnAgentEvent | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        model_identity: str | None = None,
        conversation_id: str | None = None,
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Route an answer by its suspension's external `pending_id`.

        A batch park executes nothing until every parked call is decided: an answer for one of
        several parked calls is recorded durably and `AgentSuspended` is raised again with the
        still-undecided requests. The final answer revalidates and finishes every parked call
        in model order, and only then does the run continue.
        """
        model, tools = _resolve_continuation(model, tools, engine_options)
        execution = _execution_context(run_id)
        conversation = conversation_id or execution.run_id
        async with self._orchestrator(
            execution, on_suspend, rules_version, on_event
        ) as orchestrator:
            active = await orchestrator.active_continuation()
            waiting = await self._decode_waiting(active, conversation, execution)
            parked = list(waiting.parked) if waiting is not None else []
            by_pending = require_pending_ids(parked)
            if (
                waiting is None
                or active is None
                or active.get("state") not in {"waiting", "finalizing", "resuming"}
                or pending_id not in by_pending
            ):
                raise LookupError(f"no active suspension for pending id {pending_id!r}")
            if execution.subject is not None and waiting.subject not in {"", execution.subject}:
                raise ValueError("subject cannot differ from the suspended execution context")
            supplied_subject = engine_options.get("subject")
            if execution.subject is not None and supplied_subject not in {
                None,
                "",
                execution.subject,
            }:
                raise ValueError("subject cannot differ from the trusted execution context")
            subject = execution.subject or str(supplied_subject or waiting.subject)
            engine_options["subject"] = subject
            active = await orchestrator.record_suspension_answer(
                by_pending[pending_id]["id"] or "", answer
            )
            answers = dict(active.get("answers") or {})
            undecided = [
                (str(request.get("pending_id")), call["id"] or "")
                for call, request in parked
                if (call["id"] or "") not in answers
            ]
            if undecided:
                raise AgentSuspended(undecided[0][0], undecided[0][1], pending=undecided)
            return await self._finalize_suspension(
                orchestrator,
                active,
                waiting,
                answers,
                model,
                tools,
                controls=controls,
                on_event=on_event,
                conversation=conversation,
                model_identity=model_identity,
                subject=subject,
                aborted=engine_options.get("aborted", lambda: False),
                engine_options=engine_options,
            )

    async def _finalize_suspension(
        self,
        orchestrator: Orchestrator,
        active: dict[str, Any],
        waiting: Any,
        answers: dict[str, Any],
        model: Any,
        tools: Tools,
        *,
        controls: Controls | None,
        on_event: OnAgentEvent | None,
        conversation: str,
        model_identity: str | None,
        subject: str,
        aborted: Aborted,
        engine_options: dict[str, Any],
    ) -> dict[str, Any]:
        """Idempotently finish a fully answered continuation after resume or recovery."""
        parked = list(waiting.parked)
        tool_call_id = waiting.call["id"] or ""
        drive_options = {**engine_options, "aborted": aborted}
        async with self._workspace_tools(orchestrator.run_id, tools) as active_tools:
            items: list[PendingInput] = []
            for index, (call, request) in enumerate(parked):
                result = await orchestrator.resume_effect(
                    active_tools,
                    call,
                    answers[call["id"] or ""],
                    request,
                    waiting.rules_version,
                    aborted=aborted,
                    turn=waiting.turn,
                    controls=controls,
                    ctx=Ctx(
                        turn=waiting.turn,
                        messages=list(waiting.messages),
                        subject=subject,
                    ),
                    park=False,
                )
                if result.get("type") == "suspend":
                    await orchestrator.repark_suspension(
                        [
                            (candidate, result if position == index else old_request)
                            for position, (candidate, old_request) in enumerate(parked)
                        ],
                        list(waiting.messages),
                        waiting.completed,
                        turn=waiting.turn,
                        subject=subject,
                        answers={
                            key: value
                            for key, value in answers.items()
                            if key != (call["id"] or "")
                        },
                        reissued=(call, result),
                    )
                items.append(
                    PendingInput(
                        "resume_result",
                        suspension_result_message(
                            call["id"] or "", result, name=call.get("name", "")
                        ),
                        f"resume:{request.get('pending_id')}",
                    )
                )
            await orchestrator.continue_with_input(
                tool_call_id, active["continuation"], items
            )
            outcome = await self._drive_active(
                orchestrator,
                model,
                active_tools,
                history=waiting.messages,
                controls=controls,
                on_event=on_event,
                recover_pending=False,
                conversation_id=conversation,
                model_identity=model_identity,
                **drive_options,
            )
        await orchestrator.complete_continuation(tool_call_id)
        return outcome

    async def recover(
        self,
        run_id: str | ExecutionContext,
        history: list[BaseMessage],
        model: BaseChatModel | Agent,
        tools: Tools | None = None,
        *,
        controls: Controls | None = None,
        aborted: Aborted = lambda: False,
        retry_running: bool = True,
        on_event: OnAgentEvent | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        model_identity: str | None = None,
        conversation_id: str | None = None,
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Recover an interrupted tool round and continue without replaying its model turn."""
        model, tools = _resolve_continuation(model, tools, engine_options)
        execution = _execution_context(run_id)
        if execution.subject is not None:
            supplied_subject = engine_options.get("subject")
            if supplied_subject not in {None, "", execution.subject}:
                raise ValueError("subject cannot differ from the trusted execution context")
            engine_options["subject"] = execution.subject
        conversation = conversation_id or execution.run_id
        async with self._orchestrator(
            execution, on_suspend, rules_version, on_event
        ) as orchestrator:
            active = await orchestrator.active_continuation()
            waiting = await self._decode_waiting(active, conversation, execution)
            if active is not None and waiting is not None and active.get("state") in {
                "waiting",
                "finalizing",
                "resuming",
            }:
                parked = list(waiting.parked)
                answers = dict(active.get("answers") or {})
                undecided = [
                    (str(request["pending_id"]), call["id"] or "")
                    for call, request in parked
                    if (call["id"] or "") not in answers
                ]
                if undecided:
                    raise AgentSuspended(undecided[0][0], undecided[0][1], pending=undecided)
                subject = execution.subject or str(engine_options.get("subject") or waiting.subject)
                engine_options["subject"] = subject
                return await self._finalize_suspension(
                    orchestrator,
                    active,
                    waiting,
                    answers,
                    model,
                    tools,
                    controls=controls,
                    on_event=on_event,
                    conversation=conversation,
                    model_identity=model_identity,
                    subject=subject,
                    aborted=aborted,
                    engine_options=engine_options,
                )
            async with self._workspace_tools(execution.run_id, tools) as active_tools:
                recovered = await orchestrator.recover_pending(
                    history,
                    active_tools,
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
                return await self._drive_active(
                    orchestrator,
                    model,
                    active_tools,
                    history=history,
                    controls=controls,
                    on_event=on_event,
                    recover_pending=False,
                    conversation_id=conversation,
                    aborted=aborted,
                    model_identity=model_identity,
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
        recover_pending: bool,
        conversation_id: str,
        **engine_options: Any,
    ) -> dict[str, Any]:
        """Acquire one attempt workspace and drive all tool effects through its context."""
        async with self._workspace_tools(orchestrator.run_id, tools) as active_tools:
            return await self._drive_active(
                orchestrator,
                model,
                active_tools,
                history=history,
                controls=controls,
                on_event=on_event,
                recover_pending=recover_pending,
                conversation_id=conversation_id,
                **engine_options,
            )

    async def _drive_active(
        self,
        orchestrator: Orchestrator,
        model: Any,
        tools: Tools,
        *,
        history: list[BaseMessage] | None,
        controls: Controls | None,
        on_event: OnAgentEvent | None,
        recover_pending: bool,
        conversation_id: str,
        **engine_options: Any,
    ) -> dict[str, Any]:
        if any(
            name in engine_options
            for name in (
                "drain_inputs",
                "discard_inputs",
                "admit_inputs",
                "invoke_model",
                "record_messages",
            )
        ):
            raise TypeError(
                "AgentRuntime owns drain_inputs, discard_inputs, admit_inputs, invoke_model, "
                "and record_messages"
            )
        engine_options.setdefault("on_model_failure", self._model_failure_policy)
        engine_options.setdefault("compact_context", self._compact_context)

        transcript = (
            await _RuntimeTranscript.open(
                self._transcript,
                conversation_id,
                orchestrator.run_id,
                context=orchestrator.context,
            )
            if self._transcript is not None
            else None
        )
        if transcript is not None:
            if history is None:
                history = list(transcript.messages)
            else:
                await transcript.replace(history)

        prefetched: list[PendingInput] = []

        async def drain_inputs() -> list[PendingInput]:
            # Rebuilt per drain, not hoisted: `history` belongs to the caller, and a set computed
            # once would answer for the transcript as it looked before the run rather than now.
            claimed = list(prefetched)
            prefetched.clear()
            claimed += await orchestrator.claim_inputs(
                {message.id for message in history or [] if message.id is not None}
            )
            return claimed

        try:
            if recover_pending:
                prefetched.extend(
                    await orchestrator.claim_inputs(
                        {message.id for message in history or [] if message.id is not None}
                    )
                )
                recovery_history = [
                    *(history or []),
                    *(
                        item.message
                        for item in prefetched
                        if isinstance(item.message, ToolMessage)
                    ),
                ]
                await self._recover_pending_inputs(
                    orchestrator,
                    recovery_history,
                    tools,
                    controls,
                    engine_options,
                )
            outcome = await drive(
                react_loop,
                model,
                tools,
                history=history,
                controls=controls,
                emit=orchestrator.emit,
                drain_inputs=drain_inputs,
                discard_inputs=orchestrator.discard_inputs,
                admit_inputs=orchestrator.admit_inputs,
                record_messages=transcript.append if transcript is not None else None,
                invoke_model=orchestrator.invoke_model,
                execute_round=orchestrator.execute_round,
                on_event=on_event,
                **engine_options,
            )
        except AgentSuspended:
            # The executor prunes unexecuted calls and includes results completed before the
            # suspension. Persist that canonical continuation rather than the model's wider batch.
            if transcript is not None:
                active = await orchestrator.active_continuation()
                waiting = decode_continuation(active.get("continuation")) if active else None
                if waiting is not None:
                    await transcript.replace(list(waiting.messages))
                    await orchestrator.compact_suspension(
                        waiting.call["id"] or "",
                        conversation_id=conversation_id,
                        leaf_uuid=transcript.writer.parent_uuid,
                    )
            raise
        if transcript is not None:
            await transcript.writer.closed(outcome)
        return outcome

    @asynccontextmanager
    async def _workspace_tools(
        self, run_id: str, tools: Tools
    ) -> AsyncGenerator[Tools, None]:
        """Inject a provider-acquired session and guarantee end-of-attempt cleanup."""
        if self._workspace_provider is None:
            yield tools
            return
        if not isinstance(tools, ContextualTools):
            raise TypeError(
                "workspace_provider requires ContextualTools with get_context() and with_context()"
            )
        context = tools.get_context()
        if not isinstance(context, ToolContext):
            raise TypeError("ContextualTools.get_context() must return ToolContext")
        workspace = await self._workspace_provider.acquire(
            run_id=run_id,
            base_workdir=context.workdir,
            manifest=self._workspace_manifest,
            seed_dirs=self._workspace_seed_dirs,
        )
        rebound = tools.with_context(
            ToolContext(
                workdir=str(workspace.root),
                workspace=workspace,
                metadata=context.metadata,
            )
        )
        try:
            yield rebound
        finally:
            await workspace.cleanup()

    async def _decode_waiting(
        self,
        active: dict[str, Any] | None,
        conversation_id: str,
        context: ExecutionContext,
    ) -> Any:
        """Hydrate either a legacy snapshot or a compact transcript continuation."""
        if active is None:
            return None
        payload = active.get("continuation")
        if not isinstance(payload, dict):
            return None
        cursor = continuation_cursor(payload)
        if cursor is None:
            return decode_continuation(payload)
        if self._transcript is None:
            raise RuntimeError("compact suspension requires the configured transcript store")
        if cursor.conversation_id != conversation_id:
            raise ValueError(
                "conversation_id does not match the suspension transcript cursor: "
                f"{conversation_id!r} != {cursor.conversation_id!r}"
            )
        transcript = self._transcript.for_execution(context)
        entries = await transcript.read(cursor.conversation_id)
        return decode_cursor_continuation(payload, messages_at(entries, cursor.leaf_uuid))

    async def _recover_pending_inputs(
        self,
        orchestrator: Orchestrator,
        history: list[BaseMessage],
        tools: Tools,
        controls: Controls | None,
        engine_options: dict[str, Any],
    ) -> None:
        """Turn an unanswered transcript tool round into durable queued results."""
        recovered = await orchestrator.recover_pending(
            history,
            tools,
            controls=controls,
            aborted=engine_options.get("aborted", lambda: False),
            retry_running=True,
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

    def _orchestrator(
        self,
        run_id: str | ExecutionContext,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        on_event: OnAgentEvent | None = None,
    ) -> Orchestrator:
        if not isinstance(self._runtime_orchestrator, DurableRuntimeOrchestrator):
            raise RuntimeError(
                "this operation requires DurableRuntimeOrchestrator or AgentRuntime(store=...)"
            )
        return self._runtime_orchestrator.session(
            self._orchestration_context(
                run_id,
                on_suspend=on_suspend,
                rules_version=rules_version,
                on_event=on_event,
            )
        )

    def _orchestration_context(
        self,
        run_id: str | ExecutionContext,
        *,
        conversation_id: str | None = None,
        on_suspend: OnSuspend | None = None,
        rules_version: str = "",
        on_event: OnAgentEvent | None = None,
    ) -> RuntimeOrchestrationContext:
        """Build the stable context passed to an attached orchestrator."""
        execution = _execution_context(run_id)
        publisher = self._emit
        if self._event_sink is not None:
            publisher = EventStream(
                self._event_sink,
                session_id=execution.session_id or execution.run_id,
                thread_id=execution.namespace or conversation_id or execution.run_id,
                run_id=execution.run_id,
            )
        return RuntimeOrchestrationContext(
            execution=execution,
            emit=publisher,
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
        normalized, cancelled = await orchestrator.cancel_and_switch(
            list(waiting.parked),
            active["continuation"],
            dict(_CANCELLED),
            incoming,
        )
        for call in cancelled:
            await orchestrator.emit(
                EventType.TOOL_REQUEST_CANCELLED,
                {
                    "turn": waiting.turn,
                    "call_id": call["id"],
                    "name": call["name"],
                    "input": call["args"],
                    "reason": dict(_CANCELLED),
                    "replacement_input_id": normalized[-1].origin_id,
                    "event_id": "\x1f".join(
                        [str(EventType.TOOL_REQUEST_CANCELLED), str(waiting.turn), str(call["id"])]
                    ),
                },
            )
        return normalized[-1]


async def run(
    model: BaseChatModel | Agent,
    tools: Tools | str | None = None,
    prompt: str = "",
    *,
    run_id: str = "default",
    runtime: AgentRuntime | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Convenience entry point for one direct Nexora agent turn."""
    active = runtime if runtime is not None else AgentRuntime()
    return await active.run(run_id, model, tools, prompt, **options)


def _resolve_agent(
    model: BaseChatModel | Agent,
    tools: Tools | str | None,
    prompt: str,
    engine_options: dict[str, Any],
) -> tuple[BaseChatModel, Tools, str]:
    """Expand an agent definition while preserving the legacy model/tools call shape."""
    if isinstance(model, Agent):
        if tools is not None and not isinstance(tools, str):
            raise TypeError("Agent owns tools; pass the user prompt as the next argument")
        if isinstance(tools, str):
            if prompt:
                raise TypeError("prompt was provided both positionally and by keyword")
            prompt = tools
        if "system_prompt" in engine_options:
            raise TypeError("Agent owns system_prompt")
        engine_options["system_prompt"] = model.system_prompt
        return model.model, model.tools, prompt
    if tools is None or isinstance(tools, str):
        raise TypeError("model execution requires a Tools instance")
    return model, tools, prompt


def _resolve_continuation(
    model: BaseChatModel | Agent,
    tools: Tools | None,
    engine_options: dict[str, Any],
) -> tuple[BaseChatModel, Tools]:
    """Expand an agent definition for a continuation, which carries no new prompt."""
    model, tools, prompt = _resolve_agent(model, tools, "", engine_options)
    if prompt:
        raise TypeError("a continuation takes no prompt")
    return model, tools


def _execution_context(value: str | ExecutionContext) -> ExecutionContext:
    """Normalize the compatibility run-id form at the host boundary."""
    return value if isinstance(value, ExecutionContext) else ExecutionContext(value)
