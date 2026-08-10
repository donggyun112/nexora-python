"""FastAPI routes for the Nexora test console."""

from __future__ import annotations

import uuid
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from nexora.contracts import ToolCall, Tools
from nexora.runtime import AgentRuntime

from .config import SETTINGS, SYSTEM_PROMPT
from .execution import AgentEvent, capped, stream_attempt
from .policy import permission_controls
from .provider import openrouter_model
from .recording import Recorder, history_of
from .schemas import AttachRequest, CancelRequest, RecoverRequest, ResumeRequest, RunRequest
from .state import STATE

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, Any]:
    """Return service health and provider configuration status."""
    return {
        "ok": True,
        "openrouter_configured": SETTINGS.configured,
        "default_model": SETTINGS.default_model,
        "engine": "plain while + orchestrator",
    }


@router.get("/steps/{run_id}")
async def steps(run_id: str) -> dict[str, Any]:
    """Return the persisted step states for a run."""
    return {"run_id": run_id, "steps": STATE.step_store.snapshot(run_id)}


@router.post("/run")
async def run_agent(request: RunRequest) -> StreamingResponse:
    """Start an agent run and stream newline-delimited events."""
    run_id = request.run_id or f"ui-{uuid.uuid4()}"
    session = STATE.session(run_id)
    session.controls = permission_controls() if request.permission_gate else None
    prompt_id = f"{run_id}:prompt:{uuid.uuid4()}"
    if request.fault_after_step_commit:
        session.recovery_history = None
        STATE.step_store.arm(run_id)

    async def attempt(
        runtime: AgentRuntime, tools: Tools, on_event: AgentEvent
    ) -> dict[str, Any]:
        """Execute the requested run through the shared runtime."""
        text: list[str] = []
        calls: list[ToolCall] = []

        async def capture(event: dict[str, Any]) -> None:
            await on_event(event)
            if not request.fault_after_step_commit:
                return
            if event.get("type") == "text":
                text.append(str(event.get("text", "")))
            if event.get("type") == "tool_call":
                calls.append(
                    cast(
                        ToolCall,
                        {
                            "id": event.get("id"),
                            "name": event.get("name"),
                            "args": event.get("input", {}),
                            "type": "tool_call",
                        },
                    )
                )
                session.recovery_history = [
                    HumanMessage(request.prompt, id=prompt_id),
                    AIMessage(content="".join(text), tool_calls=list(calls)),
                ]

        try:
            return await runtime.run(
                run_id,
                openrouter_model(request.model),
                tools,
                request.prompt,
                controls=session.controls,
                on_event=capture,
                system_prompt=SYSTEM_PROMPT,
                should_stop_after_turn=capped,
                prompt_id=prompt_id,
            )
        finally:
            STATE.step_store.disarm(run_id)

    return StreamingResponse(
        stream_attempt(run_id, attempt, model=request.model),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/recover")
async def recover_agent(request: RecoverRequest) -> StreamingResponse:
    """Recover a run after a simulated post-commit worker crash."""
    if request.run_id not in STATE.sessions:
        raise HTTPException(status_code=404, detail="unknown run_id")
    session = STATE.sessions[request.run_id]
    if session.recovery_history is None:
        raise HTTPException(status_code=409, detail="run has no recoverable step crash")

    async def attempt(
        runtime: AgentRuntime, tools: Tools, on_event: AgentEvent
    ) -> dict[str, Any]:
        """Recover the requested run through the shared runtime."""
        outcome = await runtime.recover(
            request.run_id,
            session.recovery_history or [],
            openrouter_model(request.model),
            tools,
            controls=session.controls,
            on_event=on_event,
            retry_running=False,
            system_prompt=SYSTEM_PROMPT,
            should_stop_after_turn=capped,
        )
        session.recovery_history = None
        return outcome

    return StreamingResponse(
        stream_attempt(request.run_id, attempt, model=request.model),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/resume")
async def resume_agent(request: ResumeRequest) -> StreamingResponse:
    """Resume a suspended tool call with an operator decision."""
    if request.run_id not in STATE.sessions:
        raise HTTPException(status_code=404, detail="unknown run_id")
    session = STATE.sessions[request.run_id]
    answer = (
        {"type": "text", "text": "approved by the human"}
        if request.approved
        else {"type": "error", "message": "denied by the human"}
    )

    async def attempt(
        runtime: AgentRuntime, tools: Tools, on_event: AgentEvent
    ) -> dict[str, Any]:
        """Resume the requested run through the shared runtime."""
        return await runtime.resume(
            request.run_id,
            request.pending_id,
            answer,
            openrouter_model(request.model),
            tools,
            controls=session.controls,
            on_event=on_event,
            system_prompt=SYSTEM_PROMPT,
            should_stop_after_turn=capped,
        )

    return StreamingResponse(
        stream_attempt(request.run_id, attempt, model=request.model),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tasks/{run_id}")
async def tasks(run_id: str) -> dict[str, Any]:
    """Return managed tasks and independent children for an agent run.

    Args:
        run_id: Parent agent run identifier.

    Returns:
        Mapping containing background task records and independent child addresses.
    """
    session = STATE.sessions.get(run_id)
    if session is None:
        return {"run_id": run_id, "background": [], "independent": []}
    return {
        "run_id": run_id,
        "background": session.tasks.list(),
        "independent": list(session.opened),
    }


@router.post("/tasks/cancel")
async def cancel_task(request: CancelRequest) -> dict[str, Any]:
    """Cancel a managed background task.

    Args:
        request: Parent run and task identifiers.

    Returns:
        Cancelled task identifier and the updated task list.

    Raises:
        HTTPException: If the run is unknown or the task is not running.
    """
    session = STATE.sessions.get(request.run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    if not session.tasks.cancel(request.task_id):
        raise HTTPException(status_code=409, detail="no running task with that id")
    return {"cancelled": request.task_id, "background": session.tasks.list()}


@router.post("/attach")
async def attach_agent(request: AttachRequest) -> StreamingResponse:
    """Submit a prompt to an independent child run.

    Args:
        request: Child run identifier, prompt, and model selection.

    Returns:
        Newline-delimited event stream for the child attempt.

    Note:
        Transcript continuity is provided by the console's recorder and restored as explicit
        runtime history.
    """

    async def attempt(
        runtime: AgentRuntime, tools: Tools, on_event: AgentEvent
    ) -> dict[str, Any]:
        """Execute one attempt for the independent child run."""
        already_said = await history_of(STATE.transcripts, request.run_id)
        recorder = await Recorder.open(STATE.transcripts, request.run_id, request.prompt)

        async def watched(event: dict[str, Any]) -> None:
            await recorder.observe(event)
            await on_event(event)

        try:
            return await runtime.run(
                request.run_id,
                openrouter_model(request.model),
                tools,
                request.prompt,
                on_event=watched,
                system_prompt=SYSTEM_PROMPT,
                should_stop_after_turn=capped,
                history=already_said,
            )
        finally:
            await recorder.closed()

    return StreamingResponse(
        stream_attempt(request.run_id, attempt, model=request.model),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/transcript/{run_id}")
async def transcript(run_id: str) -> dict[str, Any]:
    """What the console recorded of this run's conversation.

    The panel for the thing that has no panel. `history` is what an attach hands back, so an
    operator debugging a reattached agent that answers as a stranger can see whether the problem
    is the recording or the model.
    """
    return {
        "run_id": run_id,
        "entries": len(await STATE.transcripts.read(run_id)),
        "history": [
            {"role": type(message).__name__, "content": str(message.content)[:400]}
            for message in await history_of(STATE.transcripts, run_id)
        ],
    }
