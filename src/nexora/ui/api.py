"""FastAPI routes for the Nexora test console."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..runtime import AgentRuntime
from .config import SETTINGS, SYSTEM_PROMPT
from .execution import AgentEvent, capped, stream_attempt
from .policy import permission_controls
from .provider import openrouter_model
from .schemas import ResumeRequest, RunRequest
from .state import STATE
from .tools import DemoTools

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "openrouter_configured": SETTINGS.configured,
        "default_model": SETTINGS.default_model,
        "engine": "plain while + orchestrator",
    }


@router.post("/run")
async def run_agent(request: RunRequest) -> StreamingResponse:
    run_id = request.run_id or f"ui-{uuid.uuid4()}"
    session = STATE.session(run_id)
    session.controls = permission_controls() if request.permission_gate else None

    async def attempt(
        runtime: AgentRuntime, tools: DemoTools, on_event: AgentEvent
    ) -> dict[str, Any]:
        return await runtime.run(
            run_id,
            openrouter_model(request.model),
            tools,
            request.prompt,
            controls=session.controls,
            on_event=on_event,
            system_prompt=SYSTEM_PROMPT,
            should_stop_after_turn=capped,
        )

    return StreamingResponse(
        stream_attempt(run_id, attempt),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/resume")
async def resume_agent(request: ResumeRequest) -> StreamingResponse:
    if request.run_id not in STATE.sessions:
        raise HTTPException(status_code=404, detail="unknown run_id")
    session = STATE.sessions[request.run_id]
    answer = (
        {"type": "text", "text": "approved by the human"}
        if request.approved
        else {"type": "error", "message": "denied by the human"}
    )

    async def attempt(
        runtime: AgentRuntime, tools: DemoTools, on_event: AgentEvent
    ) -> dict[str, Any]:
        return await runtime.resume(
            request.run_id,
            request.tool_call_id,
            answer,
            openrouter_model(request.model),
            tools,
            controls=session.controls,
            on_event=on_event,
            system_prompt=SYSTEM_PROMPT,
            should_stop_after_turn=capped,
        )

    return StreamingResponse(
        stream_attempt(request.run_id, attempt),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
