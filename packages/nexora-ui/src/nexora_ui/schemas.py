"""HTTP request contracts for the local UI."""

from pydantic import BaseModel, Field

from .config import SETTINGS


class RunRequest(BaseModel):
    """Request payload for starting an agent run."""
    prompt: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)
    run_id: str | None = None
    permission_gate: bool = False
    fault_after_step_commit: bool = False


class ResumeRequest(BaseModel):
    """Request payload for resuming a suspended tool call."""
    run_id: str = Field(min_length=1)
    pending_id: str = Field(min_length=1)
    approved: bool
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)


class RecoverRequest(BaseModel):
    """Request payload for recovering an interrupted run."""
    run_id: str = Field(min_length=1)
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)


class CancelRequest(BaseModel):
    """Request payload for cancelling a managed background task."""
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)


class AttachRequest(BaseModel):
    """Request payload for prompting an independent child run."""
    run_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)
