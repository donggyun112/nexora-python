"""HTTP request contracts for the local UI."""

from pydantic import BaseModel, Field

from .config import SETTINGS


class RunRequest(BaseModel):
    """Validate a request to start an agent run."""
    prompt: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)
    run_id: str | None = None
    permission_gate: bool = False
    fault_after_step_commit: bool = False


class ResumeRequest(BaseModel):
    """Validate an operator decision for a suspended run."""
    run_id: str = Field(min_length=1)
    pending_id: str = Field(min_length=1)
    approved: bool
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)


class RecoverRequest(BaseModel):
    """Validate a request to recover an interrupted run."""
    run_id: str = Field(min_length=1)
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)
