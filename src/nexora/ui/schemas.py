"""HTTP request contracts for the local UI."""

from pydantic import BaseModel, Field

from .config import SETTINGS


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)
    run_id: str | None = None
    permission_gate: bool = False


class ResumeRequest(BaseModel):
    run_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    approved: bool
    model: str = Field(default=SETTINGS.default_model, min_length=1, max_length=200)
