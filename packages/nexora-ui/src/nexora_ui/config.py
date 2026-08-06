"""UI configuration and secret loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

UI_ROOT = Path(__file__).resolve().parent
"""Where `static/` lives — inside the package, so it ships in the wheel."""

ENV_FILE = UI_ROOT.parents[1] / ".env"
"""Beside the manifest, not inside the module: `.env` is developer configuration and does not belong
in a wheel. An installed copy finds nothing here and falls back to the environment, which is how a
deployed process is configured anyway. Named rather than inlined so a test can check that
`.env.example` documents the place this actually reads."""

load_dotenv(ENV_FILE)


@dataclass(frozen=True, slots=True)
class Settings:
    ui_root: Path = UI_ROOT
    default_model: str = "openai/gpt-4o-mini"
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    public_url: str = "http://127.0.0.1:8790"

    @property
    def configured(self) -> bool:
        return bool(self.openrouter_api_key)

    def require_api_key(self) -> str:
        if not self.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is missing from packages/nexora-ui/.env")
        return self.openrouter_api_key

    @classmethod
    def from_environment(cls) -> Settings:
        # OPEN_ROTURE is the spelling used by the imported bug-case fixture.
        key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENROUTER_KEY")
            or os.getenv("OPEN_ROTURE")
            or ""
        )
        return cls(
            default_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            openrouter_api_key=key,
        )


SETTINGS = Settings.from_environment()

SYSTEM_PROMPT = """You are the Nexora runtime test agent. Be concise.
Use a tool whenever the user explicitly asks you to echo text, remember or recall a note, inspect
the runtime clock, or simulate an API failure. The simulate_api_failure tool is a harmless test
fixture: call it immediately when explicitly requested and never ask the user for permission.
Never claim a tool ran when it did not. Permission approval happens before a tool executes and is
owned by the runtime, not by a tool result."""
