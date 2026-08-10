"""OpenRouter model construction, isolated from the UI transport."""

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import SETTINGS, Settings


def openrouter_model(name: str, settings: Settings = SETTINGS) -> ChatOpenAI:
    """Create a streaming OpenRouter chat model."""
    return ChatOpenAI(
        model=name,
        api_key=SecretStr(settings.require_api_key()),
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": settings.public_url,
            "X-Title": "Nexora Durable Agent Lab",
        },
        max_retries=1,
        streaming=True,
        timeout=60,
    )
