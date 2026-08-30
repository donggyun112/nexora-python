"""OpenAI-compatible chat model used by Nexora's planner."""

from .chat import ChatModel, openrouter, xai
from .dsml import DsmlFilter, parse_dsml_tool_calls, recover_dsml_chunks, strip_dsml

__all__ = [
    "ChatModel",
    "DsmlFilter",
    "openrouter",
    "parse_dsml_tool_calls",
    "recover_dsml_chunks",
    "strip_dsml",
    "xai",
]
