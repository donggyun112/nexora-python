"""Nexora's public Python package."""

from importlib.metadata import PackageNotFoundError, version

from .contracts import BaseMessage, ToolCall, Tools
from .engines.plain import react_loop

try:
    __version__ = version("nexora")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"

__all__ = ["BaseMessage", "ToolCall", "Tools", "__version__", "react_loop"]
