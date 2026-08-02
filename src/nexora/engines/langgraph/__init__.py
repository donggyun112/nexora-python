"""The loop driven by LangChain's `create_agent`.

Requires the `langgraph` extra; importing this without it raises `ImportError`.
"""

from .engine import langgraph_loop

__all__ = ["langgraph_loop"]
