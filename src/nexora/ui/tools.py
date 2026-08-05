"""Harmless effects exposed by the local test UI."""

from __future__ import annotations

import asyncio
from typing import Any


class DemoTools:
    """Visible effects for checking execution and ordering."""

    def __init__(self) -> None:
        self.notes: dict[str, str] = {}

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        args = arguments if isinstance(arguments, dict) else {}
        if name == "echo":
            return {"type": "text", "text": str(args.get("text", ""))}
        if name == "remember_note":
            key = str(args.get("key", "default"))
            value = str(args.get("value", ""))
            self.notes[key] = value
            return {"type": "text", "text": f"remembered {key}={value}"}
        if name == "recall_note":
            key = str(args.get("key", "default"))
            recalled = self.notes.get(key)
            return {
                "type": "text" if recalled is not None else "error",
                "text" if recalled is not None else "message": recalled or f"no note named {key}",
            }
        if name == "runtime_clock":
            return {"type": "text", "text": str(asyncio.get_running_loop().time())}
        return {"type": "error", "message": f"unknown tool: {name}"}

    def get(self, name: str) -> dict[str, Any] | None:
        return next((item for item in self.list() if item["name"] == name), None)

    def list(self) -> list[dict[str, Any]]:
        string = {"type": "string"}
        return [
            {
                "name": "echo",
                "description": "Echo text through a durable tool effect.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": string},
                    "required": ["text"],
                },
            },
            {
                "name": "remember_note",
                "description": "Store a note in this test session.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": string, "value": string},
                    "required": ["key", "value"],
                },
            },
            {
                "name": "recall_note",
                "description": "Read a note stored in this test session.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": string},
                    "required": ["key"],
                },
            },
            {
                "name": "runtime_clock",
                "description": "Return the monotonic runtime clock through a durable effect.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
