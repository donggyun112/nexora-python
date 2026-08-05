"""A scripted chat model and a toy tool box, so the examples run with no API key.

Swap `scripted(...)` for a real model and `Files` for your own executor — nothing else in the
examples changes, because the runtime only asks a model to stream and a tool box to execute.
"""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


def says(text: str = "", *calls: dict[str, Any]) -> AIMessage:
    """One scripted turn: some text, and any tool calls the model asks for."""
    return AIMessage(content=text, tool_calls=list(calls))


def calling(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


class Scripted(GenericFakeChatModel):
    """Replays scripted turns as a stream.

    `_stream` is overridden because the stock fake cannot stream a turn that carries tool calls.
    A real provider needs none of this.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        reply = next(self.messages)
        assert isinstance(reply, AIMessage)
        if reply.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls)
            )
            return
        for index, word in enumerate(str(reply.content).split(" ")):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=word if index == 0 else f" {word}")
            )


def scripted(*turns: AIMessage) -> Scripted:
    return Scripted(messages=iter(turns))


class Files:
    """A tool box: three named tools over a dict. `Tools` is this protocol and nothing more."""

    def __init__(self, contents: dict[str, str] | None = None) -> None:
        self.contents = contents or {"notes.md": "ssn 123-45-6789 on file"}
        self.ran: list[str] = []

    async def execute(self, name: str, call_id: str, args: Any) -> dict[str, Any]:
        self.ran.append(name)
        if name == "read":
            return {"type": "text", "text": self.contents.get(args["path"], "not found")}
        if name == "write":
            self.contents[args["path"]] = args["text"]
            return {"type": "text", "text": f"wrote {args['path']}"}
        return {"type": "text", "text": f"deleted {args['path']}"}

    def get(self, name: str) -> dict[str, Any] | None:
        """Each tool's own declaration. `is_concurrency_safe` is how a tool opts into batching."""
        return {"name": name, "is_concurrency_safe": name == "read"}

    def list(self) -> list[dict[str, Any]]:
        return [
            {"name": "read", "description": "read a file", "parameters": {}},
            {"name": "write", "description": "write a file", "parameters": {}},
            {"name": "delete", "description": "delete a file", "parameters": {}},
        ]
