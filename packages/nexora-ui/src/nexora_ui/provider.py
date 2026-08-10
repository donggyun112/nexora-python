"""OpenRouter model construction, isolated from the UI transport."""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import SETTINGS, Settings

REASONING = "reasoning_details"
"""OpenRouter's structured reasoning, carrying the signature a replayed turn is checked against."""


class OpenRouterChat(ChatOpenAI):
    """`ChatOpenAI` with the one field it drops on purpose put back.

    `ChatOpenAI` targets the official OpenAI schema and documents that a third party's extra
    response fields are neither extracted nor preserved. For OpenRouter that discards the model's
    reasoning entirely — the loop never sees it, and cannot return it on the next turn. Both
    directions are restored here rather than in the loop: which provider is in use, and what it
    puts beside the standard schema, is the host's business.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        """Keep the reasoning a streamed chunk carries, in both shapes it is needed in."""
        generated = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        delta = ((chunk.get("choices") or [{}])[0] or {}).get("delta") or {}
        if generated is None:
            return generated
        if reasoning := delta.get(REASONING):
            # Verbatim, signature and all: this is what goes back on the next request.
            generated.message.additional_kwargs[REASONING] = reasoning
        if thought := delta.get("reasoning"):
            # And again as a standard content block, which is the only reasoning the engine reads.
            # Translating here is what keeps `reasoning_details` out of the loop's vocabulary.
            # LangChain drops these blocks when it serializes the turn, so nothing is sent twice.
            spoken = generated.message.content
            generated.message.content = [
                # `index` is what makes chunk addition fold these into one block instead of one
                # per delta; without it a turn's reasoning arrives as a hundred stutters.
                {"type": "reasoning", "reasoning": thought, "index": 0},
                *([{"type": "text", "text": spoken}] if isinstance(spoken, str) and spoken else []),
            ]
        return generated

    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Send the reasoning back out on the turn it belongs to."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for message, sent in zip(messages, payload.get("messages") or [], strict=False):
            if reasoning := message.additional_kwargs.get(REASONING):
                sent[REASONING] = reasoning
        return payload


def openrouter_model(name: str, settings: Settings = SETTINGS) -> ChatOpenAI:
    """Create a streaming OpenRouter chat model."""
    return OpenRouterChat(
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
        # Asked for, or no provider sends any: reasoning is opt-in on OpenRouter. `low` because the
        # console is here to show the loop, not to buy the deepest answer.
        extra_body={"reasoning": {"effort": "low"}},
    )
