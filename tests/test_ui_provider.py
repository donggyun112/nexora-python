"""The console's provider keeps the reasoning `ChatOpenAI` is documented to discard.

Both hooks override private LangChain methods, so a rename upstream turns them into dead code that
raises nothing and changes nothing — which is how the outbound half shipped broken once already.
These pin the round trip in both directions without touching the network.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import AIMessageChunk as Chunk
from nexora_ui.provider import OpenRouterChat
from pydantic import SecretStr

DETAILS: list[dict[str, Any]] = [
    {
        "type": "reasoning.text",
        "text": "먼저 파일을 읽자",
        "format": "anthropic-claude-v1",
        "signature": "sig",
    }
]


@pytest.fixture
def model() -> OpenRouterChat:
    return OpenRouterChat(model="anthropic/claude-haiku-4.5", api_key=SecretStr("test-key"))


def test_a_streamed_chunk_keeps_its_reasoning(model: OpenRouterChat) -> None:
    """OpenRouter streams reasoning in the delta; `ChatOpenAI` copies only fields it knows."""
    delta = {"role": "assistant", "content": "읽을게요", "reasoning_details": DETAILS}
    chunk = {"choices": [{"delta": delta}]}

    generated = model._convert_chunk_to_generation_chunk(chunk, Chunk, None)

    assert generated is not None
    assert generated.message.additional_kwargs["reasoning_details"] == DETAILS


def test_reasoning_also_arrives_as_the_block_the_engine_reads(model: OpenRouterChat) -> None:
    """The loop reads `reasoning` content blocks and nothing else — see `test_loop.py`.

    Translating here is what keeps the provider's own spelling out of the engine.
    """
    delta = {"role": "assistant", "content": "말", "reasoning": "생각"}

    chunk = {"choices": [{"delta": delta}]}

    generated = model._convert_chunk_to_generation_chunk(chunk, Chunk, None)

    assert generated is not None
    assert generated.message.content == [
        {"type": "reasoning", "reasoning": "생각", "index": 0},
        {"type": "text", "text": "말"},
    ]


def test_streamed_reasoning_folds_into_one_block(model: OpenRouterChat) -> None:
    """Without the shared index a turn's reasoning accumulates as one block per delta."""
    pieces = [
        model._convert_chunk_to_generation_chunk(
            {"choices": [{"delta": {"role": "assistant", "reasoning": piece}}]}, Chunk, None
        )
        for piece in ("먼저 ", "파일을 ", "읽자")
    ]

    merged = pieces[0].message + pieces[1].message + pieces[2].message  # type: ignore[union-attr]

    assert merged.content == [{"type": "reasoning", "reasoning": "먼저 파일을 읽자", "index": 0}]


def test_the_reasoning_goes_back_out_on_its_own_turn(model: OpenRouterChat) -> None:
    """The signature only counts if the turn that earned it carries it back."""
    turn = AIMessage(content="읽을게요", additional_kwargs={"reasoning_details": DETAILS})

    payload = model._get_request_payload([HumanMessage("안녕"), turn], stop=None)

    sent = [m.get("reasoning_details") for m in payload["messages"]]
    assert sent == [None, DETAILS], "on the turn that earned it, not another"
