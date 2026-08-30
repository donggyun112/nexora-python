"""A gateway leaves DeepSeek's tool markup in assistant content; the client puts it back.

Repair belongs to the provider client, not the planner. The loop speaks LangChain tool
calls, and teaching it one vendor's markup would make every other provider pay for a
dialect it never emits.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk
from semora_llm import ChatModel, openrouter
from semora_llm.dsml import (
    DsmlFilter,
    parse_dsml_tool_calls,
    recover_dsml_chunks,
    strip_dsml,
)

BAR = "\uff5c"  # FULLWIDTH VERTICAL LINE — what DeepSeek actually writes

_BLOCK = (
    f"<{BAR}DSML{BAR}tool_calls>\n"
    f'<{BAR}DSML{BAR}invoke name="charge_card">\n'
    f'<{BAR}DSML{BAR}parameter name="customer_id" string="true">c-001'
    f"</{BAR}DSML{BAR}parameter>\n"
    f'<{BAR}DSML{BAR}parameter name="amount" string="true">10</{BAR}DSML{BAR}parameter>\n'
    f"</{BAR}DSML{BAR}invoke>\n"
    f"</{BAR}DSML{BAR}tool_calls>"
)


async def _recovered(pieces: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Run the pieces through the repair; report visible text and recovered calls."""

    async def source() -> Any:
        for piece in pieces:
            yield piece if isinstance(piece, AIMessageChunk) else AIMessageChunk(content=piece)

    text = ""
    calls: list[dict[str, Any]] = []
    async for chunk in recover_dsml_chunks(source()):
        if isinstance(chunk.content, str):
            text += chunk.content
        calls.extend(dict(call) for call in chunk.tool_calls or [])
    return text, calls


def test_parse_dsml_tool_calls_reads_invoke_parameters() -> None:
    calls = parse_dsml_tool_calls(_BLOCK)
    assert len(calls) == 1
    assert calls[0]["name"] == "charge_card"
    assert calls[0]["args"] == {"customer_id": "c-001", "amount": "10"}


def test_dsml_filter_holds_a_truncated_open_tag() -> None:
    filt = DsmlFilter()
    assert filt.push(f"<{BAR}DSML{BAR}tool_c") == ""
    assert filt.finish() == ""
    assert filt.swallowed is True


def test_dsml_filter_emits_prose_and_swallows_the_block() -> None:
    filt = DsmlFilter()
    assert filt.push("ok\n") == "ok\n"
    assert filt.push(_BLOCK) == ""
    assert filt.finish() == ""
    assert parse_dsml_tool_calls(filt.markup)[0]["name"] == "charge_card"


def test_strip_dsml_keeps_preamble() -> None:
    assert strip_dsml("미리말\n" + _BLOCK) == "미리말"
    assert strip_dsml("all done") == "all done"


async def test_a_leaked_block_becomes_tool_calls_and_leaves_no_text() -> None:
    text, calls = await _recovered(["확인했습니다\n", _BLOCK])
    assert text == "확인했습니다\n"
    assert [call["name"] for call in calls] == ["charge_card"]


@pytest.mark.parametrize(
    "pieces",
    [
        ["금액은 ", "<", "10 달러입니다."],
        ["확인했습니다 ", f"<{BAR}", " 끝"],
        ["<", "|", "그냥 텍스트"],
        ["평범한 ", "응답입니다."],
    ],
)
async def test_a_held_open_tag_prefix_is_emitted_exactly_once(pieces: list[str]) -> None:
    """One held prefix, emitted once.

    A chunk boundary isolating an open-tag prefix used to replay the held chunk after the
    whole reply, appending a stray fragment to every answer.
    """
    text, calls = await _recovered(pieces)
    assert text == "".join(pieces)
    assert calls == []


async def test_text_held_before_a_native_tool_call_is_not_dropped() -> None:
    native = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "charge_card", "args": "{}", "id": "c1", "index": 0,
             "type": "tool_call_chunk"},
        ],
    )
    text, _ = await _recovered(["확인했습니다 ", "<", native])
    assert text == "확인했습니다 <"


def test_the_openrouter_preset_repairs_and_keeps_it_through_bind_tools() -> None:
    plain = ChatModel("deepseek/deepseek-v4-flash-latest", api_key="k")
    assert plain.recover_dsml is False, "off unless a preset asks for it"

    routed = openrouter("deepseek/deepseek-v4-flash-latest", api_key="k")
    assert routed.recover_dsml is True
    assert routed.bind_tools([]).recover_dsml is True


def test_a_request_deadline_survives_bind_tools() -> None:
    """bind_tools copies a frozen dataclass; a field it forgets is silently lost."""
    model = openrouter("m", api_key="k", timeout=60)
    assert model.timeout == 60
    assert model.bind_tools([]).timeout == 60
