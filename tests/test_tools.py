"""The pure helpers, tested without driving a turn.

Three of the reference's trickiest rules live here — exclusive selection, terminating tools,
and suspension snapshots. They have no I/O, so a unit test is enough and a failure points at
the rule rather than at the loop.
"""

from typing import Any

from nexora.history import suspend_history_snapshot
from nexora.model_turn import ModelTurn, parse_arguments
from nexora.tools import render_for_model, select_for_execution, terminates_loop
from nexora.types import LLMMessage, ToolCall


class Defs:
    """A tool table that only answers definition lookups."""

    def __init__(self, **defs: dict[str, Any]) -> None:
        self.defs = defs

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        raise AssertionError("these tests never execute")

    def get(self, name: str) -> dict[str, Any] | None:
        return self.defs.get(name)


def a_call(name: str, cid: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(cid, name, arguments)


# ── select_for_execution ─────────────────────────────────────────────────────


def test_without_an_exclusive_call_the_batch_is_untouched() -> None:
    calls = [a_call("read", "c1"), a_call("grep", "c2")]
    assert select_for_execution(Defs(), calls) == calls


def test_an_exclusive_call_displaces_the_whole_batch() -> None:
    calls = [a_call("read", "c1"), a_call("deploy", "c2"), a_call("grep", "c3")]

    assert select_for_execution(Defs(deploy={"is_exclusive": True}), calls) == [calls[1]]


def test_exclusivity_can_depend_on_the_arguments() -> None:
    """`is_exclusive` may be a predicate — `rm -rf /` is exclusive, `rm tmp` is not."""
    defs = Defs(rm={"is_exclusive": lambda args: args.get("recursive", False)})

    safe = [a_call("read", "c1"), a_call("rm", "c2", recursive=False)]
    assert select_for_execution(defs, safe) == safe

    risky = [a_call("read", "c1"), a_call("rm", "c2", recursive=True)]
    assert select_for_execution(defs, risky) == [risky[1]]


# ── terminates_loop ──────────────────────────────────────────────────────────


def test_a_tool_without_a_definition_never_terminates() -> None:
    assert terminates_loop(Defs(), a_call("read")) is False


def test_terminates_loop_reads_the_flag() -> None:
    assert terminates_loop(Defs(submit={"terminates_loop": True}), a_call("submit")) is True


# ── render_for_model ─────────────────────────────────────────────────────────


def test_text_results_pass_through() -> None:
    assert render_for_model({"type": "text", "text": "hello"}) == "hello"


def test_errors_are_marked_so_the_model_can_tell() -> None:
    assert render_for_model({"type": "error", "message": "nope"}) == "[ERROR] nope"


def test_images_are_summarized_never_inlined() -> None:
    """Serializing an image here would put base64 in the transcript."""
    rendered = render_for_model({"type": "image", "data": "AAAA" * 1000, "mime_type": "image/png"})

    assert rendered == "[image]"
    assert "AAAA" not in rendered


def test_a_multimodal_result_keeps_text_and_marks_images_in_order() -> None:
    result = {
        "type": "content",
        "blocks": [
            {"type": "text", "text": "page 1"},
            {"type": "image", "data": "xxx", "mime_type": "image/png"},
            {"type": "text", "text": "page 2"},
        ],
    }

    assert render_for_model(result) == "page 1\n[image]\npage 2"


# ── parse_arguments ──────────────────────────────────────────────────────────


def test_arguments_parse_from_concatenated_fragments() -> None:
    assert parse_arguments('{"path": "a.py"}') == {"path": "a.py"}


def test_empty_and_malformed_arguments_both_degrade_to_no_args() -> None:
    assert parse_arguments("") == {}
    assert parse_arguments("{unclosed") == {}


# ── ModelTurn ────────────────────────────────────────────────────────────────


def test_a_snapshot_provider_that_skips_deltas_still_yields_its_text() -> None:
    turn = ModelTurn()

    done = {"type": "done", "content": "all at once", "stop_reason": "end_turn"}

    assert turn.absorb(done) is None
    assert turn.text == "all at once"


def test_everything_the_provider_reported_is_kept() -> None:
    """A provider says these once. `thinking` especially cannot be recovered later."""
    turn = ModelTurn()
    turn.absorb({"type": "thinking_delta", "delta": "let me "})
    turn.absorb({"type": "thinking_delta", "delta": "check"})
    turn.absorb(
        {
            "type": "done",
            "content": "here",
            "stop_reason": "max_tokens",
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
    )

    assert turn.thinking == "let me check"
    assert turn.stop_reason == "max_tokens"
    assert turn.usage == {"prompt_tokens": 12, "completion_tokens": 3}


def test_tool_calls_keep_the_order_the_model_issued_them() -> None:
    turn = ModelTurn()
    for chunk in [
        {"type": "tool_call_start", "id": "c2", "name": "grep"},
        {"type": "tool_call_start", "id": "c1", "name": "read"},
        {"type": "tool_call_delta", "id": "c1", "delta": '{"p":'},
        {"type": "tool_call_delta", "id": "c1", "delta": '"a"}'},
    ]:
        turn.absorb(chunk)

    calls = turn.tool_calls()
    assert [c.id for c in calls] == ["c2", "c1"]
    assert calls[1].arguments == {"p": "a"}


# ── suspend_history_snapshot ─────────────────────────────────────────────────


def _assistant(*call_ids: str) -> LLMMessage:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_call", "id": cid, "name": "t", "arguments": {}} for cid in call_ids
        ],
    }


def test_the_snapshot_keeps_the_suspended_call_and_the_completed_ones() -> None:
    messages: list[LLMMessage] = [
        {"role": "user", "content": "go"},
        _assistant("c1", "c2", "c3"),
    ]

    snapshot = suspend_history_snapshot(messages, "c2", ["c1"])

    blocks = snapshot[1]["content"]
    assert isinstance(blocks, list)
    assert [b["id"] for b in blocks] == ["c1", "c2"]


def test_the_snapshot_does_not_mutate_the_live_history() -> None:
    messages: list[LLMMessage] = [_assistant("c1", "c2")]

    suspend_history_snapshot(messages, "c1", [])

    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    assert [b["id"] for b in blocks] == ["c1", "c2"]


def test_earlier_turns_are_left_alone() -> None:
    """Only the message that issued the suspended call is pruned."""
    messages: list[LLMMessage] = [_assistant("old1", "old2"), _assistant("c1", "c2")]

    snapshot = suspend_history_snapshot(messages, "c1", [])

    earlier = snapshot[0]["content"]
    assert isinstance(earlier, list)
    assert [b["id"] for b in earlier] == ["old1", "old2"]
