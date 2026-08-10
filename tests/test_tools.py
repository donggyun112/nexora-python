"""The pure helpers, tested without driving a turn.

Three of the reference's trickiest rules live here — exclusive selection, terminating tools,
and suspension snapshots. They have no I/O, so a unit test is enough and a failure points at
the rule rather than at the loop.
"""

from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from nexora.contracts import BaseMessage, ToolCall
from nexora.history import suspend_history_snapshot
from nexora.tools import (
    InvalidToolCall,
    as_model_tools,
    render_for_model,
    require_call_ids,
    select_for_execution,
    terminates_loop,
)


class Defs:
    """A tool table that only answers definition lookups."""

    def __init__(self, **defs: dict[str, Any]) -> None:
        self.defs = defs

    async def execute(self, name: str, call_id: str, arguments: Any) -> dict[str, Any]:
        raise AssertionError("these tests never execute")

    def get(self, name: str) -> dict[str, Any] | None:
        return self.defs.get(name)

    def list(self) -> list[dict[str, Any]]:
        return []


def a_call(name: str, cid: str = "c1", **arguments: Any) -> ToolCall:
    return {"id": cid, "name": name, "args": arguments, "type": "tool_call"}


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


# ── suspend_history_snapshot ─────────────────────────────────────────────────


def _assistant(*call_ids: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[a_call("t", cid) for cid in call_ids])


def test_the_snapshot_keeps_the_suspended_call_and_the_completed_ones() -> None:
    messages: list[BaseMessage] = [HumanMessage("go"), _assistant("c1", "c2", "c3")]

    snapshot = suspend_history_snapshot(messages, "c2", ["c1"])

    kept = snapshot[1]
    assert isinstance(kept, AIMessage)
    assert [c["id"] for c in kept.tool_calls] == ["c1", "c2"]


def test_the_snapshot_does_not_mutate_the_live_history() -> None:
    original = _assistant("c1", "c2")
    messages: list[BaseMessage] = [original]

    suspend_history_snapshot(messages, "c1", [])

    assert [c["id"] for c in original.tool_calls] == ["c1", "c2"]


def test_earlier_turns_are_left_alone() -> None:
    """Only the message that issued the suspended call is pruned."""
    messages: list[BaseMessage] = [_assistant("old1", "old2"), _assistant("c1", "c2")]

    snapshot = suspend_history_snapshot(messages, "c1", [])

    earlier = snapshot[0]
    assert isinstance(earlier, AIMessage)
    assert [c["id"] for c in earlier.tool_calls] == ["old1", "old2"]


# ── The round has to be keyable before any of it runs ────────────────────────


def test_a_round_with_an_unkeyable_call_is_refused_whole() -> None:
    """The call id is the step name, so a round that cannot be keyed cannot be recorded.

    What this replaced: the first tool executed and the second raised `duplicate step name` from
    inside the ledger — after the effect. Refused whole rather than filtered, because a dropped
    call would stay in the assistant message with no `ToolMessage` answering it, which
    `_unanswered_tool_calls` reads as still pending, forever.
    """
    with pytest.raises(InvalidToolCall, match="has no id"):
        require_call_ids([a_call("read", cast(str, None))])
    with pytest.raises(InvalidToolCall, match="has no id"):
        require_call_ids([a_call("read", "")])
    with pytest.raises(InvalidToolCall, match="appears twice"):
        require_call_ids([a_call("read", "c1"), a_call("write", "c1")])

    keyed = [a_call("read", "c1"), a_call("write", "c2")]
    assert require_call_ids(keyed) == keyed


def test_selection_refuses_the_round_before_announcing_a_single_call() -> None:
    """Invalid call identifiers reject the round before any call is announced."""
    with pytest.raises(InvalidToolCall):
        select_for_execution(Defs(), [a_call("read", "c1"), a_call("read", "c1")])


# ── What reaches the model is ordered ────────────────────────────────────────


def _definition(name: str) -> dict[str, Any]:
    return {"name": name, "description": f"{name} something", "parameters": {"type": "object"}}


def test_the_same_tool_set_binds_in_the_same_order_however_it_was_listed() -> None:
    """`registry.ts`'s `assemble` — tool definitions bind in deterministic name order."""
    listed = [_definition(n) for n in ("write", "read", "Bash", "grep")]

    names = [tool["function"]["name"] for tool in as_model_tools(listed)]

    assert names == ["Bash", "grep", "read", "write"]
    assert names == [
        tool["function"]["name"] for tool in as_model_tools(list(reversed(listed)))
    ]
