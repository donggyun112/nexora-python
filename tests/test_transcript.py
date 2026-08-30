"""Transcript semantics: round-trip, idempotency, and the chain. Fakes, not mocks.

The property that matters most here is the first one — a conversation stored and read back must be
the conversation that ran. Everything else about the storage design (a document per row, columns
generated from it, `entry` kept verbatim) exists to make that true, and this file is where it is
actually checked.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from semora.transcript import SCHEMA_VERSION, TranscriptWriter, message_entry, messages_of
from semora_store import MemoryTranscript, Transcript

pytestmark = pytest.mark.anyio


def a_conversation() -> list[Any]:
    """One of every message shape the loop puts into context, including a tool round."""
    return [
        SystemMessage("you are terse"),
        HumanMessage("read the file"),
        AIMessage(
            content="reading",
            tool_calls=[{"id": "c1", "name": "read", "args": {"path": "a.txt"}}],
        ),
        ToolMessage(content="contents", tool_call_id="c1", name="read"),
        AIMessage("done"),
    ]


async def write(store: Transcript, messages: list[Any], **kwargs: Any) -> TranscriptWriter:
    writer = TranscriptWriter(
        store,
        conversation_id="conv-1",
        run_id="run-1",
        **kwargs,
    )
    await writer.opened()
    for message in messages:
        await writer.record(message)
    return writer


# ── Round-trip ───────────────────────────────────────────────────────────────


async def test_a_stored_conversation_reads_back_as_the_conversation_that_ran() -> None:
    """A stored conversation decodes to the messages that originally ran."""
    store = MemoryTranscript()
    original = a_conversation()

    await write(store, original)

    restored = messages_of(await store.read("conv-1"))
    assert [(type(m), m.content) for m in restored] == [(type(m), m.content) for m in original]


async def test_the_tool_call_survives_the_round_trip() -> None:
    """Tool-call identifiers and names survive transcript round trips."""
    store = MemoryTranscript()

    await write(store, a_conversation())

    restored = messages_of(await store.read("conv-1"))
    asking = next(m for m in restored if isinstance(m, AIMessage) and m.tool_calls)
    assert [(call["id"], call["name"]) for call in asking.tool_calls] == [("c1", "read")]


async def test_unknown_entry_kinds_are_skipped_rather_than_raising() -> None:
    """The contract's first design principle: a kind a reader does not know must not break it."""
    store = MemoryTranscript()
    await write(store, [HumanMessage("hi")])
    await store.append(
        {"uuid": "manifest-1", "conversation_id": "conv-1", "type": "attachment", "ref": "x.png"}
    )

    assert [m.content for m in messages_of(await store.read("conv-1"))] == ["hi"]


# ── Idempotency ──────────────────────────────────────────────────────────────


async def test_replaying_a_conversation_appends_nothing_new() -> None:
    """Replaying an identical conversation appends no duplicate entries."""
    store = MemoryTranscript()
    await write(store, a_conversation())

    before = await store.read("conv-1")
    await write(store, a_conversation())

    assert await store.read("conv-1") == before


async def test_append_reports_a_duplicate_rather_than_hiding_it() -> None:
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])
    writer._parent_uuid = None  # rewind to the same chain position

    assert await writer.record(HumanMessage("hi")) is False


async def test_a_diverging_turn_gets_its_own_id() -> None:
    """Same position, different words — two entries, not a collision."""
    first = message_entry(AIMessage("left"), conversation_id="conv-1", parent_uuid="root")
    second = message_entry(AIMessage("right"), conversation_id="conv-1", parent_uuid="root")

    assert first["uuid"] != second["uuid"]


async def test_the_same_words_at_a_different_position_get_their_own_id() -> None:
    """Identical content at different chain positions receives distinct identifiers."""
    early = message_entry(AIMessage("ok"), conversation_id="conv-1", parent_uuid="a")
    late = message_entry(AIMessage("ok"), conversation_id="conv-1", parent_uuid="b")

    assert early["uuid"] != late["uuid"]


async def test_a_token_count_does_not_change_the_id() -> None:
    """Provider usage metadata does not change a transcript entry identifier."""
    plain = AIMessage("ok")
    measured = AIMessage(
        "ok", usage_metadata={"input_tokens": 9, "output_tokens": 1, "total_tokens": 10}
    )

    assert (
        message_entry(plain, conversation_id="c", parent_uuid="p")["uuid"]
        == message_entry(measured, conversation_id="c", parent_uuid="p")["uuid"]
    )


# ── Envelope and chain ───────────────────────────────────────────────────────


async def test_entries_chain_through_parent_uuid() -> None:
    """The order the DB replays by is `seq`; the order a reader can *verify* is this chain."""
    store = MemoryTranscript()

    await write(store, a_conversation())

    entries = await store.read("conv-1")
    assert entries[0]["parent_uuid"] is None
    assert [e["parent_uuid"] for e in entries[1:]] == [e["uuid"] for e in entries[:-1]]


async def test_the_run_id_rides_in_metadata_where_the_store_derives_its_column_from() -> None:
    """So a `.jsonl` handed to someone without the database still says which run wrote it."""
    store = MemoryTranscript()

    await write(store, [HumanMessage("hi")])

    assert (await store.read("conv-1"))[0]["metadata"]["run_id"] == "run-1"


async def test_the_run_context_is_stamped_on_every_entry() -> None:
    """Where it ran and on what build.

    A transcript without these cannot be re-run or debugged, and they cost no schema change because
    `entry` is stored verbatim.
    """
    store = MemoryTranscript()
    context = {"cwd": "/repo", "git_branch": "main", "version": "0.1.0"}

    await write(store, a_conversation(), context=context)

    assert all(e["metadata"].items() >= context.items() for e in await store.read("conv-1"))


async def test_type_mirrors_the_role_so_the_filter_axis_needs_no_body_lookup() -> None:
    store = MemoryTranscript()

    await write(store, a_conversation())

    types = [e["type"] for e in await store.read("conv-1")]
    assert types == ["system", "human", "ai", "tool", "ai"]


async def test_the_schema_version_says_this_body_is_not_the_typescript_one() -> None:
    """Envelope shared, body LangChain. A reader told `v2` would expect Anthropic content blocks."""
    entry = message_entry(HumanMessage("hi"), conversation_id="conv-1")

    assert entry["schema_version"] == SCHEMA_VERSION != "v2"


async def test_the_newest_tail_comes_back_in_replay_order() -> None:
    store = MemoryTranscript()
    await write(store, a_conversation())

    tail = await store.read("conv-1", limit=2)

    assert [e["type"] for e in tail] == ["tool", "ai"]


# ── Rewind and delete ────────────────────────────────────────────────────────


async def test_a_rewind_shortens_the_branch_without_deleting_anything() -> None:
    """The property a rewrite-based rewind cannot offer: the later entries are still on disk."""
    store = MemoryTranscript()
    writer = await write(store, a_conversation())
    entries = await store.read("conv-1")

    await writer.rewind(entries[1]["uuid"])

    assert [m.content for m in messages_of(await store.read("conv-1"))] == [
        "you are terse",
        "read the file",
    ]
    assert len(await store.read("conv-1")) == 6  # five messages plus the leaf marker


async def test_a_later_leaf_returns_to_the_branch_an_earlier_one_left() -> None:
    store = MemoryTranscript()
    writer = await write(store, a_conversation())
    entries = await store.read("conv-1")
    await writer.rewind(entries[1]["uuid"])

    await writer.rewind(entries[-1]["uuid"])

    assert len(messages_of(await store.read("conv-1"))) == 5


async def test_returning_to_a_tip_the_branch_already_held_moves_the_branch() -> None:
    """A leaf states where the branch ends now, so the same one made twice is not one statement.

    Content-addressed leaves made this third rewind a duplicate the store absorbed, leaving the
    branch at the second target.
    """
    store = MemoryTranscript()
    writer = await write(store, a_conversation())
    entries = await store.read("conv-1")
    await writer.rewind(entries[1]["uuid"])
    await writer.rewind(entries[-1]["uuid"])

    await writer.rewind(entries[1]["uuid"])

    assert [m.content for m in messages_of(await store.read("conv-1"))] == [
        "you are terse",
        "read the file",
    ]


async def test_an_entry_recorded_after_returning_to_an_earlier_tip_is_on_the_branch() -> None:
    """The harm an absorbed leaf does: the writer chains onto a tip no reader walks to.

    The entry reaches the store either way, so only reading the branch back catches it.
    """
    store = MemoryTranscript()
    writer = await write(store, a_conversation())
    entries = await store.read("conv-1")
    await writer.rewind(entries[1]["uuid"])
    await writer.rewind(entries[-1]["uuid"])
    await writer.rewind(entries[1]["uuid"])

    await writer.record(HumanMessage("after the rewind"))

    branch = [m.content for m in messages_of(await store.read("conv-1"))]
    assert branch == ["you are terse", "read the file", "after the rewind"]


async def test_rewinding_to_nothing_empties_the_branch() -> None:
    store = MemoryTranscript()
    writer = await write(store, a_conversation())

    await writer.rewind(None)

    assert messages_of(await store.read("conv-1")) == []


async def test_a_forgotten_entry_leaves_the_branch() -> None:
    store = MemoryTranscript()
    writer = await write(store, a_conversation())
    entries = await store.read("conv-1")

    await writer.forget(entries[1]["uuid"])

    contents = [m.content for m in messages_of(await store.read("conv-1"))]
    assert "read the file" not in contents
    assert len(contents) == 4


async def test_forgetting_a_middle_entry_does_not_re_parent_the_conversation() -> None:
    """The hole stays a hole.

    Splicing its children onto its parent would silently rewrite what the model was told, which is
    worse than a gap.
    """
    store = MemoryTranscript()
    writer = await write(store, a_conversation())
    entries = await store.read("conv-1")

    await writer.forget(entries[2]["uuid"])

    survivors = [e["uuid"] for e in await store.read("conv-1") if "parent_uuid" in e]
    assert entries[3]["parent_uuid"] == entries[2]["uuid"]
    assert entries[2]["uuid"] in survivors


async def test_markers_are_not_links_in_the_chain() -> None:
    """A marker is a statement about the chain, not a link in it.

    Chaining through one would move the conversation every time a leaf pointer was appended.
    """
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])

    await writer.forget("nobody")

    marker = next(e for e in await store.read("conv-1") if e["type"] == "tombstone")
    assert "parent_uuid" not in marker


# ── Run cost ─────────────────────────────────────────────────────────────────


async def test_a_run_that_never_ends_is_a_row_with_no_ending() -> None:
    """`opened` exists for this: an operator has to tell a crash from a run that never began."""
    store = MemoryTranscript()

    await write(store, [HumanMessage("hi")])

    row = await store.read_run("run-1")
    assert row is not None
    assert set(row) == {"conversation_id", "started_at"}
    assert row["conversation_id"] == "conv-1"


async def test_closing_records_the_cost_against_the_model_that_spent_it() -> None:
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])

    await writer.closed(
        {
            "type": "done",
            "content": "fin",
            "tool_calls": [{"name": "read", "input": {}}],
            "stop_reason": "completed",
            "model": "claude-opus-4-7",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 5,
                "total_tokens": 1005,
                "cached_tokens": 800,
                "cache_write_tokens": 100,
            },
        },
        cost_usd={"claude-opus-4-7": 0.0125},
    )

    run = await store.read_run("run-1")
    assert run is not None
    assert run["tool_calls"] == 1
    assert run["ended_at"] is not None
    assert await store.read_model_usage("run-1") == {
        "claude-opus-4-7": {
            "prompt_tokens": 1000,
            "completion_tokens": 5,
            "total_tokens": 1005,
            "cached_tokens": 800,
            "cache_write_tokens": 100,
            "cost_usd": 0.0125,
        }
    }


async def test_a_fallback_run_is_priced_per_model_not_blended() -> None:
    """Two models at one rate prices neither. The loop reports the breakdown; this keeps it."""
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])

    await writer.closed(
        {
            "type": "done",
            "content": "fin",
            "stop_reason": "completed",
            "model": "claude-opus-4-8",
            "usage": {"prompt_tokens": 300, "completion_tokens": 30},
            "usage_by_model": {
                "claude-opus-5": {"prompt_tokens": 100, "completion_tokens": 10},
                "claude-opus-4-8": {"prompt_tokens": 200, "completion_tokens": 20},
            },
        }
    )

    per_model = await store.read_model_usage("run-1")
    assert sorted(per_model) == ["claude-opus-4-8", "claude-opus-5"]
    assert per_model["claude-opus-5"]["prompt_tokens"] == 100


async def test_tokens_reported_before_any_model_was_named_stay_unattributed() -> None:
    """Charging them to whichever model came later would be a guess."""
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])

    await writer.closed(
        {"type": "done", "content": "", "stop_reason": "completed", "usage": {"prompt_tokens": 7}}
    )

    assert await store.read_model_usage("run-1") == {"": {"prompt_tokens": 7}}


async def test_an_unreported_usage_writes_no_model_row_at_all() -> None:
    """Absent and zero are different facts — a `0` here would price an unmeasured run as free."""
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])

    await writer.closed({"type": "done", "content": "", "stop_reason": "completed"})

    assert await store.read_model_usage("run-1") == {}


async def test_closing_merges_rather_than_replacing_what_opening_wrote() -> None:
    """A replace would erase `started_at`, which is the field that says the run began."""
    store = MemoryTranscript()
    writer = await write(store, [HumanMessage("hi")])
    await store.record_run("run-1", {"started_at": "then"})

    await writer.closed({"type": "done", "content": "", "stop_reason": "aborted"})

    row = await store.read_run("run-1")
    assert row is not None
    assert row["started_at"] == "then"
    assert row["stop_reason"] == "aborted"
