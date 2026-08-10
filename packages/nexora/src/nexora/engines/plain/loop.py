"""The ReAct loop, ported from Nexora's `packages/architectures/src/react.ts`.

Control flow and nothing else: the model, the tools, and every policy hook are injected. The
loop runs until something tells it to stop — there is no built-in iteration cap, because how
long an agent may run is the caller's decision (see `ShouldStopAfterTurn`).

`controls` is one object because each control point is one decision: an ordered chain of stages,
composed by whoever supervises this run (`nexora.controls.ControlPlane`). They are calls and not
subscriptions, because the order of those stages is the policy and no event dispatch can promise
an order. `emit` is the other side of that: observation, published after each decision, and
dropped rather than raised if a sink is unwell.

The remaining hooks each take a position the loop alone has: `drain_inputs` and
`should_stop_after_turn` need a round boundary. Suspension is deliberately absent: the injected
orchestrator commits it and terminates this execution instead of returning it as agent state.

`execute_round` defaults to the plain `execute_calls`, so durability is injected rather than
assumed. Driven directly, this loop touches no ledger and no store while keeping every control
point, and the one thing it then cannot do is park a call — a suspension is only a suspension once
its continuation is written down. See `examples/06_bare_loop.py`.

The model is a LangChain `BaseChatModel`, so provider differences, tool binding, and the
reassembly of tool arguments that arrive as JSON fragments all happen below this file.
"""

from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    messages_to_dict,
)

from ...contracts.events import EventType
from ...contracts.types import (
    Aborted,
    AdmitInputs,
    BaseMessage,
    CompactContext,
    DrainInputs,
    Emit,
    ModelErrorKind,
    ModelFailure,
    OnModelFailure,
    PendingInput,
    ShouldStopAfterTurn,
    StopReason,
    Tools,
)
from ...controls import Controls, Ctx, Halt, Proceed
from ...tools import (
    ExecuteRound,
    absorb_round,
    as_model_tools,
    execute_calls,
    select_for_execution,
    tool_result,
)


@dataclass(slots=True)
class _Spend:
    """What a run cost, accumulated across its turns.

    One object rather than a counter and a string because every terminal path reports both, and
    the alternative was another parameter threaded through ten `_done` calls. `Counter` and not a
    plain dict for the tokens: `.update()` adds instead of overwriting, and a key nobody reported
    stays absent rather than becoming a zero somebody has to interpret.

    Kept per model, because one run is not always answered by one model — the same reason
    `_model_of` reads the name off the reply instead of the bound model. A single total plus the
    last name seen prices the whole run at that model's rate, which is wrong by whatever the rates
    differ by. Tokens reported before any model was named key on `""`: unattributed is a fact, and
    charging them to whichever model came later would be a guess.
    """

    by_model: dict[str, Counter[str]] = field(default_factory=dict)
    model: str = ""


async def react_loop(
    model: Any,
    tools: Tools,
    *,
    system_prompt: str | None = None,
    history: list[BaseMessage] | None = None,
    aborted: Aborted = lambda: False,
    controls: Controls | None = None,
    emit: Emit | None = None,
    drain_inputs: DrainInputs | None = None,
    admit_inputs: AdmitInputs | None = None,
    should_stop_after_turn: ShouldStopAfterTurn | None = None,
    on_model_failure: OnModelFailure | None = None,
    compact_context: CompactContext | None = None,
    execute_round: ExecuteRound = execute_calls,
) -> AsyncIterator[dict[str, Any]]:
    """Reason, act, repeat. Yields events as they happen.

    Every incremental message arrives through `drain_inputs`. `history` is only the already
    committed transcript baseline, so initial prompts, steers and asynchronous results all cross
    the same admission point and produce the same audit fact.
    """
    messages: list[BaseMessage] = [
        *([SystemMessage(system_prompt)] if system_prompt else []),
        *(history or []),
    ]
    available = tools.list()
    bound = model.bind_tools(as_model_tools(available)) if available else model
    calls_made: list[dict[str, Any]] = []
    spent = _Spend()
    last_text = ""
    turn = -1
    carried_inputs: list[PendingInput] = []

    while True:
        turn += 1
        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return
        pending_inputs = [
            *carried_inputs,
            *(list(await drain_inputs()) if drain_inputs else []),
        ]
        carried_inputs = []
        if controls is not None and pending_inputs:
            # Screened before `before_model` reads them: a gate deciding on the pending messages
            # must see what will actually enter context, never a pre-mask original.
            match await controls.on_inputs(
                Ctx(
                    turn=turn,
                    messages=list(messages),
                    calls_made=list(calls_made),
                    text=last_text,
                ),
                pending_inputs,
            ):
                case Halt(halt_reason):
                    yield await _done(emit, last_text, calls_made, halt_reason, spent)
                    return
                case screened:
                    pending_inputs = screened
        if controls is not None:
            action = await controls.before_model(
                Ctx(
                    turn=turn,
                    messages=[*messages, *(item.message for item in pending_inputs)],
                    calls_made=list(calls_made),
                    text=last_text,
                )
            )
            match action:
                case Halt(halt_reason):
                    yield await _done(emit, last_text, calls_made, halt_reason, spent)
                    return
                case Proceed(more):
                    pending_inputs += [PendingInput("control", message) for message in more]
        await _commit_inputs(messages, pending_inputs, admit_inputs, emit, turn)

        # ── Reason ───────────────────────────────────────────────────────────
        reply: AIMessageChunk | None = None
        failed: _FailedModel | None = None
        async with aclosing(
            _model_stream(
                bound,
                messages,
                aborted,
                on_model_failure,
                compact_context,
                emit,
            )
        ) as model_stream:
            async for item in model_stream:
                if isinstance(item, _FailedModel):
                    failed = item
                    break
                chunk = item
                # Chunks add, and the sum reassembles tool arguments that arrive as fragments.
                reply = chunk if reply is None else reply + chunk
                if delta := chunk.text:
                    yield {"type": "text", "text": delta}
                if aborted():
                    yield await _done(
                        emit, _text_of(reply), calls_made, "aborted", spent, mid_turn=True
                    )
                    return
        if failed is not None:
            if aborted():
                yield await _done(
                    emit, failed.partial, calls_made, "aborted", spent, mid_turn=True
                )
            else:
                cut: dict[str, Any] = {
                    "reason": "error",
                    "message": str(failed.error),
                    "error_type": type(failed.error).__name__,
                    "error_kind": failed.kind,
                }
                if failed.partial:
                    cut["partial"] = failed.partial
                if emit is not None:
                    await emit(EventType.STOP_FAILURE, cut)
                yield {"type": "error", **{k: v for k, v in cut.items() if k != "reason"}}
            return

        turn_text = _text_of(reply)
        last_text = turn_text
        # Kept, not overwritten with a blank: a provider that names the model on some turns and not
        # others should not erase what it already told us.
        spent.model = _model_of(reply) or spent.model
        # Resolved first, then charged, so a turn the provider did not label lands on the model
        # already known rather than on a second, nameless bucket.
        if counts := _usage_of(reply):
            spent.by_model.setdefault(spent.model, Counter()).update(counts)

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        requested_tool_calls = select_for_execution(tools, list(reply.tool_calls) if reply else [])

        if not requested_tool_calls:
            messages.append(AIMessage(turn_text))
            # The caller's turn cap covers tool-free rounds too. Without this check an
            # always-vetoing `before_finish` gate can bypass the only iteration bound the loop
            # exposes.
            if should_stop_after_turn is not None and await should_stop_after_turn(
                turn, turn_text, calls_made
            ):
                yield await _done(emit, turn_text, calls_made, "policy", spent)
                return
            # A steer that landed while the turn was finishing cancels the stop.
            if drain_inputs and (late_inputs := list(await drain_inputs())):
                # Preserve their arrival order, but commit them only beside the next model call.
                # That next turn's `before_model` may still halt, in which case claiming the model
                # received these inputs would be an audit-log lie.
                carried_inputs = late_inputs
                continue
            if controls is not None:
                # The last word. A verifier that says "not done yet" gets another round, which is
                # why this sits after the late-input check and not instead of it: an arriving steer
                # and a policy objection are different reasons to keep going, and both may apply.
                match await controls.before_finish(
                    Ctx(
                        turn=turn,
                        messages=list(messages),
                        calls_made=list(calls_made),
                        text=turn_text,
                    ),
                    "completed",
                ):
                    case Proceed(steers):
                        carried_inputs += [PendingInput("control", message) for message in steers]
                        continue
                    case Halt(halt_reason):
                        yield await _done(emit, turn_text, calls_made, halt_reason, spent)
                        return
            yield await _done(emit, turn_text, calls_made, "completed", spent)
            return

        # ── Act ──────────────────────────────────────────────────────────────
        messages.append(AIMessage(content=turn_text, tool_calls=list(requested_tool_calls)))
        for call in requested_tool_calls:
            calls_made.append({"name": call["name"], "input": call["args"]})
            yield {
                "type": "tool_call",
                "id": call["id"],
                "name": call["name"],
                "input": call["args"],
            }

        resolved = await execute_round(
            tools,
            requested_tool_calls,
            aborted,
            emit,
            turn,
            controls,
            Ctx(
                turn=turn,
                messages=list(messages),
                calls_made=list(calls_made),
                text=turn_text,
            ),
        )
        for call, result, refused in resolved:
            event = tool_result(call, result)
            event["executed"] = not refused
            yield event

        round_ = absorb_round(tools, resolved)
        carried_inputs += [
            PendingInput("tool_result", answer, str(completed["id"]))
            for answer, completed in zip(round_.answers, round_.completed, strict=True)
        ]

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        # ── Stop? ────────────────────────────────────────────────────────────
        # Also asked above on tool-free rounds. Even a terminating tool reaches it: the hook is
        # where budget and verification accounting lives, and it must see every completed round.
        policy_says_stop = should_stop_after_turn is not None and await should_stop_after_turn(
            turn, turn_text, calls_made
        )
        if round_.ended_by_tool or policy_says_stop:
            reason: StopReason = "tool" if round_.ended_by_tool else "policy"
            yield await _done(emit, last_text or "(stopped after turn)", calls_made, reason, spent)
            return


async def _commit_inputs(
    messages: list[BaseMessage],
    inputs: list[PendingInput],
    admit_inputs: AdmitInputs | None,
    emit: Emit | None,
    turn: int,
) -> None:
    """Append accepted inputs and record the one point where they enter model context."""
    for item in inputs:
        messages.append(item.message)
    if inputs and admit_inputs is not None:
        await admit_inputs(inputs)
    for item in inputs:
        if emit is not None:
            await emit(
                EventType.CONTEXT_INJECTED,
                {
                    "turn": turn,
                    "kind": item.kind,
                    "origin_id": item.origin_id,
                    "message": messages_to_dict([item.message])[0],
                },
            )


@dataclass(frozen=True, slots=True)
class _FailedModel:
    """A terminal provider failure returned by `_model_stream` to the event loop."""

    error: Exception
    kind: ModelErrorKind
    partial: str


async def _model_stream(
    bound: Any,
    messages: list[BaseMessage],
    aborted: Aborted,
    on_failure: OnModelFailure | None,
    compact_context: CompactContext | None,
    emit: Emit | None,
) -> AsyncGenerator[AIMessageChunk | _FailedModel, None]:
    """Stream one model round, applying caller-owned recovery before exposing failure.

    Recovery stays inside the round so a retry cannot replay earlier tool effects. Once any text
    was yielded, recovery fails closed: a second generation would duplicate text the consumer has
    already displayed and could diverge from the unfinished first answer.
    """
    attempts: Counter[ModelErrorKind] = Counter()
    while True:
        reply: AIMessageChunk | None = None
        try:
            # Closing an abandoned stream reaches the provider instead of waiting for garbage
            # collection while it keeps generating billable tokens.
            async with aclosing(bound.astream(messages)) as stream:
                async for chunk in stream:
                    reply = chunk if reply is None else reply + chunk
                    yield chunk
        except Exception as error:
            kind = _model_error_kind(error)
            partial = _text_of(reply)
            attempts[kind] += 1
            failure = ModelFailure(
                type(error).__name__,
                kind,
                str(error),
                partial,
                attempts[kind],
            )
            action = (
                await on_failure(failure)
                if on_failure is not None and not partial and not aborted()
                else "fail"
            )
            if action == "retry":
                continue
            if action == "compact" and compact_context is not None:
                if emit is not None:
                    await emit(EventType.PRE_COMPACT, {"reason": kind})
                messages[:] = await compact_context(list(messages), failure)
                if emit is not None:
                    await emit(EventType.POST_COMPACT, {"reason": kind})
                continue
            yield _FailedModel(error, kind, partial)
        return


def _text_of(reply: AIMessageChunk | None) -> str:
    """The assistant text of a turn, whatever shape the provider streamed it in."""
    return reply.text if reply is not None else ""


def _model_of(reply: AIMessageChunk | None) -> str:
    """The model that answered, when the provider named it.

    Read off the reply rather than the bound model, because those are different facts and the
    billed one is here: an alias resolves to a dated snapshot, and a provider-side fallback can
    answer on a model nobody asked for. A cost record naming what was requested rather than what
    ran is a cost record that reconciles against nothing.
    """
    metadata = getattr(reply, "response_metadata", None) if reply is not None else None
    if not metadata:
        return ""
    return str(metadata.get("model_name") or metadata.get("model") or "")


def _model_error_kind(failure: Exception) -> ModelErrorKind:
    """Classify a provider failure without depending on its human-readable message."""
    markers = {type(failure).__name__}
    for attribute in ("code", "type"):
        if value := getattr(failure, attribute, None):
            markers.add(str(value))
    body = getattr(failure, "body", None)
    if isinstance(body, Mapping):
        markers.update(_error_markers(body))

    compact = {
        "".join(character for character in marker.lower() if character.isalnum())
        for marker in markers
    }
    if any("ratelimit" in marker or marker == "429" for marker in compact):
        return "rate_limit"
    if any(
        token in marker
        for marker in compact
        for token in ("contextlength", "contextwindow", "contextoverflow", "prompttoolong")
    ):
        return "context_overflow"
    if any("maxoutputtoken" in marker or marker == "maxtokenserror" for marker in compact):
        return "max_output_tokens"
    if any(
        token in marker
        for marker in compact
        for token in ("authentication", "unauthorized", "permissiondenied")
    ):
        return "authentication"
    if any(
        token in marker
        for marker in compact
        for token in (
            "internalserver",
            "servererror",
            "apiconnection",
            "apitimeout",
            "overloaded",
        )
    ):
        return "server"
    if any("invalidrequest" in marker or "badrequest" in marker for marker in compact):
        return "invalid_request"

    status = _status_code_of(failure)
    if status == 429:
        return "rate_limit"
    if status is not None and status >= 500:
        return "server"
    if status == 400:
        return "invalid_request"
    return "unknown"


def _error_markers(body: Mapping[Any, Any]) -> set[str]:
    """Collect structured provider error codes from one response body."""
    markers = {str(body[key]) for key in ("code", "type") if body.get(key)}
    nested = body.get("error")
    if isinstance(nested, Mapping):
        markers.update(str(nested[key]) for key in ("code", "type") if nested.get(key))
    return markers


def _status_code_of(failure: Exception) -> int | None:
    """Read an HTTP status exposed directly or through a provider response object."""
    status = getattr(failure, "status_code", None)
    if status is None and (response := getattr(failure, "response", None)) is not None:
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _usage_of(reply: AIMessageChunk | None) -> dict[str, int]:
    """Token counts if the provider reported any. Absent and zero are different facts.

    The cache lines are separated out because the three input kinds are priced differently — a
    cache read costs about a tenth of a fresh input token and a cache write about a quarter more,
    so a caller that multiplies one input number by one rate is wrong by up to an order of
    magnitude on a cached prompt. Names follow `LLMUsage` in the TypeScript contract:
    `cachedTokens` is the read, `cacheWriteTokens` the creation.

    **Whether `prompt_tokens` already contains the cache counts depends on the provider, and this
    function does not decide.** LangChain documents `input_tokens` as the sum of every input type,
    which would include them; Anthropic's own API treats the three as disjoint and expects them
    added (`UsageInfo::total_input` in the Claude Code reference does exactly that). Subtracting
    under the wrong convention double-counts, so `total_tokens` is carried through as the
    discriminator: `prompt + completion == total` means the cache counts are already inside
    `prompt`, and needing the cache counts to reach `total` means they are not. Storing the
    reported total instead of a derived "fresh" figure keeps that decision at the query, where it
    can be fixed, rather than baked into a column that was written wrong.
    """
    usage = getattr(reply, "usage_metadata", None) if reply is not None else None
    if not usage:
        return {}
    details = usage.get("input_token_details") or {}
    counts = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cached_tokens": details.get("cache_read", 0),
        "cache_write_tokens": details.get("cache_creation", 0),
    }
    # A provider that reports no caching at all should not grow two zero-valued keys — the same
    # rule the whole `usage` dict follows one level up.
    return {name: count for name, count in counts.items() if count or name in _ALWAYS_REPORTED}


_ALWAYS_REPORTED = frozenset({"prompt_tokens", "completion_tokens"})


async def _done(
    emit: Emit | None,
    content: str,
    calls_made: list[dict[str, Any]],
    reason: StopReason,
    spent: _Spend,
    *,
    mid_turn: bool = False,
) -> dict[str, Any]:
    """The single terminal event. Every exit but `suspended` goes through here.

    `stop_reason` exists so an abort leaves a record: without it a cancelled run and a finished
    one both look like a stream that simply ended.

    `mid_turn` separates the two places an abort lands, which produce the same shape and mean
    different things. Stopped at a round boundary, `content` is a finished assistant turn. Stopped
    inside a generation, it is a fragment: the words a person already read, of a turn that might
    have been about to call a tool. A host appending that as a completed turn tells the model it
    said something it never finished, so the flag is what says a marker is needed beside it.
    """
    if emit is not None:
        await emit(EventType.STOP, {"reason": reason, "content": content})
    done: dict[str, Any] = {
        "type": "done",
        "content": content,
        "tool_calls": calls_made,
        "stop_reason": reason,
    }
    if mid_turn:
        done["interrupted_mid_turn"] = True
    # Summed with `update` and not `+`: adding Counters drops non-positive counts, and a provider
    # that reported a zero said something different from a provider that reported nothing.
    total: Counter[str] = Counter()
    for counts in spent.by_model.values():
        total.update(counts)
    if total:
        done["usage"] = dict(total)
    # Beside `usage` rather than inside it, matching `done` in react.ts — and load-bearing for the
    # cost side: rates are per-model, so a token count without the model it was spent on cannot be
    # priced. Absent when the provider never named one.
    if spent.model:
        done["model"] = spent.model
    # Only when more than one model answered. With one, `usage` and `model` already say the whole
    # thing and a breakdown repeating them is a second answer to the same question; with two, the
    # flat total is unpriceable on its own. The store keys its token rows by model for exactly
    # this case; what shape they take there is its problem, not this event's.
    if len(spent.by_model) > 1:
        done["usage_by_model"] = {name: dict(counts) for name, counts in spent.by_model.items()}
    return done
