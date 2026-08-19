"""Plain asynchronous ReAct loop compatible with LangChain chat models.

The loop owns deterministic ``model -> tools -> model`` control flow. Policy, persistence,
cancellation, input admission, and iteration limits are supplied through explicit collaborators.
Behavior is ported from ``packages/architectures/src/react.ts``.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, cast

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
    ControlSignal,
    DiscardInputs,
    DrainInputs,
    DynamicTools,
    Emit,
    InvokeModel,
    ModelErrorKind,
    ModelFailure,
    ModelStepError,
    OnModelFailure,
    PendingInput,
    RecordMessages,
    ShouldStopAfterTurn,
    StopReason,
    SystemPromptSource,
    ToolCall,
    Tools,
)
from ...controls import Controls, Ctx, Halt, Proceed
from ...tools import (
    Absorbed,
    ExecuteRound,
    absorb_round,
    as_model_tools,
    execute_calls,
    select_for_execution,
    tool_result,
)


@dataclass(slots=True)
class _Spend:
    """Accumulate token usage per responding model across planner turns."""

    by_model: dict[str, Counter[str]] = field(default_factory=dict)
    model: str = ""


async def react_loop(
    model: Any,
    tools: Tools,
    *,
    system_prompt: str | SystemPromptSource | None = None,
    history: list[BaseMessage] | None = None,
    subject: str = "",
    aborted: Aborted = lambda: False,
    controls: Controls | None = None,
    emit: Emit | None = None,
    drain_inputs: DrainInputs | None = None,
    discard_inputs: DiscardInputs | None = None,
    admit_inputs: AdmitInputs | None = None,
    record_messages: RecordMessages | None = None,
    should_stop_after_turn: ShouldStopAfterTurn | None = None,
    on_model_failure: OnModelFailure | None = None,
    compact_context: CompactContext | None = None,
    invoke_model: InvokeModel | None = None,
    model_identity: str | None = None,
    execute_round: ExecuteRound = execute_calls,
) -> AsyncIterator[dict[str, Any]]:
    """Reason, act, repeat. Yields events as they happen.

    Every incremental message arrives through `drain_inputs`. `history` is only the already
    committed transcript baseline, so initial prompts, steers and asynchronous results all cross
    the same admission point and produce the same audit fact.

    `subject` is who the run acts for, carried into every `Ctx` so a control point can decide with
    it and every tool event is stamped with it. Opaque — see `Ctx.subject`.
    """
    messages: list[BaseMessage] = list(history or [])
    managed_system = False
    calls_made: list[dict[str, Any]] = []
    spent = _Spend()
    last_text = ""
    turn = -1
    carried_inputs: list[PendingInput] = []

    while True:
        turn += 1
        managed_system = await _refresh_system_message(
            messages, system_prompt, managed_system
        )
        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return
        pending_inputs = _identify_inputs([
            *carried_inputs,
            *(list(await drain_inputs()) if drain_inputs else []),
        ])
        carried_inputs = []
        admission = await _admit_turn_inputs(
            pending_inputs,
            messages=messages,
            controls=controls,
            discard_inputs=discard_inputs,
            admit_inputs=admit_inputs,
            record_messages=record_messages,
            emit=emit,
            turn=turn,
            calls_made=calls_made,
            last_text=last_text,
            subject=subject,
        )
        if isinstance(admission, Halt):
            yield await _done(emit, last_text, calls_made, admission.reason, spent)
            return

        if isinstance(tools, DynamicTools):
            prepared = tools.prepare(messages)
            if isawaitable(prepared):
                await prepared
        available = tools.list()
        bound = model.bind_tools(as_model_tools(available)) if available else model
        request_identity = _model_request_identity(model, available, model_identity)

        # ── Reason ───────────────────────────────────────────────────────────
        reply: AIMessageChunk | None = None
        turn_model = ""
        turn_response_metadata: dict[str, Any] = {}
        failed: _FailedModel | None = None
        async with aclosing(
            _model_stream(
                bound,
                messages,
                aborted,
                on_model_failure,
                compact_context,
                emit,
                invoke_model,
                request_identity,
            )
        ) as model_stream:
            async for item in model_stream:
                if isinstance(item, _FailedModel):
                    failed = item
                    break
                chunk = item
                # Read identity before addition: LangChain concatenates repeated string metadata,
                # so two terminal chunks naming one model otherwise become a fictitious name.
                turn_model = _model_of(chunk) or turn_model
                # Response metadata is provider-owned passthrough data. Last-write-wins keeps every
                # exposed field without applying LangChain's string-concatenating chunk merge.
                turn_response_metadata.update(chunk.response_metadata)
                # Chunks add, and the sum reassembles tool arguments that arrive as fragments.
                reply = chunk if reply is None else reply + chunk
                for thought in _reasoning_of(chunk):
                    yield {"type": "thinking", "content": thought}
                if delta := chunk.text:
                    yield {"type": "text", "text": delta}
                if aborted():
                    yield await _done(
                        emit, _text_of(reply), calls_made, "aborted", spent, mid_turn=True
                    )
                    return
        if failed is not None:
            yield await _failed_model_outcome(failed, aborted(), emit, calls_made, spent)
            return

        turn_text = _text_of(reply)
        last_text = turn_text
        # Kept, not overwritten with a blank: a provider that names the model on some turns and not
        # others should not erase what it already told us.
        spent.model = turn_model or spent.model
        # Resolved first, then charged, so a turn the provider did not label lands on the model
        # already known rather than on a second, nameless bucket.
        if counts := _usage_of(reply):
            spent.by_model.setdefault(spent.model, Counter()).update(counts)

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        requested_tool_calls = select_for_execution(tools, list(reply.tool_calls) if reply else [])

        if not requested_tool_calls:
            assistant = _assistant_turn(
                reply, turn_text, [], response_metadata=turn_response_metadata
            )
            messages.append(assistant)
            await _record(record_messages, [assistant])
            finish, carried_inputs = await _decide_finish(
                turn=turn,
                turn_text=turn_text,
                calls_made=calls_made,
                messages=messages,
                subject=subject,
                should_stop_after_turn=should_stop_after_turn,
                drain_inputs=drain_inputs,
                controls=controls,
            )
            match finish:
                case Proceed():
                    # Inputs remain uncommitted until the next model admission point.
                    # `before_model` may still halt, in which case claiming the model received
                    # them would be an audit-log lie.
                    continue
                case Halt(halt_reason):
                    yield await _done(emit, turn_text, calls_made, halt_reason, spent)
                    return

        # ── Act ──────────────────────────────────────────────────────────────
        assistant = _assistant_turn(
            reply, turn_text, requested_tool_calls, response_metadata=turn_response_metadata
        )
        messages.append(assistant)
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
            _control_ctx(
                turn=turn,
                messages=messages,
                calls_made=calls_made,
                text=turn_text,
                subject=subject,
            ),
        )
        # `record_pending` and the durable effects land before this write. If the process died
        # earlier, the durable model step replays this exact assistant turn on the next attempt;
        # recording it before execution would leave a suspended or failed round as transcript
        # fact before its durable boundary had accepted it.
        await _record(record_messages, [assistant])
        for call, result, refused in resolved:
            event = tool_result(call, result)
            event["executed"] = not refused
            yield event

        round_ = absorb_round(tools, resolved)
        carried_inputs += _round_inputs(round_)

        if aborted():
            await _record(record_messages, _round_messages(round_))
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        # ── Stop? ────────────────────────────────────────────────────────────
        # Also asked above on tool-free rounds. Even a terminating tool reaches it: the hook is
        # where budget and verification accounting lives, and it must see every completed round.
        stop_reason = await _tool_round_stop_reason(
            round_, turn, turn_text, calls_made, should_stop_after_turn
        )
        if stop_reason is not None:
            # There is no next model round whose input admission could persist these answers.
            # `TranscriptRecorder.onEvent` in the TypeScript runtime likewise flushes a complete
            # tool-result group immediately rather than leaving it one step behind.
            await _record(record_messages, _round_messages(round_))
            yield await _done(
                emit, last_text or "(stopped after turn)", calls_made, stop_reason, spent
            )
            return


async def _failed_model_outcome(
    failed: "_FailedModel",
    aborted: bool,
    emit: Emit | None,
    calls_made: list[dict[str, Any]],
    spent: _Spend,
) -> dict[str, Any]:
    """Render the single terminal event for a provider failure."""
    if aborted:
        return await _done(emit, failed.partial, calls_made, "aborted", spent, mid_turn=True)

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
    return {"type": "error", **{key: value for key, value in cut.items() if key != "reason"}}


async def _refresh_system_message(
    messages: list[BaseMessage],
    system_prompt: str | SystemPromptSource | None,
    managed: bool,
) -> bool:
    """Refresh the loop-owned system message while preserving caller history."""
    if system_prompt is None:
        return managed
    rendered = system_prompt if isinstance(system_prompt, str) else await system_prompt.render()
    if rendered:
        message = SystemMessage(rendered)
        if managed:
            messages[0] = message
        else:
            messages.insert(0, message)
        return True
    if managed:
        messages.pop(0)
    return False


def _control_ctx(
    *,
    turn: int,
    messages: list[BaseMessage],
    calls_made: list[dict[str, Any]],
    text: str,
    subject: str,
    pending: list[PendingInput] | None = None,
) -> Ctx:
    """Snapshot model-visible state for a control point."""
    return Ctx(
        turn=turn,
        messages=[*messages, *(item.message for item in pending or [])],
        calls_made=list(calls_made),
        text=text,
        subject=subject,
    )


async def _admit_turn_inputs(
    inputs: list[PendingInput],
    *,
    messages: list[BaseMessage],
    controls: Controls | None,
    discard_inputs: DiscardInputs | None,
    admit_inputs: AdmitInputs | None,
    record_messages: RecordMessages | None,
    emit: Emit | None,
    turn: int,
    calls_made: list[dict[str, Any]],
    last_text: str,
    subject: str,
) -> Halt | None:
    """Screen, steer, and commit one ordered input group at the model boundary."""
    if controls is not None and inputs:
        # Screened before `before_model` reads them: a gate deciding on the pending messages must
        # see what will actually enter context, never a pre-mask original.
        screened = await controls.on_inputs(
            _control_ctx(
                turn=turn,
                messages=messages,
                calls_made=calls_made,
                text=last_text,
                subject=subject,
            ),
            inputs,
        )
        if isinstance(screened, Halt):
            return screened
        screened = _identify_inputs(screened)
        surviving = {item.origin_id for item in screened if item.origin_id is not None}
        discarded = [
            item
            for item in inputs
            if item.origin_id is not None and item.origin_id not in surviving
        ]
        if discarded and discard_inputs is not None:
            await discard_inputs(discarded)
        inputs = screened

    if controls is not None:
        action = await controls.before_model(
            _control_ctx(
                turn=turn,
                messages=messages,
                calls_made=calls_made,
                text=last_text,
                subject=subject,
                pending=inputs,
            )
        )
        if isinstance(action, Halt):
            return action
        inputs += [PendingInput("control", message) for message in action.steers]

    await _commit_inputs(messages, inputs, admit_inputs, record_messages, emit, turn)
    return None


async def _decide_finish(
    *,
    turn: int,
    turn_text: str,
    calls_made: list[dict[str, Any]],
    messages: list[BaseMessage],
    subject: str,
    should_stop_after_turn: ShouldStopAfterTurn | None,
    drain_inputs: DrainInputs | None,
    controls: Controls | None,
) -> tuple[Proceed | Halt, list[PendingInput]]:
    """Decide whether a tool-free turn ends or carries work into another model round."""
    # The caller's turn cap covers tool-free rounds too. Without this check an always-vetoing
    # `before_finish` gate can bypass the only iteration bound the loop exposes.
    if should_stop_after_turn is not None and await should_stop_after_turn(
        turn, turn_text, calls_made
    ):
        return Halt("policy"), []

    # A steer that landed while the turn was finishing cancels the stop. Preserve its arrival
    # order, but commit it only beside the next model call.
    if drain_inputs and (late_inputs := list(await drain_inputs())):
        return Proceed(), late_inputs

    if controls is not None:
        # The last word. A verifier that says "not done yet" gets another round. This is after the
        # late-input check because an arriving steer and a policy objection are different reasons
        # to keep going, and both may apply.
        decision = await controls.before_finish(
            _control_ctx(
                turn=turn,
                messages=messages,
                calls_made=calls_made,
                text=turn_text,
                subject=subject,
            ),
            "completed",
        )
        if isinstance(decision, Proceed):
            return decision, [PendingInput("control", message) for message in decision.steers]
        return decision, []

    return Halt("completed"), []


def _round_inputs(round_: Absorbed) -> list[PendingInput]:
    """Build the next model inputs in protocol-answer, image, then context order."""
    return [
        *(
            PendingInput("tool_result", answer, str(completed["id"]))
            for answer, completed in zip(round_.answers, round_.completed, strict=True)
        ),
        *(PendingInput("tool_image", message, call_id) for call_id, message in round_.images),
        *(PendingInput("tool_context", message, call_id) for call_id, message in round_.context),
    ]


def _round_messages(round_: Absorbed) -> list[BaseMessage]:
    """Flatten an absorbed round in the same order used for model admission."""
    return [
        *round_.answers,
        *(message for _, message in round_.images),
        *(message for _, message in round_.context),
    ]


async def _tool_round_stop_reason(
    round_: Absorbed,
    turn: int,
    turn_text: str,
    calls_made: list[dict[str, Any]],
    should_stop_after_turn: ShouldStopAfterTurn | None,
) -> StopReason | None:
    """Return the terminal reason for a completed tool round, if it must stop."""
    policy_says_stop = should_stop_after_turn is not None and await should_stop_after_turn(
        turn, turn_text, calls_made
    )
    if round_.ended_by_tool:
        return "tool"
    if policy_says_stop:
        return "policy"
    return None


async def _commit_inputs(
    messages: list[BaseMessage],
    inputs: list[PendingInput],
    admit_inputs: AdmitInputs | None,
    record_messages: RecordMessages | None,
    emit: Emit | None,
    turn: int,
) -> None:
    """Append accepted inputs and record the one point where they enter model context."""
    for item in inputs:
        messages.append(item.message)
    await _record(record_messages, [item.message for item in inputs])
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


async def _record(record_messages: RecordMessages | None, messages: list[BaseMessage]) -> None:
    """Persist one ordered message group when the runtime supplied a transcript writer."""
    if record_messages is not None and messages:
        await record_messages(messages)


def _identify_inputs(inputs: list[PendingInput]) -> list[PendingInput]:
    """Keep queue identity attached when a screen replaces a LangChain message."""
    return [
        item
        if item.origin_id is None or item.message.id == item.origin_id
        else PendingInput(
            item.kind,
            item.message.model_copy(update={"id": item.origin_id}),
            item.origin_id,
        )
        for item in inputs
    ]


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
    invoke_model: InvokeModel | None,
    request_identity: dict[str, Any],
) -> AsyncGenerator[AIMessageChunk | _FailedModel, None]:
    """Stream one model round with caller-owned retry and compaction policy."""
    attempts: Counter[ModelErrorKind] = Counter()
    while True:
        reply: AIMessageChunk | None = None
        try:
            # Closing an abandoned stream reaches the provider instead of waiting for garbage
            # collection while it keeps generating billable tokens.
            def factory() -> AsyncIterator[AIMessageChunk]:
                return cast(AsyncIterator[AIMessageChunk], bound.astream(messages))

            source = (
                invoke_model(_model_step_key(request_identity, messages), factory)
                if invoke_model is not None
                else factory()
            )
            async with aclosing(cast(Any, source)) as stream:
                async for chunk in stream:
                    reply = chunk if reply is None else reply + chunk
                    yield chunk
        except ControlSignal:
            raise
        except ModelStepError as interrupted:
            raise interrupted.cause from interrupted
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


def _model_request_identity(
    model: Any,
    tools: list[dict[str, Any]],
    explicit: str | None,
) -> dict[str, Any]:
    """Describe the stable parts of a provider request that are not in its messages."""
    if explicit is not None:
        selected: Any = explicit
    else:
        selected = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameters": getattr(model, "_identifying_params", {}),
        }
    return {"model": _stable_value(selected), "tools": _stable_value(tools)}


def _model_step_key(identity: dict[str, Any], messages: list[BaseMessage]) -> str:
    """Derive one model effect id from its provider, tools, and model-visible context."""
    encoded = messages_to_dict(messages)
    for item in encoded:
        data = item.get("data")
        if isinstance(data, dict):
            item["data"] = {
                key: value
                for key, value in data.items()
                if key not in {"id", "response_metadata", "usage_metadata"}
            }
    body = {**identity, "messages": _stable_value(encoded)}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    return f"agent:model:{digest}"


def _stable_value(value: Any) -> Any:
    """Normalize provider identifiers and JSON-like request fields for deterministic hashing."""
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_value(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


_CALL_BLOCKS = frozenset({"tool_call", "tool_use", "tool_call_chunk", "invalid_tool_call"})
"""Content blocks that restate a tool call. `tool_calls` is this loop's single source for those."""


def _assistant_turn(
    reply: AIMessageChunk | None,
    text: str,
    calls: list[ToolCall],
    *,
    response_metadata: Mapping[str, Any],
) -> AIMessage:
    """The turn as the provider streamed it, so its own blocks travel back with it.

    Rebuilt from `text` alone it would lose them — reasoning above all, which the next request has
    to return unmodified, signature included, or the provider rejects the turn. Call blocks are the
    one exclusion: `calls` may be a narrowed subset of what the model asked for, and LangChain
    renders that set back into whatever shape the provider reads.
    """
    if reply is None:
        return AIMessage(content=text, tool_calls=list(calls))
    kept = (
        [
            block
            for block in reply.content
            if not (isinstance(block, dict) and block.get("type") in _CALL_BLOCKS)
        ]
        if isinstance(reply.content, list)
        else reply.content
    )
    return AIMessage(
        content=kept or text,
        tool_calls=list(calls),
        id=reply.id,
        # Where a provider without content blocks puts its reasoning — same fact, other shelf.
        additional_kwargs=dict(reply.additional_kwargs),
        # Opaque to the loop and transcript store; provider and storage adapters own these fields.
        response_metadata=dict(response_metadata),
        usage_metadata=reply.usage_metadata,
    )


def _reasoning_of(chunk: AIMessageChunk) -> list[str]:
    """The reasoning a chunk carries, which `.text` excludes by design and so needs its own read.

    Both spellings are read: the standard `reasoning` block, and the provider-native `thinking`
    one that reaches us whenever no block translator normalized it first.
    """
    if not isinstance(chunk.content, list):
        return []
    return [
        text
        for block in chunk.content
        if isinstance(block, dict)
        and block.get("type") in ("reasoning", "thinking")
        and (text := str(block.get("reasoning") or block.get("thinking") or ""))
    ]


def _text_of(reply: AIMessageChunk | None) -> str:
    """The assistant text of a turn, whatever shape the provider streamed it in."""
    return reply.text if reply is not None else ""


def _model_of(reply: AIMessageChunk | None) -> str:
    """Return the provider-reported model that produced a reply."""
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
        return _invalid_request_kind(failure)

    status = _status_code_of(failure)
    if status == 429:
        return "rate_limit"
    if status is not None and status >= 500:
        return "server"
    if status == 400:
        return _invalid_request_kind(failure)
    return "unknown"


def _invalid_request_kind(failure: Exception) -> ModelErrorKind:
    """Separate an over-long prompt from every other rejected request.

    Anthropic reports overflow as a plain `invalid_request_error` carrying the token counts in its
    message alone, so no structured field distinguishes it and compaction would never be offered a
    request it could actually shrink. Message text is read here and nowhere else: every kind that a
    provider does expose structurally has already been decided by this point, and a wording change
    can only cost this one failure its compaction, never misroute another category.
    """
    return "context_overflow" if _OVERFLOW_TEXT.search(str(failure)) else "invalid_request"


_OVERFLOW_TEXT = re.compile(r"prompt is too long|context (length|window)|too many tokens", re.I)


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
    """Extract provider-reported token usage without deriving missing values.

    Prompt, completion, total, cache-read, and cache-write counts remain separate because provider
    conventions differ on whether cached tokens are included in prompt totals.
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
    """Build and publish the terminal event for a non-suspended run.

    ``mid_turn`` marks incomplete streamed content so transcript writers do not treat it as a
    completed assistant turn.
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
