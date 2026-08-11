"""Plain asynchronous ReAct loop compatible with LangChain chat models.

The loop owns deterministic ``model -> tools -> model`` control flow. Policy, persistence,
cancellation, input admission, and iteration limits are supplied through explicit collaborators.
Behavior is ported from ``packages/architectures/src/react.ts``.
"""

import hashlib
import json
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
        if system_prompt is not None:
            rendered_prompt = (
                system_prompt if isinstance(system_prompt, str) else await system_prompt.render()
            )
            if rendered_prompt:
                message = SystemMessage(rendered_prompt)
                if managed_system:
                    messages[0] = message
                else:
                    messages.insert(0, message)
                    managed_system = True
            elif managed_system:
                messages.pop(0)
                managed_system = False
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
                    subject=subject,
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
                    subject=subject,
                )
            )
            match action:
                case Halt(halt_reason):
                    yield await _done(emit, last_text, calls_made, halt_reason, spent)
                    return
                case Proceed(more):
                    pending_inputs += [PendingInput("control", message) for message in more]
        await _commit_inputs(
            messages,
            pending_inputs,
            admit_inputs,
            record_messages,
            emit,
            turn,
        )

        if isinstance(tools, DynamicTools):
            prepared = tools.prepare(messages)
            if isawaitable(prepared):
                await prepared
        available = tools.list()
        bound = model.bind_tools(as_model_tools(available)) if available else model
        request_identity = _model_request_identity(model, available, model_identity)

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
                invoke_model,
                request_identity,
            )
        ) as model_stream:
            async for item in model_stream:
                if isinstance(item, _FailedModel):
                    failed = item
                    break
                chunk = item
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
            assistant = _assistant_turn(reply, turn_text, [])
            messages.append(assistant)
            await _record(record_messages, [assistant])
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
                        subject=subject,
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
        assistant = _assistant_turn(reply, turn_text, requested_tool_calls)
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
            Ctx(
                turn=turn,
                messages=list(messages),
                calls_made=list(calls_made),
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
        carried_inputs += [
            PendingInput("tool_result", answer, str(completed["id"]))
            for answer, completed in zip(round_.answers, round_.completed, strict=True)
        ]
        # After the answers, never among them: an image message is a second message about a call
        # already answered, and a tool answer that has not landed yet cannot be commented on.
        carried_inputs += [
            PendingInput("tool_image", message, call_id) for call_id, message in round_.images
        ]
        carried_inputs += [
            PendingInput("tool_context", message, call_id) for call_id, message in round_.context
        ]

        if aborted():
            await _record(
                record_messages,
                [
                    *round_.answers,
                    *(message for _, message in round_.images),
                    *(message for _, message in round_.context),
                ],
            )
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        # ── Stop? ────────────────────────────────────────────────────────────
        # Also asked above on tool-free rounds. Even a terminating tool reaches it: the hook is
        # where budget and verification accounting lives, and it must see every completed round.
        policy_says_stop = should_stop_after_turn is not None and await should_stop_after_turn(
            turn, turn_text, calls_made
        )
        if round_.ended_by_tool or policy_says_stop:
            # There is no next model round whose input admission could persist these answers.
            # `TranscriptRecorder.onEvent` in the TypeScript runtime likewise flushes a complete
            # tool-result group immediately rather than leaving it one step behind.
            await _record(
                record_messages,
                [
                    *round_.answers,
                    *(message for _, message in round_.images),
                    *(message for _, message in round_.context),
                ],
            )
            reason: StopReason = "tool" if round_.ended_by_tool else "policy"
            yield await _done(emit, last_text or "(stopped after turn)", calls_made, reason, spent)
            return


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


def _assistant_turn(reply: AIMessageChunk | None, text: str, calls: list[ToolCall]) -> AIMessage:
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
