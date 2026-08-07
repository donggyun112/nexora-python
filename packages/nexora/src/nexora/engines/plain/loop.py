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
from collections.abc import AsyncIterator
from contextlib import aclosing
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
    DrainInputs,
    Emit,
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
    spent: Counter[str] = Counter()
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
        try:
            # `aclosing` is what makes an abort reach the provider. Leaving an `async for` does
            # not close the generator it was iterating — Python waits for garbage collection —
            # so abandoning a stream leaves the HTTP connection open and the model generating
            # tokens nobody will read, and still billed. Closing it throws `GeneratorExit` down
            # into the transport, which ends the request.
            #
            # It also covers the caller: if whoever is iterating `react_loop` walks away, this
            # generator is closed, and that closes the provider stream too.
            async with aclosing(bound.astream(messages)) as stream:
                async for chunk in stream:
                    # Chunks add, and the sum reassembles tool arguments that arrived as JSON
                    # fragments — the one part of streaming worth not writing ourselves.
                    reply = chunk if reply is None else reply + chunk
                    # `.text` and not `.content`: a provider may stream content blocks rather than
                    # a plain string, and reading `content` directly meant a block-shaped chunk
                    # produced no `text` event at all — the caller saw an empty answer until the
                    # run finished. The property concatenates the text blocks and ignores the rest.
                    if delta := chunk.text:
                        yield {"type": "text", "text": delta}
                    # Checked per chunk, not just per round: a SIGTERM arriving early in a long
                    # generation should not wait out the rest of it. Still a poll at a point the
                    # loop chose — a callback firing at an arbitrary await would make the step
                    # sequence depend on timing, and a replay could not reproduce it.
                    if aborted():
                        yield await _done(
                            emit, _text_of(reply), calls_made, "aborted", spent, mid_turn=True
                        )
                        return
        except Exception as failure:
            # A provider failure ends the run as a reported error rather than an exception
            # escaping into the caller's event loop. Cancellation is not an `Exception`, so it
            # still propagates.
            #
            # Either way the text this turn had already streamed travels with the ending. It is
            # the fragment a person watching has already read, and dropping it leaves their screen
            # holding words no transcript knows about. `_text_of(reply)` and not `last_text`: the
            # fragment belongs to the turn that died, not to the one before it.
            fragment = _text_of(reply)
            if aborted():
                yield await _done(emit, fragment, calls_made, "aborted", spent, mid_turn=True)
            else:
                # A failed run has to reach the event log too, or the audit record just stops.
                # `partial`, never `content`: an unfinished turn is not an answer. Nobody knows
                # whether this one was about to call a tool, so a host may show the fragment but
                # must not append it as a completed assistant turn. Absent when there is none, the
                # way `usage` is — an empty fragment is not a fact worth carrying.
                cut: dict[str, Any] = {"reason": "error", "message": str(failure)}
                if fragment:
                    cut["partial"] = fragment
                if emit is not None:
                    await emit(EventType.STOP_FAILURE, cut)
                yield {"type": "error", **{k: v for k, v in cut.items() if k != "reason"}}
            return

        turn_text = _text_of(reply)
        last_text = turn_text
        spent.update(_usage_of(reply))

        if aborted():
            yield await _done(emit, last_text, calls_made, "aborted", spent)
            return

        requested_tool_calls = select_for_execution(tools, list(reply.tool_calls) if reply else [])

        if not requested_tool_calls:
            messages.append(AIMessage(turn_text))
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
                # A gate that vetoes forever runs forever — this loop caps nothing by design, and
                # `should_stop_after_turn` is not consulted on a round that asked for no tools.
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
        # Always asked, even when a tool already ended the run: the hook is where budget and
        # verification accounting lives, and it must see every completed round.
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


def _text_of(reply: AIMessageChunk | None) -> str:
    """The assistant text of a turn, whatever shape the provider streamed it in."""
    return reply.text if reply is not None else ""


def _usage_of(reply: AIMessageChunk | None) -> dict[str, int]:
    """Token counts if the provider reported any. Absent and zero are different facts."""
    usage = getattr(reply, "usage_metadata", None) if reply is not None else None
    if not usage:
        return {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


async def _done(
    emit: Emit | None,
    content: str,
    calls_made: list[dict[str, Any]],
    reason: StopReason,
    spent: Counter[str],
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
    if spent:
        done["usage"] = dict(spent)
    return done
