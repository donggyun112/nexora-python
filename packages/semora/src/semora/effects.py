"""The execution boundary: replay committed effects and expose uncertain execution.

Every tool call is a durable step keyed by its `tool_call_id`. `start` commits intent before the
tool runs and `finish_effect` commits the result after; a step already `done` is answered from the
record without running the tool, and a step `running` with no result is `Indeterminate` — the
ledger will not claim the effect went out, and will not claim it did not. A model request is a
durable step too, keyed by what was asked, so a recovering attempt replays the turn instead of
paying for it again and getting a different answer.

This is also the one catch boundary. A tool that raises failed; the round did not: the exception
becomes that call's recorded error result and the model is told. Runtime signals pass through
untouched, because a second catch above this one would convert exactly the signals it let through.

The control points are asked here, at the boundary, so a policy author never sees a Pydantic AI
hook: `pre_tool_use` before the effect, `post_tool_use` after it and once per call, `on_resume`
when an approved call comes back, and the three turn-level points around the graph's nodes.
"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Collection, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any, NamedTuple, Protocol

from pydantic import TypeAdapter
from pydantic_ai import CallToolsNode, DeferredToolRequests, ModelRequestNode, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.result import FinalResult
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_jsonable_python
from pydantic_graph import End
from semora_store import Contended, ExecutionStore, Fenced, Indeterminate, Step

from .contracts import ControlSignal, PendingInput, StopReason
from .controls import (
    Continue,
    Controls,
    Ctx,
    Deny,
    Halt,
    Proceed,
    ResumeInput,
    Suspend,
    ToolDecision,
)
from .transcript import stripped

__all__ = [
    "CONCURRENCY_SAFE",
    "PENDING_ROUND",
    "Effects",
    "Inputs",
    "Record",
    "Resumed",
    "after_key",
    "model_step_key",
    "represented_inputs",
    "step_key",
]

CONCURRENCY_SAFE = "concurrency_safe"
"""Tool metadata flag. Without it a tool is a barrier: the batch runs in call order."""

PENDING_ROUND = "agent:pending-round"
"""Control key holding the latest model-issued call order, committed before any gate or effect."""

INPUT_IDS = "input_ids"
"""`ModelRequest.metadata` key naming the queued inputs a request carried into model context."""

NOT_EXECUTED = (
    "not executed: an earlier call in this round awaits approval; reissue it once that is decided"
)
"""Model-visible stand-in for a call behind a suspension. The effect did not happen."""

_NOTHING_HAPPENED = (ApprovalRequired, CallDeferred, ModelRetry)
"""Signals a tool raises before any side effect. Intent is cleared so the call can rerun."""

_RUNTIME_SIGNALS = (ControlSignal, Indeterminate, Fenced, Contended)
"""Ledger signals. Never converted into a tool result."""

_RESPONSE = TypeAdapter(ModelResponse)


def step_key(call_id: str) -> str:
    """Name the durable step for one tool call."""
    return f"tool:{call_id}"


def after_key(call_id: str) -> str:
    """Name the marker that says `post_tool_use` already crossed its boundary for one call."""
    return f"after:{call_id}"


def model_step_key(request: ModelRequestContext) -> str:
    """Derive one model effect id from its model, tools, and model-visible context."""
    model = request.model_id or getattr(request.model, "model_id", None) or repr(request.model)
    body = {
        "model": model,
        "tools": [
            [tool.name, tool.description, tool.parameters_json_schema]
            for tool in request.model_request_parameters.function_tools
        ],
        "messages": stripped(to_jsonable_python(request.messages)),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"agent:model:{hashlib.sha256(canonical.encode()).hexdigest()[:32]}"


def represented_inputs(messages: list[ModelMessage]) -> set[str]:
    """Queued inputs already in model-visible history, by the request that carried them."""
    return {
        str(input_id)
        for message in messages
        if isinstance(message, ModelRequest)
        for input_id in (message.metadata or {}).get(INPUT_IDS, [])
    }


class Resumed(NamedTuple):
    """What a parked call comes back with: the answer, the request it answers, the rules then."""

    answer: dict[str, Any]
    request: dict[str, Any]
    rules_version: str


class Inputs(Protocol):
    """Admit external inputs through one attempt."""

    async def claim(self, represented: set[str]) -> list[PendingInput]:
        """Claim inputs not already represented in model-visible history."""
        ...

    async def admit(self, items: list[PendingInput]) -> None:
        """Commit inputs appended to model-visible history."""
        ...

    async def discard(self, items: list[PendingInput]) -> None:
        """Commit inputs permanently removed by ingress controls."""
        ...


Record = Callable[[list[ModelMessage]], Awaitable[None]]
"""Persist messages in the exact shape admitted to model-visible history."""


class Effects(AbstractCapability[Any]):
    """Wrap one run's model and tool steps in the ledger and ask the control points at the boundary.

    One instance per run, holding that run's fencing token. Without a store the control plane is
    still there, but a crash may run a tool again and a suspension has nowhere to park.
    """

    def __init__(
        self,
        store: ExecutionStore | None,
        branch_id: str,
        token: int = 0,
        *,
        controls: Controls | None = None,
        retry_running: bool = False,
        rules_version: str = "",
        subject: str = "",
        resumed: Mapping[str, Resumed] | None = None,
        inputs: Inputs | None = None,
        record: Record | None = None,
        regate: Collection[str] = (),
    ) -> None:
        """Bind the ledger, the run, its token, and the policy in force for this attempt.

        Args:
            store: The execution ledger, or `None` to record nothing.
            branch_id: The run whose steps this capability keys.
            token: Fencing token from the run lease; zero means unleased.
            controls: The seven decision points. `None` gates nothing.
            retry_running: Treat a step that started and never reported as safe to run again.
                Only the caller knows whether repeating that particular effect is safe.
            rules_version: Label of the policy in force now, handed to `on_resume`.
            subject: Who the run acts for, stamped onto every control context.
            resumed: Answers for parked calls this attempt finishes, keyed by call id.
            inputs: The durable input queue, drained at every model boundary.
            record: Where admitted messages are persisted, in the shape the model sees them.
            regate: Call ids whose finished record is a copy from another run, to be gated
                again under this run's policy before it is replayed. Everything else that is
                already recorded bypasses the gate.
        """
        self.store = store
        self.branch_id = branch_id
        self.token = token
        self.controls = controls
        self.retry_running = retry_running
        self.rules_version = rules_version
        self.subject = subject
        self.resumed: Mapping[str, Resumed] = resumed or {}
        self.inputs = inputs
        self.record = record
        self.regate = set(regate)
        self.calls_made: list[dict[str, Any]] = []
        self.stop_reason: StopReason | None = None
        self.turn = 0
        self._suspended_this_round = False
        self._prefetched: list[PendingInput] = []
        self._approval_tools: set[str] = set()

    def get_ordering(self) -> CapabilityOrdering:
        """Sit outermost so no other capability catches a ledger signal first."""
        return CapabilityOrdering(position="outermost")

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Make every tool a barrier unless it is declared concurrency-safe.

        Two individually idempotent writes to one file give different results in different
        orders, so a batch runs sequentially unless every call in it says order cannot matter.

        A tool declared `requires_approval=True` is routed through the gate instead of Pydantic
        AI's own deferral: that one would park the call before any gate ran, with no `pending_id`
        an answer could be routed back to.
        """
        prepared: list[ToolDefinition] = []
        for tool in tool_defs:
            if tool.kind == "unapproved":
                self._approval_tools.add(tool.name)
                tool = replace(tool, kind="function")
            if not (tool.metadata or {}).get(CONCURRENCY_SAFE):
                tool = replace(tool, sequential=True)
            prepared.append(tool)
        return prepared

    # ── turn-level control points, around the graph's nodes ──────────────────

    async def wrap_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Admit inputs, steer the model, record the round, and verify the finish."""
        self.turn = ctx.run_step
        if isinstance(node, ModelRequestNode):
            return await self._admit(ctx, node, handler)
        if isinstance(node, CallToolsNode):
            self._suspended_this_round = False
            await self._record_pending(ctx, node)
            result = await handler(node)
            parked = isinstance(result, End) and isinstance(
                result.data.output, DeferredToolRequests
            )
            if self.record is not None and not parked:
                # After the round, never before: recording the assistant turn first would leave a
                # suspended or failed round as transcript fact.
                await self.record([node.model_response])
            return await self._decide_finish(ctx, result)
        return await handler(node)

    async def _admit(
        self, ctx: RunContext[Any], node: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Screen, steer, and commit one ordered input group at the model boundary."""
        request: ModelRequest = node.request
        pending = list(self._prefetched)
        self._prefetched = []
        if self.inputs is not None:
            pending += await self.inputs.claim(represented_inputs(ctx.messages))
        inputs = [
            *(PendingInput("user", p) for p in request.parts if isinstance(p, UserPromptPart)),
            *pending,
        ]
        steers: list[Any] = []
        if self.controls is not None:
            here = self._ctx(ctx, pending=[request] if ctx.messages[-1:] != [request] else [])
            if inputs:
                # Screened before `before_model` reads them: a gate deciding on the pending
                # messages must see what will actually enter context, never a pre-mask original.
                screened = await self.controls.on_inputs(here, inputs)
                if isinstance(screened, Halt):
                    return self._halt(ctx, screened.reason)
                surviving = {i.origin_id for i in screened if i.origin_id is not None}
                dropped = [i for i in pending if i.origin_id and i.origin_id not in surviving]
                if dropped and self.inputs is not None:
                    await self.inputs.discard(dropped)
                inputs = screened
            decision = await self.controls.before_model(here)
            if isinstance(decision, Halt):
                return self._halt(ctx, decision.reason)
            steers = list(decision.steers)
        admitted = [i for i in inputs if i.origin_id is not None]
        # Rewritten in place: this request object is what the graph appends to history.
        request.parts = [
            *(p for p in request.parts if not isinstance(p, UserPromptPart)),
            *(i.part for i in inputs),
            *steers,
        ]
        if admitted:
            request.metadata = {
                **(request.metadata or {}),
                INPUT_IDS: [i.origin_id for i in admitted],
            }
        if self.record is not None:
            await self.record([request])
        if admitted and self.inputs is not None:
            await self.inputs.admit(admitted)
        return await handler(node)

    async def _record_pending(self, ctx: RunContext[Any], node: Any) -> None:
        """Commit call order before the first policy check or external effect."""
        calls = [
            {"id": part.tool_call_id, "name": part.tool_name, "args": part.args_as_dict()}
            for part in node.model_response.parts
            if isinstance(part, ToolCallPart)
        ]
        if self.store is not None and calls:
            await self.store.write_control(
                self.branch_id, PENDING_ROUND, {"calls": calls, "turn": ctx.run_step}, self.token
            )

    async def _decide_finish(self, ctx: RunContext[Any], result: Any) -> Any:
        """A verifier that says "not done yet" gets another round. A parked run is not a finish."""
        if not isinstance(result, End) or isinstance(result.data.output, DeferredToolRequests):
            return result
        # A steer that landed while the turn was finishing cancels the stop. It is committed only
        # beside the next model call, and before the verifier: an arriving steer and a policy
        # objection are different reasons to keep going, and both may apply.
        if self.inputs is not None:
            late = await self.inputs.claim(represented_inputs(ctx.messages))
            if late:
                self._prefetched = late
                return ModelRequestNode[Any, Any](request=ModelRequest(parts=[]))
        if self.controls is None:
            self.stop_reason = "completed"
            return result
        decision = await self.controls.before_finish(self._ctx(ctx), "completed")
        if isinstance(decision, Proceed):
            return ModelRequestNode[Any, Any](request=ModelRequest(parts=list(decision.steers)))
        self.stop_reason = decision.reason
        return result

    def _halt(self, ctx: RunContext[Any], reason: StopReason) -> End[Any]:
        self.stop_reason = reason
        return End(FinalResult(output=_last_text(ctx.messages)))

    # ── the model boundary ────────────────────────────────────────────────────

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Execute or replay one model request as a durable step.

        A request that raised is known not to have exposed output, so its intent is cleared. One
        that was interrupted stays `running`, and the next attempt refuses to guess rather than
        risk a duplicate charge and a different answer.
        """
        if self.store is None:
            return await handler(request_context)
        key = model_step_key(request_context)
        step = await self.store.read(self.branch_id, key)
        if step.status == "done":
            return _RESPONSE.validate_python(step.value["response"])
        if step.status == "running":
            if not self.retry_running:
                raise Indeterminate(self.branch_id, key)
            await self.store.forget(self.branch_id, key, self.token)
        if not await self.store.start(self.branch_id, key, self.token):
            raise Indeterminate(self.branch_id, key)
        try:
            response = await handler(request_context)
        except asyncio.CancelledError:
            raise  # a cancellation is a crash, not the step's report about itself
        except BaseException:
            await self.store.forget(self.branch_id, key, self.token)
            raise
        await self.store.finish_effect(
            self.branch_id,
            key,
            {"type": "model_result", "response": to_jsonable_python(response)},
            self.token,
        )
        return response

    # ── the tool boundary ─────────────────────────────────────────────────────

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
    ) -> Any:
        """Ask the gate. A denial is a result the model sees; a suspension parks first."""
        here = self._ctx(ctx)
        decision: ToolDecision
        if call.tool_call_id not in self.regate and await self._recorded(call):
            # The effect happened. No gate can undo it, and asking a person to approve it would
            # be asking about the past: the record replays and only the journal sees it. A fork
            # that wants the new policy's verdict on a copied record says so with `regate`.
            decision = Continue()
        elif ctx.tool_call_approved:
            # `args` is what will run: the answer may have replaced the model's arguments.
            call = replace(call, args=args) if isinstance(args, dict) else call
            decision = await self._resume_decision(here, call)
        else:
            decision = await self._gate(here, call)
        match decision:
            case Deny(result):
                self._made(call, refused=True)
                raise SkipToolExecution(result)
            case Suspend(request):
                if not request.get("pending_id"):
                    raise ValueError(f"suspension of {call.tool_call_id!r} carries no pending_id")
                self._suspended_this_round = True
                self._made(call, refused=True)
                raise ApprovalRequired(metadata=dict(request))
            case _:
                self._made(call, refused=False)
                return args

    async def _gate(self, here: Ctx, call: ToolCallPart) -> ToolDecision:
        decision = (
            await self.controls.pre_tool_use(here, call)
            if self.controls is not None
            else Continue()
        )
        if isinstance(decision, Continue) and call.tool_name in self._approval_tools:
            decision = Suspend({"pending_id": call.tool_call_id})
        if self._suspended_this_round and not isinstance(decision, Suspend):
            # Behind a suspension only another suspension is kept, so every approval of the round
            # parks together. Anything else waits: it must not run before the earlier call is
            # decided.
            return Deny(NOT_EXECUTED)
        return decision

    async def _recorded(self, call: ToolCallPart) -> bool:
        """Whether this run's ledger already holds the call's finished effect."""
        if self.store is None:
            return False
        return (await self.store.read(self.branch_id, step_key(call.tool_call_id))).status == "done"

    async def _resume_decision(self, here: Ctx, call: ToolCallPart) -> ToolDecision:
        """Ask `on_resume` about an approved call that still has to run."""
        resumed = self.resumed.get(call.tool_call_id)
        if resumed is None or self.controls is None:
            return Continue()
        return await self.controls.on_resume(
            here,
            call,
            ResumeInput(resumed.answer, resumed.request, resumed.rules_version, self.rules_version),
        )

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Answer from the record, refuse to guess, or run the tool once. Then journal once."""
        if self.store is None:
            step = await self._execute(None, args, handler)
        else:
            key = step_key(call.tool_call_id)
            step = await self.store.read(self.branch_id, key)
            if step.status == "running":
                if not self.retry_running:
                    raise Indeterminate(self.branch_id, key)
                await self.store.forget(self.branch_id, key, self.token)
                step = Step("absent")
            if step.status == "absent":
                if not await self.store.start(self.branch_id, key, self.token):
                    raise Indeterminate(self.branch_id, key)
                step = await self._execute(key, args, handler)
        record = await self._journal_once(ctx, call, step.value)
        if record["ok"]:
            return record["value"]
        raise ToolFailed(record["error"])

    async def _execute(
        self, key: str | None, args: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Step:
        try:
            value = await handler(args)
        except _NOTHING_HAPPENED:
            if self.store is not None and key is not None:
                await self.store.forget(self.branch_id, key, self.token)
            raise
        except _RUNTIME_SIGNALS:
            raise
        except Exception as error:  # the one catch boundary
            record: dict[str, Any] = {"ok": False, "error": str(error)}
        else:
            record = {"ok": True, "value": value}
        # A cancellation between `handler` and here leaves the step `running`, which is the truth.
        if self.store is not None and key is not None:
            await self.store.finish_effect(self.branch_id, key, record, self.token)
        return Step("done", record)

    async def _journal_once(
        self, ctx: RunContext[Any], call: ToolCallPart, record: dict[str, Any]
    ) -> dict[str, Any]:
        """Replay the model-visible projection, or journal a copy of the effect result.

        Every crash between the hook and its marker re-runs the hook, so exactly-once still
        requires the hook itself to be idempotent per call id.
        """
        key = after_key(call.tool_call_id)
        if self.store is not None:
            journal = await self.store.read(self.branch_id, key)
            if journal.status == "done":
                projection = journal.value.get("record")
                if not isinstance(projection, dict):
                    raise ValueError("journal completion has no recorded model-visible result")
                return deepcopy(projection)
        if self.controls is None:
            return record
        projected = deepcopy(record)
        result = (
            projected["value"]
            if projected["ok"]
            else {"type": "error", "message": projected["error"]}
        )
        await self.controls.post_tool_use(self._ctx(ctx), call, result)
        if self.store is not None:
            await self.store.write_control(
                self.branch_id, key, {"hooked": True, "record": projected}, self.token
            )
        return projected

    # ── context ───────────────────────────────────────────────────────────────

    def _ctx(self, ctx: RunContext[Any], *, pending: list[ModelMessage] | None = None) -> Ctx:
        messages = [*ctx.messages, *(pending or [])]
        return Ctx(
            turn=ctx.run_step,
            messages=messages,
            calls_made=list(self.calls_made),
            text=_last_text(messages),
            subject=self.subject,
        )

    def _made(self, call: ToolCallPart, *, refused: bool) -> None:
        entry: dict[str, Any] = {
            "id": call.tool_call_id,
            "name": call.tool_name,
            "input": call.args_as_dict(),
        }
        if refused:
            entry["refused"] = True
        self.calls_made.append(entry)


def _last_text(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return "".join(part.content for part in message.parts if isinstance(part, TextPart))
    return ""
