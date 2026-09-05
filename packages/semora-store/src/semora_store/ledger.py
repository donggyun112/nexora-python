"""Define the durable step ledger and its in-memory implementation.

The ledger records opaque step values and distinguishes absent, running, and completed steps.
It surfaces ambiguous interrupted effects instead of claiming exactly-once execution.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from .context import ExecutionContext, ScopedStore

__all__ = [
    "Contended",
    "ConversationScopedSteps",
    "EffectCompletion",
    "EffectConflict",
    "ExecutionStore",
    "ExecutionTransition",
    "Fenced",
    "Indeterminate",
    "InputRecord",
    "MemorySteps",
    "Step",
    "StepLog",
]


class Fenced(Exception):
    """Report a write attempted with a stale fencing token."""

    def __init__(self, branch_id: str, presented: int, issued: int) -> None:
        """Initialize the error with the stale and current tokens."""
        super().__init__(
            f"branch {branch_id!r} moved on: write presented token {presented}, current is {issued}"
        )
        self.branch_id = branch_id
        self.presented = presented
        self.issued = issued


class Contended(Exception):
    """Report that another worker holds the branch lease."""

    def __init__(self, branch_id: str) -> None:
        """Initialize the error for the contended branch."""
        super().__init__(f"branch {branch_id!r} is held by another worker")
        self.branch_id = branch_id


class Indeterminate(Exception):
    """Report a step whose external effect may have occurred."""

    def __init__(self, branch_id: str, step: str) -> None:
        """Initialize the error for the interrupted step."""
        super().__init__(f"step {step!r} of branch {branch_id!r} may or may not have happened")
        self.branch_id = branch_id
        self.step = step


class EffectConflict(Exception):
    """Report an effect completion that contradicts its durable state."""

    def __init__(self, branch_id: str, key: str, reason: str) -> None:
        """Initialize the conflict with its durable coordinates."""
        super().__init__(f"effect {key!r} of branch {branch_id!r} cannot complete: {reason}")
        self.branch_id = branch_id
        self.key = key
        self.reason = reason


class Step(NamedTuple):
    """Represent the persisted state and value of one step."""

    status: Literal["absent", "running", "done"]
    value: Any = None


class InputRecord(NamedTuple):
    """Represent one durable input queue row."""

    input_id: str
    status: Literal["pending", "claimed", "admitted", "discarded"]
    value: dict[str, Any]
    sequence: int


class EffectCompletion(NamedTuple):
    """Complete one immutable effect from the state that authorized the transition."""

    key: str
    value: Any
    expected: Literal["absent", "running"] = "running"


@dataclass(frozen=True, slots=True)
class ExecutionTransition:
    """Atomically commit effect results, mutable control state, and queued inputs."""

    effects: tuple[EffectCompletion, ...] = ()
    controls: Mapping[str, Any] = field(default_factory=dict)
    inputs: tuple[tuple[str, dict[str, Any]], ...] = ()


@runtime_checkable
class ExecutionStore(ScopedStore, Protocol):
    """Persist effect intent, control transitions, run leases, and queued inputs.

    Implementations commit ``start`` before an effect executes and ``finish_effect`` afterward.
    Completed effect results are immutable. Mutable continuation and protocol state uses
    ``write_control`` or ``commit_transition`` instead. Lease-protected writes use the fencing
    token returned by ``acquire``.
    """

    def for_execution(self, context: ExecutionContext) -> "ExecutionStore":
        """Return the ledger as ``context`` sees it.

        Without a ``conversation_id`` that is the whole ledger. With one it is that conversation's
        view: a branch recorded under a conversation and the same branch id outside it are
        different branches, so a recorded effect can only replay from inside the conversation
        that recorded it.
        """
        ...

    async def read(self, branch_id: str, key: str) -> Step:
        """Return the persisted state of a step."""
        ...

    async def start(self, branch_id: str, key: str, token: int = 0) -> bool:
        """Atomically record new step intent and report whether this caller inserted it."""
        ...

    async def finish_effect(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Complete a running effect idempotently without replacing a committed result."""
        ...

    async def write_control(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Upsert mutable framework control state."""
        ...

    async def acquire(self, branch_id: str, owner: str, ttl_seconds: float) -> int:
        """Acquire or renew a run lease and return its fencing token, or zero on contention."""
        ...

    async def release(self, branch_id: str, owner: str) -> None:
        """Release a run lease held by ``owner``."""
        ...

    async def enqueue_input(self, branch_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently and report whether it was inserted."""
        ...

    async def list_inputs(self, branch_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        ...

    async def claim_input(self, branch_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        ...

    async def admit_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted to model context."""
        ...

    async def discard_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark screened-out inputs as permanently discarded."""
        ...

    async def commit_transition(
        self,
        branch_id: str,
        transition: ExecutionTransition,
        token: int = 0,
    ) -> set[str]:
        """Atomically apply typed effect, control, and input changes."""
        ...

    async def forget(self, branch_id: str, key: str, token: int = 0) -> None:
        """Remove an unfinished step. A `done` step is left alone.

        Not optional, because clearing is what makes a reported failure retryable: a step that
        raises, a model request that failed before its first chunk, and an aborted stream all end
        by removing their intent. A ledger that silently kept them would turn every one of those
        into a permanent `Indeterminate`.

        Lease-protected like every other write, and for the sharper reason: erasing intent is the
        one write that can make a *live* worker's step look like it never happened. A replaced
        worker clearing its own abandoned attempt would otherwise delete the successor's running
        intent, and the effect the successor is mid-way through would replay on the attempt after.
        """
        ...


StepLog = ExecutionStore
"""Compatibility name for the execution-store contract."""


class ConversationScopedSteps:
    """One conversation's view of a ledger: every branch it touches is filed under the conversation.

    ``ExecutionContext.conversation_id`` is opaque to the framework, so the adapters share one
    layout instead of each inventing a column: a branch of a conversation is stored as
    ``"<conversation>/<branch>"``. Leases, inputs and fencing tokens follow the same name, so two
    conversations holding the same branch id never contend. Errors the inner store raises name
    the branch as it stored it.
    """

    def __init__(self, inner: ExecutionStore, conversation_id: str) -> None:
        """Bind the view to ``inner`` and the conversation it files under."""
        self._inner = inner
        self._conversation_id = conversation_id

    def _filed(self, branch_id: str) -> str:
        return f"{self._conversation_id}/{branch_id}"

    def for_execution(self, context: ExecutionContext) -> ExecutionStore:
        """Rescope from the underlying ledger; views do not nest."""
        return self._inner.for_execution(context)

    async def read(self, branch_id: str, key: str) -> Step:
        """Return the persisted state of a step."""
        return await self._inner.read(self._filed(branch_id), key)

    async def start(self, branch_id: str, key: str, token: int = 0) -> bool:
        """Atomically record new step intent and report whether this caller inserted it."""
        return await self._inner.start(self._filed(branch_id), key, token)

    async def finish_effect(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Complete a running effect idempotently without replacing a committed result."""
        await self._inner.finish_effect(self._filed(branch_id), key, value, token)

    async def write_control(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Upsert mutable framework control state."""
        await self._inner.write_control(self._filed(branch_id), key, value, token)

    async def finish(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Compatibility alias for ``write_control``."""
        await self.write_control(branch_id, key, value, token)

    async def forget(self, branch_id: str, key: str, token: int = 0) -> None:
        """Remove an unfinished step. A `done` step is left alone."""
        await self._inner.forget(self._filed(branch_id), key, token)

    async def acquire(self, branch_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        """Acquire or renew a run lease and return its fencing token, or zero on contention."""
        return await self._inner.acquire(self._filed(branch_id), owner, ttl_seconds)

    async def release(self, branch_id: str, owner: str) -> None:
        """Release a run lease held by ``owner``."""
        await self._inner.release(self._filed(branch_id), owner)

    async def enqueue_input(self, branch_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently and report whether it was inserted."""
        return await self._inner.enqueue_input(self._filed(branch_id), input_id, value)

    async def list_inputs(self, branch_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        return await self._inner.list_inputs(self._filed(branch_id))

    async def claim_input(self, branch_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        await self._inner.claim_input(self._filed(branch_id), input_id, token)

    async def admit_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted to model context."""
        await self._inner.admit_inputs(self._filed(branch_id), input_ids, token)

    async def discard_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark screened-out inputs as permanently discarded."""
        await self._inner.discard_inputs(self._filed(branch_id), input_ids, token)

    async def commit_transition(
        self,
        branch_id: str,
        transition: ExecutionTransition,
        token: int = 0,
    ) -> set[str]:
        """Atomically apply typed effect, control, and input changes."""
        return await self._inner.commit_transition(self._filed(branch_id), transition, token)


class MemorySteps:
    """Implement ``ExecutionStore`` with process-local dictionaries."""

    def __init__(self) -> None:
        """Initialize empty step, lease, and input stores."""
        self._entries: dict[tuple[str, str], Step] = {}
        self._leases: dict[str, tuple[str, int, float]] = {}
        self._tokens: dict[str, int] = {}
        self._inputs: dict[tuple[str, str], InputRecord] = {}
        self._input_sequence: dict[str, int] = {}

    def for_execution(self, context: ExecutionContext) -> ExecutionStore:
        """The whole store without a conversation; one conversation's view of it with."""
        if context.conversation_id is None:
            return self
        return ConversationScopedSteps(self, context.conversation_id)

    async def read(self, branch_id: str, key: str) -> Step:
        """Return the stored step or an absent state."""
        record = self._entries.get((branch_id, key), Step("absent"))
        return Step(record.status, copy.deepcopy(record.value))

    async def start(self, branch_id: str, key: str, token: int = 0) -> bool:
        """Record running intent only when the step is absent."""
        self._fence(branch_id, token)
        if (branch_id, key) in self._entries:
            return False
        self._entries[branch_id, key] = Step("running")
        return True

    async def finish_effect(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Complete a running effect while preserving an existing result."""
        self._fence(branch_id, token)
        record = self._entries.get((branch_id, key), Step("absent"))
        if record.status == "done":
            if record.value == value:
                return
            raise EffectConflict(branch_id, key, "a different result is already committed")
        if record.status != "running":
            raise EffectConflict(branch_id, key, f"expected 'running', found {record.status!r}")
        # Stored as a copy, the way a database would store it. The caller keeps mutating
        # the dict it handed in — a journal rewrites a tool result in place — and the
        # record's whole point is to still say what the tool returned.
        self._entries[branch_id, key] = Step("done", copy.deepcopy(value))

    async def write_control(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Upsert mutable process-local control state."""
        self._fence(branch_id, token)
        self._entries[branch_id, key] = Step("done", copy.deepcopy(value))

    async def finish(self, branch_id: str, key: str, value: Any, token: int = 0) -> None:
        """Compatibility alias for ``write_control``."""
        await self.write_control(branch_id, key, value, token)

    async def forget(self, branch_id: str, key: str, token: int = 0) -> None:
        """Remove unfinished step intent while preserving completed results."""
        self._fence(branch_id, token)
        if self._entries.get((branch_id, key), Step("absent")).status != "done":
            self._entries.pop((branch_id, key), None)

    async def acquire(self, branch_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        """Acquire or renew a run lease using a monotonic TTL.

        Returns:
            Current fencing token, or ``0`` when another owner holds the lease.
        """
        held = self._leases.get(branch_id)
        expired = held is not None and held[2] <= monotonic()
        if held is not None and held[0] == owner:
            self._leases[branch_id] = (owner, held[1], monotonic() + ttl_seconds)
            return held[1]  # the holder renewing keeps its token
        if held is not None and not expired:
            return 0
        # A new lease or a takeover. Either way the token moves, so a previous holder is fenced.
        self._tokens[branch_id] = self._tokens.get(branch_id, 0) + 1
        self._leases[branch_id] = (owner, self._tokens[branch_id], monotonic() + ttl_seconds)
        return self._tokens[branch_id]

    async def release(self, branch_id: str, owner: str) -> None:
        """Expire an owned lease without resetting its fencing token."""
        held = self._leases.get(branch_id)
        if held is not None and held[0] == owner:
            self._leases[branch_id] = ("", held[1], monotonic())

    async def enqueue_input(self, branch_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input unless its identifier already exists."""
        key = (branch_id, input_id)
        if key in self._inputs:
            return False
        sequence = self._input_sequence.get(branch_id, 0)
        self._input_sequence[branch_id] = sequence + 1
        self._inputs[key] = InputRecord(input_id, "pending", value, sequence)
        return True

    async def list_inputs(self, branch_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        return sorted(
            (record for (item_run, _), record in self._inputs.items() if item_run == branch_id),
            key=lambda record: record.sequence,
        )

    async def claim_input(self, branch_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        self._fence(branch_id, token)
        key = (branch_id, input_id)
        # Absent is a no-op, not a `KeyError`: the durable store expresses this as an `update` that
        # matches no row, and a caller claiming an input that is already gone must not crash on one
        # store and continue on the other.
        record = self._inputs.get(key)
        if record is not None and record.status not in {"admitted", "discarded"}:
            self._inputs[key] = InputRecord(input_id, "claimed", record.value, record.sequence)

    async def admit_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted."""
        self._fence(branch_id, token)
        for input_id in input_ids:
            key = (branch_id, input_id)
            record = self._inputs.get(key)
            if record is not None and record.status != "discarded":
                self._inputs[key] = InputRecord(input_id, "admitted", record.value, record.sequence)

    async def discard_inputs(self, branch_id: str, input_ids: list[str], token: int = 0) -> None:
        """Make screened-out inputs terminal without deleting their idempotency keys."""
        self._fence(branch_id, token)
        for input_id in input_ids:
            key = (branch_id, input_id)
            record = self._inputs.get(key)
            if record is not None:
                self._inputs[key] = InputRecord(
                    input_id, "discarded", record.value, record.sequence
                )

    async def commit_transition(
        self,
        branch_id: str,
        transition: ExecutionTransition,
        token: int = 0,
    ) -> set[str]:
        """Atomically apply a process-local control transition."""
        self._fence(branch_id, token)
        completed: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for effect in transition.effects:
            if effect.key in seen:
                raise ValueError(f"effect {effect.key!r} appears twice in one transition")
            seen.add(effect.key)
            record = self._entries.get((branch_id, effect.key), Step("absent"))
            if record.status == "done":
                if record.value != effect.value:
                    raise EffectConflict(
                        branch_id, effect.key, "a different result is already committed"
                    )
                continue
            if record.status != effect.expected:
                raise EffectConflict(
                    branch_id,
                    effect.key,
                    f"expected {effect.expected!r}, found {record.status!r}",
                )
            completed.append((effect.key, effect.value))
        overlap = seen.intersection(transition.controls)
        if overlap:
            raise ValueError(
                f"transition classifies keys as both effect and control: {sorted(overlap)}"
            )

        inserted: set[str] = set()
        for key, value in completed:
            self._entries[branch_id, key] = Step("done", value)
        for key, value in transition.controls.items():
            self._entries[branch_id, key] = Step("done", value)
        for input_id, value in transition.inputs:
            input_key = (branch_id, input_id)
            if input_key in self._inputs:
                continue
            sequence = self._input_sequence.get(branch_id, 0)
            self._input_sequence[branch_id] = sequence + 1
            self._inputs[input_key] = InputRecord(input_id, "pending", value, sequence)
            inserted.add(input_id)
        return inserted

    def _fence(self, branch_id: str, token: int) -> None:
        """Reject stale lease holders while allowing unleased writes with token zero."""
        issued = self._tokens.get(branch_id, 0)
        if token and token < issued:
            raise Fenced(branch_id, token, issued)
