"""Define the durable step ledger and its in-memory implementation.

The ledger records opaque step values and distinguishes absent, running, and completed steps.
It surfaces ambiguous interrupted effects instead of claiming exactly-once execution.
"""

from time import monotonic
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

__all__ = [
    "ClearableSteps",
    "Contended",
    "Fenced",
    "Indeterminate",
    "InputRecord",
    "MemorySteps",
    "Step",
    "StepLog",
]


class Fenced(Exception):
    """Report a write attempted with a stale fencing token."""

    def __init__(self, run_id: str, presented: int, issued: int) -> None:
        """Initialize the error with the stale and current tokens."""
        super().__init__(
            f"run {run_id!r} moved on: write presented token {presented}, current is {issued}"
        )
        self.run_id = run_id
        self.presented = presented
        self.issued = issued

class Contended(Exception):
    """Report that another worker holds the run lease."""

    def __init__(self, run_id: str) -> None:
        """Initialize the error for the contended run."""
        super().__init__(f"run {run_id!r} is held by another worker")
        self.run_id = run_id

class Indeterminate(Exception):
    """Report a step whose external effect may have occurred."""

    def __init__(self, run_id: str, step: str) -> None:
        """Initialize the error for the interrupted step."""
        super().__init__(f"step {step!r} of run {run_id!r} may or may not have happened")
        self.run_id = run_id
        self.step = step

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

@runtime_checkable
class StepLog(Protocol):
    """Persist step intent, results, run leases, and queued inputs.

    Implementations must commit ``start`` before an effect executes and ``finish`` afterward.
    Lease-protected writes use the fencing token returned by ``acquire``.
    """

    async def read(self, run_id: str, key: str) -> Step:
        """Return the persisted state of a step."""
        ...

    async def start(self, run_id: str, key: str, token: int = 0) -> bool:
        """Atomically record new step intent and report whether this caller inserted it."""
        ...

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        """Record a completed step result."""
        ...

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float) -> int:
        """Acquire or renew a run lease and return its fencing token, or zero on contention."""
        ...

    async def release(self, run_id: str, owner: str) -> None:
        """Release a run lease held by ``owner``."""
        ...

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently and report whether it was inserted."""
        ...

    async def list_inputs(self, run_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        ...

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        ...

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted to model context."""
        ...

    async def discard_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark screened-out inputs as permanently discarded."""
        ...

    async def commit_transition(
        self,
        run_id: str,
        steps: dict[str, Any],
        inputs: list[tuple[str, dict[str, Any]]],
        token: int = 0,
    ) -> set[str]:
        """Atomically finish metadata steps and append idempotent inbox inputs."""
        ...

@runtime_checkable
class ClearableSteps(StepLog, Protocol):
    """Optional ledger capability for removing unfinished step intent.

    Implementations must preserve completed steps so recorded effects are never replayed.
    """

    async def forget(self, run_id: str, key: str) -> None:
        """Remove an unfinished step. A `done` step is left alone."""
        ...


class MemorySteps:
    """Implement ``StepLog`` with process-local dictionaries."""

    def __init__(self) -> None:
        """Initialize empty step, lease, and input stores."""
        self._entries: dict[tuple[str, str], Step] = {}
        self._leases: dict[str, tuple[str, int, float]] = {}
        self._tokens: dict[str, int] = {}
        self._inputs: dict[tuple[str, str], InputRecord] = {}
        self._input_sequence: dict[str, int] = {}

    async def read(self, run_id: str, key: str) -> Step:
        """Return the stored step or an absent state."""
        return self._entries.get((run_id, key), Step("absent"))

    async def start(self, run_id: str, key: str, token: int = 0) -> bool:
        """Record running intent only when the step is absent."""
        self._fence(run_id, token)
        if (run_id, key) in self._entries:
            return False
        self._entries[run_id, key] = Step("running")
        return True

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        """Record a completed step after validating the fencing token."""
        self._fence(run_id, token)
        self._entries[run_id, key] = Step("done", value)

    async def forget(self, run_id: str, key: str) -> None:
        """Remove unfinished step intent while preserving completed results."""
        if self._entries.get((run_id, key), Step("absent")).status != "done":
            self._entries.pop((run_id, key), None)

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        """Acquire or renew a run lease using a monotonic TTL.

        Returns:
            Current fencing token, or ``0`` when another owner holds the lease.
        """
        held = self._leases.get(run_id)
        expired = held is not None and held[2] <= monotonic()
        if held is not None and held[0] == owner:
            self._leases[run_id] = (owner, held[1], monotonic() + ttl_seconds)
            return held[1]  # the holder renewing keeps its token
        if held is not None and not expired:
            return 0
        # A new lease or a takeover. Either way the token moves, so a previous holder is fenced.
        self._tokens[run_id] = self._tokens.get(run_id, 0) + 1
        self._leases[run_id] = (owner, self._tokens[run_id], monotonic() + ttl_seconds)
        return self._tokens[run_id]

    async def release(self, run_id: str, owner: str) -> None:
        """Expire an owned lease without resetting its fencing token."""
        held = self._leases.get(run_id)
        if held is not None and held[0] == owner:
            self._leases[run_id] = ("", held[1], monotonic())

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input unless its identifier already exists."""
        key = (run_id, input_id)
        if key in self._inputs:
            return False
        sequence = self._input_sequence.get(run_id, 0)
        self._input_sequence[run_id] = sequence + 1
        self._inputs[key] = InputRecord(input_id, "pending", value, sequence)
        return True

    async def list_inputs(self, run_id: str) -> list[InputRecord]:
        """Return a run's inputs in submission order."""
        return sorted(
            (record for (item_run, _), record in self._inputs.items() if item_run == run_id),
            key=lambda record: record.sequence,
        )

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None:
        """Mark an input as claimed unless it is already terminal."""
        self._fence(run_id, token)
        key = (run_id, input_id)
        # Absent is a no-op, not a `KeyError`: the durable store expresses this as an `update` that
        # matches no row, and a caller claiming an input that is already gone must not crash on one
        # store and continue on the other.
        record = self._inputs.get(key)
        if record is not None and record.status not in {"admitted", "discarded"}:
            self._inputs[key] = InputRecord(input_id, "claimed", record.value, record.sequence)

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        """Mark the selected inputs as admitted."""
        self._fence(run_id, token)
        for input_id in input_ids:
            key = (run_id, input_id)
            record = self._inputs.get(key)
            if record is not None and record.status != "discarded":
                self._inputs[key] = InputRecord(
                    input_id, "admitted", record.value, record.sequence
                )

    async def discard_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        """Make screened-out inputs terminal without deleting their idempotency keys."""
        self._fence(run_id, token)
        for input_id in input_ids:
            key = (run_id, input_id)
            record = self._inputs.get(key)
            if record is not None:
                self._inputs[key] = InputRecord(
                    input_id, "discarded", record.value, record.sequence
                )

    async def commit_transition(
        self,
        run_id: str,
        steps: dict[str, Any],
        inputs: list[tuple[str, dict[str, Any]]],
        token: int = 0,
    ) -> set[str]:
        """Atomically apply a process-local control transition."""
        self._fence(run_id, token)
        inserted: set[str] = set()
        for key, value in steps.items():
            self._entries[run_id, key] = Step("done", value)
        for input_id, value in inputs:
            input_key = (run_id, input_id)
            if input_key in self._inputs:
                continue
            sequence = self._input_sequence.get(run_id, 0)
            self._input_sequence[run_id] = sequence + 1
            self._inputs[input_key] = InputRecord(input_id, "pending", value, sequence)
            inserted.add(input_id)
        return inserted

    def _fence(self, run_id: str, token: int) -> None:
        """Reject stale lease holders while allowing unleased writes with token zero."""
        issued = self._tokens.get(run_id, 0)
        if token and token < issued:
            raise Fenced(run_id, token, issued)
