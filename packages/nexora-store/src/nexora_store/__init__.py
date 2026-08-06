"""The step ledger: durable intent, ambiguity detection, and exclusive execution.

Its own distribution with no dependencies at all, which is the whole claim: this is the one layer
that knows nothing about agents — no transcript, no message type, no control point, nothing from
`nexora`. A `StepLog` stores opaque values under opaque keys, so `nexora-store-pg` and anything
else anyone writes need only this package. Anything appearing in its dependency list means the
boundary moved.

`StepLog` answers one narrow question for those keys: absent, running, or done. It does not claim
exactly-once execution. A crash between an external effect and its committed result stays
`running`/`Indeterminate` until an idempotency key or reconciliation resolves the ambiguity.
"""

from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

__all__ = [
    "Contended",
    "Fenced",
    "Indeterminate",
    "InputRecord",
    "MemorySteps",
    "Step",
    "StepLog",
]


class Fenced(Exception):
    """A write arrived from a worker whose lease has been taken over. Refused.

    The failure this exists for: a worker stalls — GC pause, a hung socket, a suspended VM — past
    its lease TTL, another worker takes the run over, and then the first one wakes up and finishes
    what it was doing. It still believes it holds the lease, so no amount of renewal helps. Only
    the token does: it is stale, and the store says no.
    """

    def __init__(self, run_id: str, presented: int, issued: int) -> None:
        super().__init__(
            f"run {run_id!r} moved on: write presented token {presented}, current is {issued}"
        )
        self.run_id = run_id
        self.presented = presented
        self.issued = issued

class Contended(Exception):
    """Someone else holds this run's lease. Not an error — try later, or let them finish."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id!r} is held by another worker")
        self.run_id = run_id

class Indeterminate(Exception):
    """A step was started and never finished. Whether its effect happened is unknown.

    Raised instead of quietly re-running, because the two safe answers are opposite and only the
    caller knows which applies: re-run (the effect is idempotent, or provably did not happen), or
    reconcile against the external system. Guessing is how a payment goes out twice.

    `force_retry` clears the intent for exactly this decision.
    """

    def __init__(self, run_id: str, step: str) -> None:
        super().__init__(f"step {step!r} of run {run_id!r} may or may not have happened")
        self.run_id = run_id
        self.step = step

class Step(NamedTuple):
    """What the log knows about one step. Three states, not two.

    `running` is the state a two-state log cannot express, and it is the one that matters: a crash
    between the effect and its result leaves exactly this, and a log that only says
    "recorded / not recorded" reports it as *not recorded* and replays the effect.
    """

    status: Literal["absent", "running", "done"]
    value: Any = None

class InputRecord(NamedTuple):
    """One durable queue row, ordered within a run."""

    input_id: str
    status: Literal["pending", "claimed", "admitted"]
    value: dict[str, Any]
    sequence: int

@runtime_checkable
class StepLog(Protocol):
    """Where step results and signal answers are written down.

    **What this gives, named precisely:** durable intent, ambiguity detection, and exclusive
    execution. *Not* exactly-once — no ledger can promise that for an arbitrary external effect.
    A crash after the pharmacy accepted the request and before `finish` committed leaves the log
    unable to say whether it went out; it declines to guess (`Indeterminate`) rather than
    repeating it. So: **at-most-once automatic attempt, with the ambiguity surfaced.** Real
    exactly-once needs the receiver's own idempotency key, a transactional outbox, or a
    reconciliation query — all outside this file.

    Three write calls, because the ledger records an *intent* before the effect runs. `start` must
    be committed before `Orchestrator.run` calls the function and `finish` after it returns; a
    store that batches or defers either one gives back the guarantee they exist to provide.

    The lease is the other half. Two workers replaying one run both see `absent` and both run the
    effect, and no per-step record can prevent that. `acquire` returns a **fencing token**, not a
    boolean: renewal alone cannot make a lease safe, because a worker can stall past its TTL and
    wake up still believing it holds the run. Carrying the token on every write is what makes a
    stalled worker harmless — its writes are refused, whatever it believes.
    """

    async def read(self, run_id: str, key: str) -> Step: ...

    async def start(self, run_id: str, key: str, token: int = 0) -> None:
        """Record the intent. Committed before the effect runs. Refused on a stale token."""
        ...

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        """Record the outcome. The step is `done` only after this. Refused on a stale token."""
        ...

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float) -> int:
        """A fencing token if this owner now holds the run, 0 if someone else does.

        The token increases every time the lease changes hands, so the newest holder always has
        the highest one. A write carrying anything lower is from a worker that has been replaced.
        """
        ...

    async def release(self, run_id: str, owner: str) -> None: ...

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        """Append an input idempotently. This write comes from outside the run lease."""
        ...

    async def list_inputs(self, run_id: str) -> list[InputRecord]: ...

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None: ...

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None: ...

    async def commit_transition(
        self,
        run_id: str,
        steps: dict[str, Any],
        inputs: list[tuple[str, dict[str, Any]]],
        token: int = 0,
    ) -> set[str]:
        """Atomically finish metadata steps and append ordered, idempotent inbox inputs.

        This is for control-flow transitions, never for wrapping an external effect. It keeps a
        cancellation ToolMessage ahead of the replacing HumanMessage across process failure.
        The returned ids are the inputs newly inserted by this transaction.
        """
        ...

class MemorySteps:
    """A `StepLog` in a dict. Correct semantics, no durability — it dies with the process.

    Useful for tests and for finding out whether a workflow's step boundaries are right before
    paying for a database. The lease is real within one process, which is enough to catch a
    workflow that accidentally runs itself twice concurrently.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Step] = {}
        self._leases: dict[str, tuple[str, int]] = {}
        self._tokens: dict[str, int] = {}
        self._inputs: dict[tuple[str, str], InputRecord] = {}
        self._input_sequence: dict[str, int] = {}

    async def read(self, run_id: str, key: str) -> Step:
        return self._entries.get((run_id, key), Step("absent"))

    async def start(self, run_id: str, key: str, token: int = 0) -> None:
        self._fence(run_id, token)
        self._entries[run_id, key] = Step("running")

    async def finish(self, run_id: str, key: str, value: Any, token: int = 0) -> None:
        self._fence(run_id, token)
        self._entries[run_id, key] = Step("done", value)

    async def forget(self, run_id: str, key: str) -> None:
        self._entries.pop((run_id, key), None)

    async def acquire(self, run_id: str, owner: str, ttl_seconds: float = 60.0) -> int:
        held = self._leases.get(run_id)
        if held is not None and held[0] != owner:
            return 0
        if held is not None:
            return held[1]  # the holder renewing keeps its token
        self._tokens[run_id] = self._tokens.get(run_id, 0) + 1
        self._leases[run_id] = (owner, self._tokens[run_id])
        return self._tokens[run_id]

    async def release(self, run_id: str, owner: str) -> None:
        held = self._leases.get(run_id)
        if held is not None and held[0] == owner:
            del self._leases[run_id]

    async def enqueue_input(self, run_id: str, input_id: str, value: dict[str, Any]) -> bool:
        key = (run_id, input_id)
        if key in self._inputs:
            return False
        sequence = self._input_sequence.get(run_id, 0)
        self._input_sequence[run_id] = sequence + 1
        self._inputs[key] = InputRecord(input_id, "pending", value, sequence)
        return True

    async def list_inputs(self, run_id: str) -> list[InputRecord]:
        return sorted(
            (record for (item_run, _), record in self._inputs.items() if item_run == run_id),
            key=lambda record: record.sequence,
        )

    async def claim_input(self, run_id: str, input_id: str, token: int = 0) -> None:
        self._fence(run_id, token)
        key = (run_id, input_id)
        record = self._inputs[key]
        if record.status != "admitted":
            self._inputs[key] = InputRecord(input_id, "claimed", record.value, record.sequence)

    async def admit_inputs(self, run_id: str, input_ids: list[str], token: int = 0) -> None:
        self._fence(run_id, token)
        for input_id in input_ids:
            key = (run_id, input_id)
            record = self._inputs.get(key)
            if record is not None:
                self._inputs[key] = InputRecord(
                    input_id, "admitted", record.value, record.sequence
                )

    async def commit_transition(
        self,
        run_id: str,
        steps: dict[str, Any],
        inputs: list[tuple[str, dict[str, Any]]],
        token: int = 0,
    ) -> set[str]:
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
        """Refuse a write from a worker that has been replaced.

        `token=0` means "I hold no lease" and is not fenced. Two kinds of caller present it: a
        single-process user who never took one, and a writer from outside the run — a webhook
        answering a signal holds no lease, and fencing it would refuse the writes it exists for.
        A worker that *did* take a lease always has a non-zero token, so it cannot land here by
        accident.
        """
        issued = self._tokens.get(run_id, 0)
        if token and token < issued:
            raise Fenced(run_id, token, issued)
