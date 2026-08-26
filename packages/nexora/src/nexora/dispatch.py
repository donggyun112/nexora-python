"""Host commands and the composable transition table that routes them.

Adapters (HTTP, CLI, queue worker, webhook) translate their wire format into one of these
commands and hand it to a :class:`CommandRouter` — usually the :func:`default_router` preset
behind ``AgentRuntime.dispatch``. The router observes the run's durable state once and applies
the first transition whose ``(command, state)`` row matches; the transitions themselves call
only the runtime's public primitives (``run``/``resume``/``recover``/``submit``), so taking a
row out removes exactly its behavior and nothing else.

The core provides mechanism and this module assembles policy: nothing here is required to use
the primitives directly, and a host may reorder, drop, or add transitions of its own. The
runtime is handed in as a value, never imported — the assembly layer must not import the core
it assembles.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage
from nexora_store import Contended

from .contracts import Agent, PendingInput

__all__ = [
    "Answer",
    "Command",
    "CommandRouter",
    "InvalidTransition",
    "Prompt",
    "QueueSteer",
    "Recover",
    "RecoverInterrupted",
    "ReplayJournal",
    "ResumeApproval",
    "StartRun",
    "Transition",
    "default_router",
]


@dataclass(frozen=True, slots=True)
class Prompt:
    """Deliver user input to a run.

    Starts an idle run; on a parked run, ``input_mode`` decides whether it cancels the pending
    request (interactive) or queues behind it (headless).
    """

    text: str
    prompt_id: str | None = None
    input_mode: Literal["interactive", "headless"] = "interactive"


@dataclass(frozen=True, slots=True)
class Answer:
    """Answer a suspension by its external pending id."""

    pending_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Recover:
    """Finish an interrupted run: a parked continuation, or a round the journal can replay.

    Hosts that own their history, or need ``retry_running=False``, call
    ``AgentRuntime.recover`` directly instead.
    """


Command = Prompt | Answer | Recover


class InvalidTransition(Exception):
    """A command the run's current durable state cannot accept.

    Carries the observed ``state`` and the rejected ``command`` so an adapter can map the
    refusal (e.g. to HTTP 409) without string-matching. An ``Answer`` refused this way may be
    retried once the park it races has landed — recording an answer is idempotent.
    """

    def __init__(self, state: str, command: Command) -> None:
        """Name the observed state and keep the refused command for the adapter."""
        self.state = state
        self.command = command
        super().__init__(f"{type(command).__name__} is invalid while the run is {state!r}")


@runtime_checkable
class Transition(Protocol):
    """One row of a router's transition table.

    ``applies`` is a pure predicate over the observed state and the command; ``apply`` executes
    through the runtime's public primitives. Raising :class:`~nexora_store.Contended` from
    ``apply`` is a hand-off, not a failure: the router offers the command to the next matching
    row, and re-raises only when no row is left.
    """

    def applies(self, state: str, command: "Command") -> bool:
        """Whether this transition accepts ``command`` in the observed ``state``."""
        ...

    async def apply(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        state: str,
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Execute the transition through the runtime's public primitives.

        ``runtime`` is an ``AgentRuntime``, typed loosely because the assembly layer must not
        import the core it assembles.
        """
        ...


_PARKED = frozenset({"waiting", "switching", "resuming"})
"""Continuation states a suspension can leave behind; everything else is unparked."""


@dataclass(frozen=True, slots=True)
class StartRun:
    """``Prompt`` → ``run``: deliver the text as this attempt's user input."""

    def applies(self, state: str, command: "Command") -> bool:
        """A prompt always may try to start; ``run`` itself owns the parked-run admission."""
        return isinstance(command, Prompt)

    async def apply(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        state: str,
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Run one attempt with the prompt as its input."""
        assert isinstance(command, Prompt)
        outcome: dict[str, Any] = await runtime.run(
            run_id,
            agent,
            command.text,
            prompt_id=command.prompt_id,
            input_mode=command.input_mode,
            controls=controls,
            **options,
        )
        return outcome


@dataclass(frozen=True, slots=True)
class QueueSteer:
    """``Prompt`` behind a live worker → ``submit``: queue durably for that worker's loop.

    Reached through the router's ``Contended`` hand-off from :class:`StartRun`. Without this
    row, contention is the host's problem again and ``Contended`` propagates.
    """

    def applies(self, state: str, command: "Command") -> bool:
        """Any prompt can queue; only the hand-off order decides when this row is reached."""
        return isinstance(command, Prompt)

    async def apply(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        state: str,
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Enqueue the prompt for the lease holder and report the durable input id."""
        assert isinstance(command, Prompt)
        item = await runtime.submit(
            run_id,
            PendingInput("user_prompt", HumanMessage(command.text), command.prompt_id),
            input_mode=command.input_mode,
            conversation_id=options.get("conversation_id"),
        )
        return {"type": "enqueued", "input_id": item.origin_id}


@dataclass(frozen=True, slots=True)
class ResumeApproval:
    """``Answer`` → ``resume``: revalidate and finish the call a permission gate parked."""

    def applies(self, state: str, command: "Command") -> bool:
        """An answer is offered in any state; resume validates the pending id durably."""
        return isinstance(command, Answer)

    async def apply(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        state: str,
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Resume the parked call, mapping a missing park to ``InvalidTransition``."""
        assert isinstance(command, Answer)
        try:
            outcome: dict[str, Any] = await runtime.resume(
                run_id, command.pending_id, command.payload, agent, controls=controls, **options
            )
        except LookupError as undecided:
            if isinstance(undecided, KeyError):
                raise
            raise InvalidTransition(state, command) from undecided
        return outcome


@dataclass(frozen=True, slots=True)
class RecoverInterrupted:
    """``Recover`` on a parked run → ``recover`` from the committed transcript."""

    def applies(self, state: str, command: "Command") -> bool:
        """Only a parked continuation has a transcript round to finish."""
        return isinstance(command, Recover) and state in _PARKED

    async def apply(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        state: str,
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Finish the interrupted round without replaying its model turn."""
        history = await runtime.committed_history(run_id, options.get("conversation_id"))
        outcome: dict[str, Any] = await runtime.recover(
            run_id, history, agent, controls=controls, **options
        )
        return outcome


@dataclass(frozen=True, slots=True)
class ReplayJournal:
    """``Recover`` on an open, unparked round → a prompt-less ``run``.

    A round that died before parking never reached the transcript — its assistant turn lives
    only in the step journal, so recovery is a ``run()`` whose durable model step replays that
    turn and whose effects dedup on the ledger. The same state also names a merely-busy run;
    the lease attempt inside ``run`` is what tells them apart, surfacing ``Contended`` for the
    live one.
    """

    def applies(self, state: str, command: "Command") -> bool:
        """An open run record with nothing parked is the journal-replay case."""
        return isinstance(command, Recover) and state == "interrupted"

    async def apply(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        state: str,
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Re-run the attempt with no new input; the journal supplies the interrupted turn."""
        outcome: dict[str, Any] = await runtime.run(run_id, agent, "", controls=controls, **options)
        return outcome


class CommandRouter:
    """An ordered transition table over the runtime's execution primitives.

    The router observes the run's durable state exactly once and offers the command to each
    matching row in order. ``Contended`` from a row is a hand-off to the next matching row —
    that is how :class:`StartRun` falls back to :class:`QueueSteer` — and propagates when no
    row remains, so the host can retry. A command no row accepts raises
    :class:`InvalidTransition` with the observed state.

    This is assembly, not core: reorder the rows, drop one to remove exactly its behavior, or
    add your own ``Transition``.
    """

    def __init__(self, *transitions: Transition) -> None:
        """Fix the transition rows in matching order."""
        self._transitions = transitions

    async def dispatch(
        self,
        runtime: Any,
        run_id: Any,
        agent: Agent,
        command: "Command",
        *,
        controls: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Route one command through the table. ``runtime`` is an ``AgentRuntime``."""
        if not isinstance(command, Prompt | Answer | Recover):
            raise TypeError(f"not a dispatch command: {command!r}")
        state = await runtime.state(run_id)
        contended: Contended | None = None
        for transition in self._transitions:
            if not transition.applies(state, command):
                continue
            try:
                return await transition.apply(
                    runtime, run_id, agent, command, state, controls=controls, **options
                )
            except Contended as busy:
                contended = busy
        if contended is not None:
            raise contended
        raise InvalidTransition(state, command)


def default_router() -> CommandRouter:
    """The preset behind ``AgentRuntime.dispatch``.

    Decomposable on purpose: without :class:`QueueSteer` a busy run surfaces ``Contended``,
    without :class:`RecoverInterrupted`/:class:`ReplayJournal` a ``Recover`` is refused, and a
    host's own transitions slot in anywhere in the order.
    """
    return CommandRouter(
        ResumeApproval(),
        RecoverInterrupted(),
        ReplayJournal(),
        StartRun(),
        QueueSteer(),
    )
