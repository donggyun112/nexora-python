# ADR-006: Interactive input cancels and switches

## Status

Accepted.

## Decision

A provider tool call in an `AIMessage` is a model request, not proof that an effect ran. Per-call
permission therefore sits between that request and the executor. Every request must nevertheless
receive a matching `ToolMessage` before another human message can enter model context.

All incoming input first enters the durable inbox. An input carrying a suspension `pending_id`
resolves that suspension. A new interactive user input without that id performs one durable
transition:

1. finish every unanswered, unexecuted request with an error ToolMessage (`code=cancelled`);
2. mark a pre-tool effect step done with that cancelled result, preventing late execution;
3. close the active suspension;
4. order the replacing HumanMessage after the cancellation answers;
5. resume the same run.

Suspension is only legal at the pre-tool permission boundary. A tool result cannot suspend after
execution, so every cancelled suspension is known to refer to an unexecuted effect. The transition
metadata and ordered inbox inserts commit atomically.

New continuations record `origin=pre_tool_use`. Legacy `kind=effect_approval` records remain
resumable; ambiguous records and legacy `kind=elicitation` records fail closed so an upgrade cannot
reinterpret an already-executed question tool as permission to execute it again.

`input_mode="headless"` is the non-interactive mode: new input stays queued and the existing
suspension remains authoritative until explicitly resolved. Dependency ordering admits the
eventual ToolMessage before those queued user inputs.

Run forking is rejected. It would introduce transcript sharing and concurrent ownership semantics
that conflict with the run lease and fencing model without a current product requirement.
