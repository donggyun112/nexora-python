"""One ordered function that decides whether a tool call may run.

The order *is* the policy, so it lives in one place and is read top to bottom. Two properties
that only hold because of where things sit:

* The **immune region** (1a-1g) runs before `bypass` is consulted, so no mode can lift a deny
  rule, a tool's own refusal, or a safety check.
* The **mode transformation** is applied last, so no early return can escape it.

Deliberately one function and not two. The reference this follows spells the same order out
twice - once for "the hook said allow, re-check the rules" and once for the ordinary path - and
two copies of an ordering drift apart. Here a subscriber's answer is an *argument*, so both paths
are the same call. `tests/test_permission_table.py` is the spec.
"""

from typing import Any, Literal, NamedTuple

from .contracts.types import ToolCall
from .controls import Continue, Deny, ResumeInput, Suspend, ToolDecision

__all__ = ["Mode", "PermissionBehavior", "PolicyContext", "Rule", "resolve_rules"]

Mode = Literal["default", "bypass", "dont_ask"]
"""How to treat what the rules did not settle. Never how to treat what they did."""

PermissionBehavior = Literal["allow", "deny", "ask"]
"""A rule's policy-level answer, before it is encoded as an execution result."""


class Rule(NamedTuple):
    """A permission rule. `content=None` matches the whole tool; otherwise a prefix of its input.

    Tool-wide and content-scoped are both immune to bypass. Making only the tool-wide ones immune
    would gut the idea - `Bash(npm publish:*)` is the case the immune region exists for.
    """

    effect: PermissionBehavior
    tool: str
    content: str | None = None


SAFETY_PREFIXES = (".git/", ".agent/", ".ssh/", ".bashrc", ".zshrc", ".profile")
"""Paths where an edit is asked about however the rules read. Cheap, and the reference's 1g."""


class PolicyContext(NamedTuple):
    """Which rules are in force, under which mode, at which version. Owned by the supervisor.

    The three travelled separately before this and so belonged to nobody: `rules` was an argument,
    `mode` was an argument, and `version` was a bare string handed to whatever wrote a suspension
    record. A resume then had to reassemble by hand what the run had been decided under.

    `version` is not used to decide anything. It is what a resume compares against to notice that
    the rules moved while a person was thinking — the window is days, so they do.
    """

    rules: list[Rule] = []  # noqa: RUF012 — NamedTuple defaults are per-field, not shared
    mode: Mode = "default"
    version: str = ""

    def stage(self, tools: Any) -> Any:
        """This context as a `pre_tool_use` stage, reading each tool's own say.

        Wrap it in `nexora.controls.gate` to put it in a `Permissions` chain.

        Placed after the hooks: a hook's `allow` does not end a chain, so the rules still run and
        re-validate it — the property Claude Code needs a second copy of its rule layer for. A
        hook's `deny` short-circuits before this, which is also what that reference does.
        """

        async def evaluate(call: ToolCall) -> dict[str, Any] | None:
            return await resolve_rules(
                call,
                rules=self.rules,
                mode=self.mode,
                definition=tools.get(call["name"]),
            )

        return evaluate

    def resume_stage(self, tools: Any) -> Any:
        """Revalidate a human approval against this policy context.

        The approval satisfies the question asked under the same rules version. A deny always
        wins; a newly changed policy that still asks creates a new suspension instead of silently
        treating the old approval as authority.
        """

        async def evaluate(_ctx: Any, call: ToolCall, resume: ResumeInput) -> ToolDecision:
            result = await resolve_rules(
                call,
                rules=self.rules,
                mode=self.mode,
                subscriber_answer={"type": "allow"},
                definition=tools.get(call["name"]),
            )
            if result is None:
                return Continue()
            if result.get("type") == "error":
                return Deny(result)
            if self.version == resume.suspended_rules_version:
                return Continue()
            return Suspend(result)

        return evaluate


async def resolve_rules(
    call: ToolCall,
    *,
    rules: list[Rule],
    mode: Mode = "default",
    subscriber_answer: dict[str, Any] | None = None,
    definition: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """`None` to allow, an `error` result to deny, a `suspend` result to ask.

    `subscriber_answer` is what a `PRE_TOOL_USE` handler said, and it is an opinion: a `deny` is
    honoured, an `ask` is remembered, and an `allow` **does not end anything** - every stage below
    still runs. That is the whole reason a hook cannot be the last word.

    `definition` is the tool's own say (`deny`, `requires_user_interaction`), read here rather
    than by the loop so the loop keeps knowing nothing about policy.
    """
    tool = call["name"]
    content = _content_of(call)
    definition = definition or {}
    asked: dict[str, Any] | None = None

    # ── 0. the subscriber's opinion ────────────────────────────────────────────
    if subscriber_answer is not None:
        kind = subscriber_answer.get("type")
        if kind == "error":
            return subscriber_answer
        if kind == "suspend":
            asked = subscriber_answer

    # ── 1a-1g. the immune region: nothing below can lift what these decide ─────
    if _matches(rules, "deny", tool, content, tool_wide_only=True):
        return _deny(f"denied by rule: {tool}")
    if _matches(rules, "ask", tool, content, tool_wide_only=True):
        asked = asked or _ask(call, f"rule asks about {tool}")

    if definition.get("deny"):
        return _deny(str(definition["deny"]))
    if definition.get("requires_user_interaction"):
        asked = asked or _ask(call, f"{tool} needs a person")

    if _matches(rules, "ask", tool, content):
        asked = asked or _ask(call, f"rule asks about {tool} with this input")
    if _unsafe(call):
        asked = asked or _ask(call, "touches a protected path")

    if asked is not None:
        return _transform(asked, mode)

    # ── 2a. only from here may anything answer "allow" ────────────────────────
    if mode == "bypass":
        return None
    if _matches(rules, "allow", tool, content):
        return None

    # ── 3. nothing matched ───────────────────────────────────────────────────
    return _transform(_ask(call, "no rule matched"), mode)


def _transform(asked: dict[str, Any], mode: Mode) -> dict[str, Any] | None:
    """Applied last, so no early return escapes it.

    `dont_ask` cannot promote anything - it only says what to do with a question nobody will
    answer, and the answer to an unanswerable question is no.
    """
    if mode == "dont_ask":
        return _deny(f"cannot ask: {asked.get('reason', 'approval required')}")
    return asked


def _matches(
    rules: list[Rule],
    effect: PermissionBehavior,
    tool: str,
    content: str,
    *,
    tool_wide_only: bool = False,
) -> bool:
    for rule in rules:
        if rule.effect != effect or rule.tool != tool:
            continue
        if rule.content is None:
            return True
        if not tool_wide_only and content.startswith(rule.content):
            return True
    return False


def _content_of(call: ToolCall) -> str:
    """The part of a call a content-scoped rule matches against.

    One string rather than per-tool fields: a rule is written by a human as a prefix, and a
    scheme that needs to know which argument matters per tool cannot be written down once.
    """
    args = call.get("args") or {}
    for key in ("command", "path", "url", "query"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return " ".join(str(v) for v in args.values())


def _unsafe(call: ToolCall) -> bool:
    target = _content_of(call)
    return any(part in target for part in SAFETY_PREFIXES)


def _deny(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def _ask(call: ToolCall, reason: str) -> dict[str, Any]:
    """A suspension, because this runtime never blocks a worker on a human.

    ponytail: `pending_id` is the call id. That is the idempotency key already (ADR-002), and a
    second identifier for the same pending decision is a second thing to keep consistent.
    """
    return {"type": "suspend", "pending_id": call["id"], "reason": reason}
