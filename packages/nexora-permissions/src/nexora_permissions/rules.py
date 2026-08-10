"""Ordered permission rules for tool execution and approval resumption.

Explicit denials, tool restrictions, and protected-path checks take precedence over execution
modes. Mode handling is applied only after all applicable rules have been evaluated.
"""

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, NamedTuple

from nexora.contracts.types import ToolCall
from nexora.controls import Continue, Deny, ResumeInput, Suspend, ToolDecision

__all__ = [
    "Mode",
    "PermissionBehavior",
    "PolicyContext",
    "Rule",
    "escalation_guard",
    "resolve_rules",
]

Mode = Literal["default", "bypass", "dont_ask"]
"""Fallback behavior for tool calls not settled by explicit rules."""

PermissionBehavior = Literal["allow", "deny", "ask"]
"""Decision expressed by a permission rule."""


class Rule(NamedTuple):
    """Permission decision scoped to a tool and optional input prefix.

    Attributes:
        effect: Decision applied when the rule matches.
        tool: Tool name to match.
        content: Input prefix, or ``None`` to match every call to the tool.
    """

    effect: PermissionBehavior
    tool: str
    content: str | None = None


SAFETY_MARKERS = (".git/", ".agent/", ".ssh/", ".bashrc", ".zshrc", ".profile")
"""Protected path fragments that always require approval."""


class PolicyContext(NamedTuple):
    """Permission rules and execution mode owned by a supervisor.

    Attributes:
        rules: Ordered permission rules.
        mode: Fallback behavior for unresolved calls.
        version: Human-readable policy version for audit and telemetry.
    """

    rules: list[Rule] = []  # noqa: RUF012 — NamedTuple defaults are per-field, not shared
    mode: Mode = "default"
    version: str = ""

    @property
    def fingerprint(self) -> str:
        """Return a deterministic identity for the effective rules and mode.

        Use this value as ``rules_version`` when suspending and resuming a tool call. The audit
        ``version`` label is intentionally excluded.
        """
        canonical = json.dumps(
            {
                "mode": self.mode,
                # `None` and `""` are different rules — one matches the tool, the other matches
                # every input of it — so they must sort and serialise apart rather than collapse.
                "rules": sorted(
                    ([rule.effect, rule.tool, rule.content] for rule in self.rules),
                    key=lambda row: (row[0], row[1], row[2] is None, row[2] or ""),
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def stage(self, tools: Any) -> Callable[[Any, ToolCall], Awaitable[ToolDecision]]:
        """Build a ``pre_tool_use`` stage for this policy.

        Args:
            tools: Tool registry used to read per-tool permission metadata.

        Returns:
            Async stage that allows, denies, or suspends each tool call.
        """

        async def evaluate(ctx: Any, call: ToolCall) -> ToolDecision:
            result = await resolve_rules(
                call,
                rules=self.rules,
                mode=self.mode,
                definition=tools.get(call["name"]),
            )
            if result is None:
                return Continue()
            # Stamped on the refusal itself, which is what the audit event publishes and what a
            # suspension stores. An approval request that cannot say who it is about is a request
            # somebody has to correlate by hand before they can answer it.
            subject = getattr(ctx, "subject", "")
            decided = {**result, "subject": subject} if subject else result
            return Deny(decided) if decided["type"] == "error" else Suspend(decided)

        return evaluate

    def resume_stage(self, tools: Any) -> Any:
        """Build an ``on_resume`` stage that revalidates an approval.

        Args:
            tools: Tool registry used to read per-tool permission metadata.

        Returns:
            Async stage evaluated against the current policy fingerprint.
        """

        async def evaluate(ctx: Any, call: ToolCall, resume: ResumeInput) -> ToolDecision:
            result = await resolve_rules(
                call,
                rules=self.rules,
                mode=self.mode,
                subscriber_answer={"type": "allow"},
                definition=tools.get(call["name"]),
            )
            if result is None:
                return Continue()
            subject = getattr(ctx, "subject", "")
            decided = {**result, "subject": subject} if subject else result
            if decided["type"] == "error":
                return Deny(decided)
            if self.fingerprint == resume.suspended_rules_version:
                return Continue()
            # A second question, so it is stamped like the first: whoever answers this one is
            # answering about the same subject, and the record has to say which.
            return Suspend(decided)

        return evaluate


def escalation_guard(
    ceiling: Sequence[str] | None,
) -> Callable[[Any, ToolCall], Awaitable[ToolDecision]]:
    """Build a stage that denies tools outside a delegation ceiling.

    Args:
        ceiling: Allowed tool names, or ``None`` to permit every tool.

    Returns:
        A ``pre_tool_use`` stage that enforces the ceiling.
    """
    if ceiling is None:
        return _permits_everything
    allowed = frozenset(ceiling)

    async def stage(_ctx: Any, call: ToolCall) -> ToolDecision:
        if call["name"] in allowed:
            return Continue()
        return Deny(
            {
                "type": "error",
                "message": (
                    f"{call['name']} is outside this agent's authority; "
                    "a delegated agent cannot reach a tool its parent could not"
                ),
            }
        )

    return stage


async def _permits_everything(_ctx: Any, _call: ToolCall) -> ToolDecision:
    """Allow a call under an unrestricted authority ceiling."""
    return Continue()


async def resolve_rules(
    call: ToolCall,
    *,
    rules: list[Rule],
    mode: Mode = "default",
    subscriber_answer: dict[str, Any] | None = None,
    definition: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve the effective permission decision for a tool call.

    Args:
        call: LangChain tool call to evaluate.
        rules: Ordered permission rules.
        mode: Fallback behavior for unresolved calls.
        subscriber_answer: Optional decision from an earlier control stage.
        definition: Optional tool metadata such as ``deny`` or ``shell_content``.

    Returns:
        ``None`` to allow, an error result to deny, or a suspension result to request approval.
    """
    tool = call["name"]
    content = _content_of(call)
    definition = definition or {}
    parts = _parts(content, definition)
    asked: dict[str, Any] | None = None

    # ── 0. the subscriber's opinion ────────────────────────────────────────────
    if subscriber_answer is not None:
        kind = subscriber_answer.get("type")
        if kind == "error":
            return subscriber_answer
        if kind == "suspend":
            asked = subscriber_answer

    # ── 1a-1g. the immune region: nothing below can lift what these decide ─────
    # Content-scoped too, not `tool_wide_only`: a deny is a deny at whatever scope it was
    # written, and `Rule` already promises that. Checking only tool-wide ones here meant
    # `deny bash(rm -rf:*)` decided nothing — it fell past every branch to "nothing matched",
    # which asks a person in `default` and **runs the command in `bypass`**. The ask path below
    # still splits tool-wide from content-scoped, because those two straddle the tool's own deny.
    if _matches(rules, "deny", tool, parts):
        return _deny(f"denied by rule: {tool}")
    if _matches(rules, "ask", tool, parts, tool_wide_only=True):
        asked = asked or _ask(call, f"rule asks about {tool}")

    if definition.get("deny"):
        return _deny(str(definition["deny"]))
    if definition.get("requires_user_interaction"):
        asked = asked or _ask(call, f"{tool} needs a person")

    if _matches(rules, "ask", tool, parts):
        asked = asked or _ask(call, f"rule asks about {tool} with this input")
    if _unsafe(call):
        asked = asked or _ask(call, "touches a protected path")

    if asked is not None:
        return _transform(asked, mode)

    # ── 2a. only from here may anything answer "allow" ────────────────────────
    if mode == "bypass":
        return None
    if _allowed(rules, tool, parts):
        return None

    # ── 3. nothing matched ───────────────────────────────────────────────────
    return _transform(_ask(call, "no rule matched"), mode)


def _transform(asked: dict[str, Any], mode: Mode) -> dict[str, Any] | None:
    """Apply fallback mode behavior to an approval request."""
    if mode == "dont_ask":
        return _deny(f"cannot ask: {asked.get('reason', 'approval required')}")
    return asked


def _matches(
    rules: list[Rule],
    effect: PermissionBehavior,
    tool: str,
    parts: list[str],
    *,
    tool_wide_only: bool = False,
) -> bool:
    """Return whether any call part matches a deny or approval rule."""
    for rule in rules:
        if rule.effect != effect or rule.tool != tool:
            continue
        if rule.content is None:
            return True
        if not tool_wide_only and any(part.startswith(rule.content) for part in parts):
            return True
    return False


def _allowed(rules: list[Rule], tool: str, parts: list[str]) -> bool:
    """Return whether allow rules cover every executable part of a call."""
    scoped = [rule for rule in rules if rule.effect == "allow" and rule.tool == tool]
    if any(rule.content is None for rule in scoped):
        return True
    return bool(parts) and all(
        any(rule.content is not None and part.startswith(rule.content) for rule in scoped)
        for part in parts
    )


_SEPARATORS = re.compile(r"&&|\|\||>>?|[;&|\n]|\$\(|`")
"""Shell operators that delimit separately evaluated command parts."""


def _parts(content: str, definition: dict[str, Any]) -> list[str]:
    """Split shell-aware tool input into independently evaluated command parts."""
    if not definition.get("shell_content"):
        return [content]
    return [part for raw in _SEPARATORS.split(content) if (part := raw.strip())] or [content]


def _content_of(call: ToolCall) -> str:
    """Extract the tool input used by content-scoped rules."""
    args = call.get("args") or {}
    for key in ("command", "path", "url", "query"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return " ".join(str(v) for v in args.values())


def _unsafe(call: ToolCall) -> bool:
    target = _content_of(call)
    return any(part in target for part in SAFETY_MARKERS)


def _deny(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def _ask(call: ToolCall, reason: str) -> dict[str, Any]:
    """Build a non-blocking suspension keyed by the tool call identifier."""
    return {"type": "suspend", "pending_id": call["id"], "reason": reason}
