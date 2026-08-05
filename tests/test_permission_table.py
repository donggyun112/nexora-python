"""The permission decision table. This is the spec; the resolver exists to satisfy it.

`(rules, mode, subscriber answer, tool) → decision`, one row per invariant. Written before the
rule engine on purpose: the two rows that matter most are

* a subscriber's `allow` plus a deny rule → **deny** (a hook is an opinion, not authority)
* bypass mode plus a content-scoped ask rule → **ask** (the immune region is not tool-wide only)

Those two hold the whole model up. If either flips, the ordering has drifted.
"""

from typing import Any, cast

import pytest
from nexora_permissions import Mode, Rule, resolve_rules

from nexora.contracts import ToolCall

READ = cast(
    ToolCall,
    {"id": "c1", "name": "read", "args": {"path": "notes.md"}, "type": "tool_call"},
)
BASH = cast(ToolCall, {
    "id": "c2",
    "name": "bash",
    "args": {"command": "npm publish --tag next"},
    "type": "tool_call",
})
GIT = cast(
    ToolCall,
    {"id": "c3", "name": "write", "args": {"path": ".git/config"}, "type": "tool_call"},
)


def deny(tool: str, content: str | None = None) -> Rule:
    return Rule(effect="deny", tool=tool, content=content)


def ask(tool: str, content: str | None = None) -> Rule:
    return Rule(effect="ask", tool=tool, content=content)


def allow(tool: str, content: str | None = None) -> Rule:
    return Rule(effect="allow", tool=tool, content=content)


# (label, call, rules, mode, subscriber answer, tool definition, expected decision kind)
TABLE: list[tuple[str, Any, list[Rule], Mode, Any, dict[str, Any], str]] = [
    # ── the two that hold the model up ────────────────────────────────────────
    (
        "subscriber allow is overridden by a deny rule",
        READ, [deny("read")], "default", {"type": "allow"}, {}, "deny",
    ),
    (
        "bypass does not lift a content-scoped ask rule",
        BASH, [ask("bash", "npm publish")], "bypass", None, {}, "ask",
    ),
    # ── 1a-1g are immune to bypass ────────────────────────────────────────────
    ("bypass does not lift a tool-wide deny", READ, [deny("read")], "bypass", None, {}, "deny"),
    ("bypass does not lift a tool-wide ask", READ, [ask("read")], "bypass", None, {}, "ask"),
    (
        "bypass does not lift the tool's own deny",
        READ, [], "bypass", None, {"deny": "read-only mount"}, "deny",
    ),
    (
        "bypass does not lift requires_user_interaction",
        READ, [], "bypass", None, {"requires_user_interaction": True}, "ask",
    ),
    ("bypass does not lift a safety check", GIT, [], "bypass", None, {}, "ask"),
    # ── 2a: bypass applies below the immune region ────────────────────────────
    (
        "bypass allows what would otherwise passthrough to ask",
        READ, [], "bypass", None, {}, "allow",
    ),
    # ── ordering inside the immune region ─────────────────────────────────────
    (
        "deny beats ask for the same tool",
        READ, [ask("read"), deny("read")], "default", None, {}, "deny",
    ),
    (
        "a subscriber deny is honoured immediately",
        READ, [allow("read")], "default", {"type": "error", "message": "no"}, {}, "deny",
    ),
    (
        "a subscriber ask is not lifted by an allow rule",
        READ, [allow("read")], "default", {"type": "suspend", "pending_id": "p"}, {}, "ask",
    ),
    # ── 2b / 3 ────────────────────────────────────────────────────────────────
    ("a tool-wide allow rule allows", READ, [allow("read")], "default", None, {}, "allow"),
    ("nothing matched is passthrough → ask", READ, [], "default", None, {}, "ask"),
    (
        "a content rule does not match a different content",
        BASH, [ask("bash", "rm -rf")], "default", None, {}, "ask",
    ),
    (
        "an allow rule scoped to content allows only that content",
        BASH, [allow("bash", "npm publish")], "default", None, {}, "allow",
    ),
    # ── mode transformation, applied last ─────────────────────────────────────
    ("dont_ask turns a passthrough ask into deny", READ, [], "dont_ask", None, {}, "deny"),
    (
        "dont_ask does not turn a deny into an allow",
        READ, [deny("read")], "dont_ask", None, {}, "deny",
    ),
    (
        "dont_ask still denies an immune ask",
        GIT, [], "dont_ask", None, {}, "deny",
    ),
]


@pytest.mark.parametrize(
    ("call", "rules", "mode", "answer", "definition", "expected"),
    [pytest.param(*row[1:], id=row[0]) for row in TABLE],
)
async def test_the_decision_table(
    call: Any,
    rules: list[Rule],
    mode: Mode,
    answer: Any,
    definition: dict[str, Any],
    expected: str,
) -> None:
    decision = await resolve_rules(
        call, rules=rules, mode=mode, subscriber_answer=answer, definition=definition
    )
    assert _kind(decision) == expected


def _kind(decision: dict[str, Any] | None) -> str:
    if decision is None:
        return "allow"
    return {"error": "deny", "suspend": "ask"}[decision["type"]]


async def test_one_definition_of_the_rule_order() -> None:
    """The drift guard. Claude Code writes 1a-1g twice - `checkRuleBasedPermissions` and
    `hasPermissionsToUseToolInner` - and two copies of an ordering is a bug waiting to happen.

    Here the "a subscriber said allow" path and the ordinary path call the same function, so
    there is nothing to keep in sync. The assertion is that the two agree.
    """
    rules = [deny("read")]

    with_hook = await resolve_rules(
        READ, rules=rules, mode="default", subscriber_answer={"type": "allow"}, definition={}
    )
    without_hook = await resolve_rules(
        READ, rules=rules, mode="default", subscriber_answer=None, definition={}
    )

    assert _kind(with_hook) == _kind(without_hook) == "deny"
