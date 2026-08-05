"""A permission rule table as `pre_tool_use` / `on_resume` stages.

The surface only. Implementation lives in `rules.py`, so importing this package
does not execute it as a side effect of touching the namespace.
"""

from .rules import (
    Mode,
    PermissionBehavior,
    PolicyContext,
    Rule,
    resolve_rules,
)

__all__ = [
    "Mode",
    "PermissionBehavior",
    "PolicyContext",
    "Rule",
    "resolve_rules",
]
