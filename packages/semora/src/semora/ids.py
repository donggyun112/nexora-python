"""Branch identifiers that sort in creation order."""

import os
import time
from uuid import UUID

__all__ = ["new_branch_id"]


def new_branch_id(prefix: str = "branch") -> str:
    """A UUIDv7 with a prefix: unique, and later ids sort after earlier ones."""
    # ponytail: hand-rolled v7; swap for `uuid.uuid7()` once 3.14 is the floor
    millis = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (
        (millis << 80)
        | (0x7 << 76)
        | ((rand >> 64) & 0x0FFF) << 64
        | (0b10 << 62)
        | (rand & ((1 << 62) - 1))
    )
    return f"{prefix}-{UUID(int=value)}"
