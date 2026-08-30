"""The API reference names symbols that exist.

`docs/API.md` is written for an agent that will copy its import lines verbatim, so a rename or a
removed export has to break something. This is that something: every `from ... import ...` in
every fenced Python block is executed against the installed packages.

A block whose imports are optional (`semora_store_pg` needs psycopg) is skipped rather than
failed — the extra is not a test dependency.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "docs" / "API.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
OPTIONAL = {"semora_store_pg", "semora_fork", "semora_permissions", "semora_ui"}


def statements(block: str) -> list[str]:
    """Split a block into `from ... import ...` statements, closing wrapped parentheses."""
    found, current = [], ""
    for line in block.splitlines():
        line = line.split("  #")[0].rstrip()
        if current:
            current += " " + line.strip()
        elif line.startswith("from "):
            current = line
        else:
            continue
        if current.count("(") == current.count(")"):
            found.append(current)
            current = ""
    return found


def imports() -> list[tuple[str, str]]:
    """Every (module, name) pair the reference tells a reader to import."""
    pairs = []
    for block in BLOCK.findall(DOC.read_text()):
        for statement in statements(block):
            parsed = ast.parse(statement).body[0]
            assert isinstance(parsed, ast.ImportFrom) and parsed.module
            pairs += [(parsed.module, alias.name) for alias in parsed.names]
    return sorted(set(pairs))


def test_the_reference_contains_import_lines_to_check() -> None:
    """An empty extraction would make the test below pass by vacuum."""
    assert len(imports()) >= 40


@pytest.mark.parametrize(("module", "name"), imports())
def test_every_documented_import_resolves(module: str, name: str) -> None:
    try:
        imported = importlib.import_module(module)
    except ImportError:
        if module.split(".")[0] in OPTIONAL:
            pytest.skip(f"{module} is an optional extra")
        raise
    assert hasattr(imported, name), f"docs/API.md imports {name!r} from {module}, which lacks it"
