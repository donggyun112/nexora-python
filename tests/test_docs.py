"""Keep documented public imports executable during API migrations."""

import ast
from pathlib import Path


def test_api_reference_imports_are_available() -> None:
    reference = Path(__file__).resolve().parents[1] / "docs" / "API.md"
    examples = reference.read_text().split("```python\n")[1:]
    assert examples, "the public reference must contain executable imports"
    for block in examples:
        tree = ast.parse(block.split("```", 1)[0])
        imports: list[ast.stmt] = [
            node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        assert imports
        exec(compile(ast.Module(body=imports, type_ignores=[]), str(reference), "exec"), {})
