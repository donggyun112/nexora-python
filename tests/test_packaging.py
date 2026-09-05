"""Two boundaries: between distributions, and between the layers inside `semora`.

The first exists because every check here runs in one virtualenv where all three distributions are
installed, so a module reaching across a boundary it never declared passes ruff, mypy and the
suite — and then fails for whoever installs that distribution alone. The store is the one this
matters most for: its dependency list is empty on purpose.

The second keeps the core's line at the effect: `contracts` knows nothing above it, `controls`
is vocabulary and composition, `effects` is the boundary, `runtime` is what happens between
attempts. A layer imports only the ones below it.
"""

import ast
import tomllib
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOTS = (ROOT / "packages", ROOT / "tests")

LAYERS = ["contracts", "ids", "controls", "transcript", "effects", "dispatch", "runtime", "agent"]
"""Core modules, lowest first. Each may import only the ones before it."""


class Distribution(NamedTuple):
    name: str
    module: str
    source: Path
    declared: frozenset[str]


def _internal(names: list[str]) -> frozenset[str]:
    return frozenset(
        requirement.split(">")[0].split("[")[0].split("=")[0].strip().replace("-", "_")
        for requirement in names
        if requirement.startswith("semora")
    )


def distributions() -> list[Distribution]:
    assert "project" not in tomllib.loads((ROOT / "pyproject.toml").read_text()), (
        "the workspace root is virtual — a `[project]` there would be another distribution"
    )
    found = []
    for manifest in sorted(ROOT.glob("packages/*/pyproject.toml")):
        project = tomllib.loads(manifest.read_text())["project"]
        module = project["name"].replace("-", "_")
        source = manifest.parent / "src" / module
        assert source.is_dir(), f"{project['name']} declares no src/{module}"
        found.append(
            Distribution(project["name"], module, source, _internal(project["dependencies"]))
        )
    assert len(found) == 3, f"expected three distributions, found {[d.name for d in found]}"
    return found


def _imports(path: Path) -> list[tuple[str, int]]:
    """Every imported module root in one file, with its line."""
    reached: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.append(("." * node.level + node.module, node.lineno))
        elif isinstance(node, ast.Import):
            reached.extend((alias.name, node.lineno) for alias in node.names)
    return reached


def test_all_python_imports_are_eager() -> None:
    """An import below module scope would hide a dependency until that branch executes."""
    lazy: list[str] = []
    for source in PYTHON_ROOTS:
        for path in sorted(source.rglob("*.py")):
            tree = ast.parse(path.read_text())
            top_level = {id(node) for node in tree.body}
            lazy.extend(
                f"{path.relative_to(ROOT)}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Import | ast.ImportFrom) and id(node) not in top_level
            )
    assert lazy == [], "imports must be module-level:\n" + "\n".join(lazy)


def test_no_distribution_imports_beyond_what_it_declares() -> None:
    for dist in distributions():
        for path in sorted(dist.source.rglob("*.py")):
            for module, line in _imports(path):
                root = module.split(".")[0]
                if not root.startswith("semora") or root == dist.module:
                    continue
                assert root in dist.declared, (
                    f"{path.relative_to(ROOT)}:{line} imports {root}, which {dist.name} "
                    f"does not declare — declared: {sorted(dist.declared) or 'nothing'}"
                )


def test_the_store_knows_nothing_of_pydantic_ai() -> None:
    """Storage adapters implement behavior contracts over opaque values."""
    store = next(d for d in distributions() if d.name == "semora-store")
    assert store.declared == frozenset()
    for path in sorted(store.source.rglob("*.py")):
        for module, line in _imports(path):
            assert not module.startswith("pydantic"), f"{path.relative_to(ROOT)}:{line}"


def test_core_layers_import_only_downward() -> None:
    core = next(d for d in distributions() if d.name == "semora").source
    for index, layer in enumerate(LAYERS):
        allowed = set(LAYERS[:index])
        for module, line in _imports(core / f"{layer}.py"):
            if module.startswith("."):
                name = module.lstrip(".").split(".")[0]
                assert name in allowed, f"{layer}.py:{line} imports {name}, which is above it"
