"""The workspace boundaries, checked against what each distribution declares it depends on.

Every other check in this repo runs in one virtualenv where all eight distributions are installed,
so a module reaching across a boundary it never declared passes ruff, mypy and the whole suite —
and then fails for whoever installs that distribution alone. These two tests are the only thing
standing between a declared dependency list and a decorative one.
"""

import ast
import tomllib
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent


class Distribution(NamedTuple):
    name: str
    module: str
    """Import name — the distribution name with hyphens turned into underscores."""
    source: Path
    declared: frozenset[str]
    """Module names of the workspace distributions this one lists as dependencies."""


def _internal(names: list[str]) -> frozenset[str]:
    """Workspace dependencies as import names. External ones (langchain, psycopg) are pip's job."""
    return frozenset(
        requirement.split(">")[0].split("[")[0].split("=")[0].strip().replace("-", "_")
        for requirement in names
        if requirement.startswith("nexora")
    )


def distributions() -> list[Distribution]:
    manifests = [ROOT / "pyproject.toml", *sorted(ROOT.glob("packages/*/pyproject.toml"))]
    found = []
    for manifest in manifests:
        project = tomllib.loads(manifest.read_text())["project"]
        module = project["name"].replace("-", "_")
        source = manifest.parent / "src" / module
        assert source.is_dir(), f"{project['name']} declares no src/{module}"
        found.append(
            Distribution(project["name"], module, source, _internal(project["dependencies"]))
        )
    return found


def _imported_modules(source: Path) -> dict[str, str]:
    """Workspace modules this source tree imports, mapped to the file that imports them."""
    reached: dict[str, str] = {}
    for path in sorted(source.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root.startswith("nexora"):
                        reached.setdefault(root, str(path.relative_to(ROOT)))
                continue
            else:
                continue
            if root.startswith("nexora"):
                reached.setdefault(root, str(path.relative_to(ROOT)))
    return reached


def test_no_distribution_imports_beyond_what_it_declares() -> None:
    """An import the manifest does not cover works here and fails on a standalone install.

    `nexora-store` is the one this matters most for: its dependency list is empty on purpose, and
    an accidental `nexora_contracts` import would make a store implementation drag in a model SDK
    without anything in the normal check loop noticing.
    """
    for dist in distributions():
        for module, where in _imported_modules(dist.source).items():
            if module == dist.module:
                continue
            assert module in dist.declared, (
                f"{where} imports {module}, which {dist.name} does not declare — "
                f"declared: {sorted(dist.declared) or 'nothing'}"
            )


def test_every_declared_workspace_dependency_is_actually_imported() -> None:
    """The other direction, which caught a real mistake: `nexora-ui` was given a dependency on
    `nexora-permissions` that it never imported, so the console would have installed a rule table
    it does not use."""
    for dist in distributions():
        reached = set(_imported_modules(dist.source))
        for declared in dist.declared:
            assert declared in reached, (
                f"{dist.name} declares {declared} and never imports it"
            )
