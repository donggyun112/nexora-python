"""Two boundaries: between distributions, and between the layers inside `semora`.

The first exists because every other check here runs in one virtualenv where all five
distributions are installed, so a module reaching across a boundary it never declared passes ruff,
mypy and the whole suite — and then fails for whoever installs that distribution alone.

The second exists because `contracts`, `controls`, `tools`, `history`, `orchestrator` and `engines`
were briefly separate distributions and are now subpackages, on the grounds that they share one
dependency footprint and nobody installs them apart. The manifests were enforcing their layering;
`LAYERS` is what took that job over. Python's usual tool for this is `import-linter`; the rule set
is small enough that a table and a walk cost less than a dependency.
"""

import ast
import tomllib
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOTS = (ROOT / "packages", ROOT / "tests", ROOT / "examples")


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
        if requirement.startswith("semora")
    )


def distributions() -> list[Distribution]:
    """Every workspace member. The root manifest builds nothing and declares no `[project]`."""
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
    assert len(found) == 7, f"expected seven distributions, found {[d.name for d in found]}"
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
                    if root.startswith("semora"):
                        reached.setdefault(root, str(path.relative_to(ROOT)))
                continue
            else:
                continue
            if root.startswith("semora"):
                reached.setdefault(root, str(path.relative_to(ROOT)))
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
                if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top_level
            )

    assert lazy == [], "imports must be module-level:\n" + "\n".join(lazy)


def test_no_distribution_imports_beyond_what_it_declares() -> None:
    """An import the manifest does not cover works here and fails on a standalone install.

    `semora-store` is the one this matters most for: its dependency list is empty on purpose, and
    an accidental `semora.contracts` import would make a store implementation drag in a model SDK
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


PROVIDER_EXTRAS = {
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google": "langchain-google-genai",
    "xai": "langchain-xai",
    "openrouter": "langchain-openai",
}


def test_provider_extras_install_adapters_the_core_does_not_import() -> None:
    """A provider SDK in the base install would be a model Semora refused to own."""
    manifest = tomllib.loads((ROOT / "packages" / "semora" / "pyproject.toml").read_text())
    extras = manifest["project"]["optional-dependencies"]
    for extra, package in PROVIDER_EXTRAS.items():
        declared = extras[extra]
        assert len(declared) == 1, extra
        assert declared[0].startswith(f"{package}>") or declared[0].startswith(f"{package}=")

    imported: set[str] = set()
    core = ROOT / "packages" / "semora" / "src" / "semora"
    for path in sorted(core.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)

    forbidden = {name.replace("-", "_") for name in PROVIDER_EXTRAS.values()}
    assert imported.isdisjoint(forbidden), imported & forbidden


def test_every_declared_workspace_dependency_is_actually_imported() -> None:
    """Every declared workspace dependency is imported by its distribution."""
    for dist in distributions():
        reached = set(_imported_modules(dist.source))
        for declared in dist.declared:
            assert declared in reached, f"{dist.name} declares {declared} and never imports it"


LAYERS: dict[str, frozenset[str]] = {
    "contracts": frozenset(),
    # Pure host-boundary identifier generation; reaches no other Semora layer.
    "ids": frozenset(),
    # Reaches nothing, like `contracts`, and for the same kind of reason: a registry of detached
    # jobs is `asyncio.Task` bookkeeping. Knowing what a subagent is would make it one.
    "background": frozenset(),
    # Built-ins are adapters over the workspace boundary; they do not own gating or execution.
    "builtins": frozenset({"workspace"}),
    # Provider selection and workspace lifecycle adapt external runtimes without reaching the
    # planner or durable execution layers.
    "providers": frozenset(),
    "prompts": frozenset(),
    "skills": frozenset({"contracts"}),
    "tool_search": frozenset({"contracts"}),
    "workspace": frozenset({"contracts"}),
    "sandbox_remote": frozenset({"workspace"}),
    "controls": frozenset({"contracts"}),
    # Host command vocabulary and the transition table that routes it. Assembly over the
    # runtime's public primitives — it reaches only the contracts it routes for, and the
    # runtime is handed in as a value so the assembly layer never imports the core it assembles.
    "dispatch": frozenset({"contracts"}),
    # Peer of `tools`, not above it: a finish gate decides with a flag and a tool's name, and
    # reaching the execution boundary would let a goal run one.
    "goal": frozenset({"contracts", "controls"}),
    "tools": frozenset({"contracts", "controls"}),
    # Above both: a permission policy reads a tool's flag and answers with a control decision.
    # `prompts` because the mode announces itself: the reminder section renders off the same flag.
    "plan_mode": frozenset({"contracts", "controls", "prompts", "tools"}),
    # Beside `tools`, not above it: a subagent wrapper composes a `Tools` the way a host does, and
    # reaching the execution boundary would make a child's launch a second kind of tool round.
    "subagents": frozenset({"background", "contracts"}),
    "history": frozenset({"contracts", "tools"}),
    # Peer of `history`, not above it: both are codecs between a message and an opaque payload some
    # store holds. Reaching `tools` would mean the transcript had an opinion about executing one.
    "transcript": frozenset({"contracts"}),
    "orchestrator": frozenset({"contracts", "controls", "history", "tools"}),
    "orchestration": frozenset({"contracts", "orchestrator", "tools"}),
    "engines": frozenset({"contracts", "controls", "tools"}),
    "driver": frozenset({"contracts", "orchestrator"}),
    "runtime": frozenset(
        {
            "background",
            "contracts",
            "controls",
            "dispatch",
            "subagents",
            "driver",
            "engines",
            "history",
            "orchestrator",
            "orchestration",
            "transcript",
            "workspace",
        }
    ),
}
"""What each subpackage of `semora` may reach. Absent from a value means absent from the layer.

`contracts` reaching nothing is the load-bearing entry — it is the hub every other layer imports,
and a single import out of it would invert the whole thing.
"""


def _layer_of(path: Path, package: Path) -> str:
    """The top-level name under `src/semora` that owns this file."""
    return path.relative_to(package).parts[0].removesuffix(".py")


def test_no_layer_of_semora_imports_above_itself() -> None:
    """The rule the manifests used to enforce, now that these layers share one distribution.

    Without it the merge would have quietly given up the one thing the split bought: `contracts`
    could import `orchestrator`, and nothing — not ruff, not mypy, not any other test — would say
    so, because they all live in the same package now.
    """
    package = ROOT / "packages" / "semora" / "src" / "semora"
    for path in sorted(package.rglob("*.py")):
        layer = _layer_of(path, package)
        if layer == "__init__":  # `semora/__init__.py` is the facade and reaches by definition
            continue
        allowed = LAYERS[layer]
        # 0 means this file sits directly in `semora/`; 1 means one package down, and so on.
        depth = len(path.relative_to(package).parts) - 1
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level:
                # `from .x` resolves against the containing package; each extra dot climbs one.
                # Anchored above `semora` itself means the first name is a layer; anchored inside
                # a layer means the import never left it.
                if depth - (node.level - 1) != 0:
                    continue
                reached = node.module.split(".")[0]
            elif node.module.startswith("semora."):
                reached = node.module.split(".")[1]
            else:
                continue
            if reached == layer:
                continue
            assert reached in allowed, (
                f"{path.relative_to(ROOT)} ({layer}) imports {reached}, "
                f"which {layer} may not reach — allowed: {sorted(allowed) or 'nothing'}"
            )
