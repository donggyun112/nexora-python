"""The examples run.

Documentation that no longer works is worse than none, and every one of these scripts asserts a
runtime property in its prose — that a suspended run leaves the tool unexecuted, that recovery does
not repeat a committed write, that a `Tools` wrapper lands inside the durable step. If a rename or
a contract change breaks one, this is what says so.

No API key: each example scripts its model, which is also the point of `examples/_scripted.py`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("[0-9]*.py"))


def test_the_examples_directory_is_not_empty() -> None:
    """A glob that matches nothing would make the test below pass by vacuum."""
    assert len(EXAMPLES) >= 4


def test_every_example_runs_to_completion() -> None:
    for example in EXAMPLES:
        finished = subprocess.run(
            [sys.executable, str(example)], capture_output=True, text=True, timeout=60
        )
        assert finished.returncode == 0, (
            f"{example.name} exited {finished.returncode}\n{finished.stderr}"
        )
        assert finished.stdout.strip(), f"{example.name} printed nothing"


def test_the_langgraph_comparison_still_produces_its_claim() -> None:
    """The README states this script's output as a fact, so the output has to keep holding.

    Skipped unless langgraph is installed; it is not a dependency of this package.
    """
    pytest.importorskip("langgraph")
    script = Path(__file__).resolve().parent.parent / "examples" / "vs_langgraph.py"
    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    assert finished.returncode == 0, finished.stderr
    assert "charged 2 times by the graph, 1 time by the runtime" in finished.stdout
