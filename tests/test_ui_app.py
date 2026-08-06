"""The console's own wiring: configuration, routes, and the static mount.

`test_ui_execution.py` covers the streaming logic by importing the modules under it, and every one
of those imports works no matter where `app.py` thinks its files are. So a move that left `.env` and
`static/` behind broke the console while the suite stayed green. These checks touch the two things
that resolve paths at import time.
"""

import re

import pytest

fastapi = pytest.importorskip("fastapi", reason="the console is the `ui` extra")
from fastapi.testclient import TestClient  # noqa: E402
from nexora_ui.app import app  # noqa: E402
from nexora_ui.config import ENV_FILE, UI_ROOT  # noqa: E402


def test_the_static_directory_ships_inside_the_package() -> None:
    """`static/` is mounted from `UI_ROOT`, so it has to live in the module and not beside it —
    otherwise the mount resolves to nothing in a wheel."""
    assert (UI_ROOT / "static" / "index.html").is_file()
    assert (UI_ROOT / "static" / "app.js").is_file()


def test_the_console_serves_its_page_and_the_assets_that_page_asks_for() -> None:
    """Importing `app` runs the config module, mounts the static files, and builds every route.
    None of that is exercised by importing the modules beneath it.

    The asset URLs come out of `index.html` rather than being written here, so renaming the mount
    without editing the page — or the reverse — fails instead of serving a blank console.
    """
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "<html" in page.text.lower()

        referenced = set(re.findall(r'(?:src|href)="(/[^"?]+)', page.text))
        assert referenced, "the page references no local assets — the regex or the page changed"
        for url in sorted(referenced):
            asset = client.get(url)
            assert asset.status_code == 200, f"{url} is referenced by index.html and 404s"
            assert asset.content, f"{url} served nothing"


@pytest.mark.skipif(
    not (ENV_FILE.parent / "pyproject.toml").is_file(),
    reason="a source-checkout invariant: an installed wheel ships no .env.example, by design",
)
def test_the_env_example_documents_the_path_configuration_actually_reads() -> None:
    """Asserted against `ENV_FILE` itself, not against a path spelled out again here.

    `.env` moved next to the manifest while `config.py` kept looking inside the module, and the
    console silently found no key. An assertion that only checked `.env.example` exists passed
    right through that, because it never mentioned where the code looks.
    """
    assert ENV_FILE.parent / ".env.example" == ENV_FILE.with_name(".env.example")
    assert ENV_FILE.with_name(".env.example").is_file(), (
        f"config reads {ENV_FILE}, but nothing documents that location"
    )


def test_the_ledger_panel_shows_step_state_and_not_only_events() -> None:
    """The panel was titled "Orchestrator ledger" and contained the emit rail — nothing from the
    ledger appeared in it, which is why a crash was invisible where someone would look for it.

    Events say what was announced; `absent`/`running`/`done` is what recovery reads.
    """
    with TestClient(app) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        empty = client.get("/api/steps/never-ran")

    assert 'id="ledger"' in page, "the panel needs somewhere to put step state"
    assert "STEP LEDGER" in page
    assert "/api/steps/" in script, "the panel has to read the ledger, not infer it from events"
    assert empty.status_code == 200
    assert empty.json() == {"run_id": "never-ran", "steps": []}


async def test_the_ledger_reports_a_committed_step_and_an_unfinished_one() -> None:
    """The two answers the panel exists to show. A crash after a commit leaves `done` — nothing to
    repeat — and a crash before one leaves `running`, which is `Indeterminate` on the next attempt.
    """
    from nexora_ui.state import FaultInjectingMemorySteps

    log = FaultInjectingMemorySteps()
    await log.start("r", "committed")
    await log.finish("r", "committed", {"type": "text", "text": "sent"})
    await log.start("r", "interrupted")

    rows = {row["key"]: row for row in log.snapshot("r")}

    assert rows["committed"]["status"] == "done"
    assert "sent" in rows["committed"]["value"]
    assert rows["interrupted"]["status"] == "running"
    assert log.snapshot("other-run") == []
