"""Single-process serving: the /api prefix, the SPA catch-all, and staleness.

The catch-all only exists when frontend/dist does, so the SPA tests skip on a
fresh clone that has never run `npm run build`.
"""
import time

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
from scripts import serve as serve_module

client = TestClient(app)

dist_missing = pytest.mark.skipif(
    not main_module.FRONTEND_DIST.is_dir(),
    reason="frontend/dist not built — run `npm run build` in frontend/",
)


def test_every_api_route_is_under_the_api_prefix():
    """A router included without prefix=API_PREFIX would be shadowed by the
    SPA catch-all and 404 in single-process serving. Catch that here."""
    paths = client.get("/openapi.json").json()["paths"]
    stray = [p for p in paths if not p.startswith("/api") and p != "/health"]
    assert stray == [], f"routes outside /api will be swallowed by the SPA: {stray}"
    assert len(paths) > 1


def test_health_answers_on_both_paths():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/health").json() == {"status": "ok"}


@dist_missing
def test_root_serves_the_spa():
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert '<div id="root">' in res.text


@dist_missing
def test_deep_link_falls_back_to_index_html():
    """The SPA owns client-side paths; a refresh must not 404."""
    assert client.get("/some/deep/link").status_code == 200
    assert '<div id="root">' in client.get("/some/deep/link").text


@dist_missing
def test_unmatched_api_path_404s_instead_of_returning_the_spa():
    """A client bug must not look like a successful deep link."""
    res = client.get("/api/definitely-not-a-route")
    assert res.status_code == 404
    assert res.json() == {"detail": "Not found"}


@dist_missing
def test_catch_all_does_not_serve_files_outside_dist():
    """`../` must not reach ./data — the SQLite DB lives there."""
    for attempt in ("/../data/survey_analyzer.db",
                    "/%2e%2e/data/survey_analyzer.db",
                    "/../../backend/app/llm.py"):
        res = client.get(attempt)
        assert b"SQLite format" not in res.content
        assert "GEMINI_API_KEY" not in res.text


def _point_serve_at(monkeypatch, root):
    """Repoint serve.py's module-level paths at a temp frontend tree."""
    frontend = root / "frontend"
    src = frontend / "src"
    dist = frontend / "dist"
    src.mkdir(parents=True)
    dist.mkdir(parents=True)
    monkeypatch.setattr(serve_module, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(serve_module, "DIST_DIR", dist)
    monkeypatch.setattr(serve_module, "BUILD_STAMP", dist / "index.html")
    monkeypatch.setattr(serve_module, "SOURCE_DIRS", [src])
    monkeypatch.setattr(serve_module, "SOURCE_FILES", [frontend / "package.json"])
    return src, dist


def test_build_state_missing(tmp_path, monkeypatch):
    _point_serve_at(monkeypatch, tmp_path)
    assert serve_module.build_state()[0] == "missing"


def test_build_state_fresh(tmp_path, monkeypatch):
    src, dist = _point_serve_at(monkeypatch, tmp_path)
    (src / "App.tsx").write_text("x")
    time.sleep(0.01)
    (dist / "index.html").write_text("built")
    assert serve_module.build_state()[0] == "fresh"


def test_build_state_stale_names_the_source_file(tmp_path, monkeypatch):
    """Serving a stale bundle silently shows an older app — the warning has to
    say which file made it stale, or it is unactionable."""
    src, dist = _point_serve_at(monkeypatch, tmp_path)
    (dist / "index.html").write_text("built")
    time.sleep(0.01)
    (src / "Ask.tsx").write_text("edited after the build")

    state, culprit = serve_module.build_state()
    assert state == "stale"
    assert culprit.name == "Ask.tsx"


def test_build_state_notices_a_nested_source_file(tmp_path, monkeypatch):
    src, dist = _point_serve_at(monkeypatch, tmp_path)
    (dist / "index.html").write_text("built")
    time.sleep(0.01)
    nested = src / "components" / "deep"
    nested.mkdir(parents=True)
    (nested / "Widget.tsx").write_text("edited")

    assert serve_module.build_state()[0] == "stale"


def test_build_state_notices_a_changed_config_file(tmp_path, monkeypatch):
    """package.json/vite.config changes alter the build as much as src does."""
    src, dist = _point_serve_at(monkeypatch, tmp_path)
    (dist / "index.html").write_text("built")
    time.sleep(0.01)
    (tmp_path / "frontend" / "package.json").write_text("{}")

    assert serve_module.build_state()[0] == "stale"
