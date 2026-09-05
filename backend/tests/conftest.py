"""Shared test setup.

Every test runs with the admin passcode gate OFF unless it explicitly turns
it on: a developer who has ADMIN_PASSCODE set in their environment or their
config.json must not see unrelated tests start answering 401. This machine's
real config.json is expected to carry a passcode, so silencing the file read
is not optional.
"""
import pytest

from app import auth, config, tokens


@pytest.fixture(autouse=True)
def _no_admin_passcode(monkeypatch):
    monkeypatch.delenv("ADMIN_PASSCODE", raising=False)
    monkeypatch.setattr(auth, "load_config", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def _no_repo_config(monkeypatch, tmp_path):
    """Point the config module at a path that does not exist, so the model
    and price accessors see an empty config rather than the developer's real
    one. A test that wants a config writes to `tmp_path / "config.json"` and
    re-points CONFIG_PATH itself.

    The memo is cleared on both edges: it is keyed on the real file's
    (mtime, size), so a value cached by an earlier test would otherwise be
    served here, and a value cached from a tmp file would outlive the test.
    """
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "absent-config.json")
    monkeypatch.setattr(config, "_cache_stamp", None)
    yield
    config._cache_stamp = None


@pytest.fixture(autouse=True)
def _no_real_data_dir(monkeypatch, tmp_path):
    """No test may touch the real data/ tree. Ever.

    This exists because it already went wrong: a delete-endpoint test whose
    fixture redirected only uploads/ and exports/ ran the purge against the
    developer's actual data/lexicon, data/locations, data/subthemes and
    data/rulings for dataset ids 1 and 2 — test ids start at 1, and so do
    real ones. The artifacts were unrecoverable.

    Redirecting every root here rather than per-file means a NEW artifact
    directory is safe by default. A test that wants a specific layout still
    patches these itself; patching twice is harmless, forgetting once is not.
    """
    from app import db as db_module
    from app import induction, lexicon, locations, rulings, subthemes, summary

    root = tmp_path / "guarded-data"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(induction, "DATA_DIR", root)
    monkeypatch.setattr(induction, "TAXONOMY_DIR", root / "taxonomy")
    monkeypatch.setattr(db_module, "DATA_DIR", root)
    monkeypatch.setattr(db_module, "UPLOADS_DIR", root / "uploads")
    monkeypatch.setattr(db_module, "EXPORTS_DIR", root / "exports")
    monkeypatch.setattr(summary, "LABELS_DIR", root / "labels")
    monkeypatch.setattr(summary, "SUMMARY_DIR", root / "summary")
    monkeypatch.setattr(summary, "LEXICON_DIR", root / "lexicon")
    monkeypatch.setattr(summary, "LOCATIONS_DIR", root / "locations")
    monkeypatch.setattr(summary, "TAXONOMY_DIR", root / "taxonomy")
    monkeypatch.setattr(subthemes, "SUBTHEMES_DIR", root / "subthemes")
    monkeypatch.setattr(lexicon, "LEXICON_DIR", root / "lexicon")
    monkeypatch.setattr(locations, "LOCATIONS_DIR", root / "locations")
    monkeypatch.setattr(rulings, "RULINGS_DIR", root / "rulings")


@pytest.fixture(autouse=True)
def _purge_cannot_escape_tmp(monkeypatch, tmp_path):
    """Backstop: `purge.purge_files` refuses any path outside tmp_path.

    The redirection above is the fix; this is the tripwire for the next root
    somebody adds and forgets, and for any test that re-patches a constant
    back to a real location. It fails the test rather than deleting anything.
    """
    from app import purge

    real = purge.purge_files

    def guarded(dataset_id):
        for root in purge.ROOTS:
            resolved = str(root.path())
            assert resolved.startswith(str(tmp_path)), (
                f"REFUSING to purge: root {root.key!r} resolves to {resolved}, "
                f"which is outside the test's tmp_path. Redirect it in the "
                f"fixture before calling purge."
            )
        return real(dataset_id)

    monkeypatch.setattr(purge, "purge_files", guarded)


@pytest.fixture(autouse=True)
def _no_token_counting(monkeypatch):
    """No countTokens round-trips from the offline suite.

    countTokens is free, but it is still a network call against Vertex with
    the developer's real credentials — and the estimate path now reaches for
    it. Left alone, every pipeline-estimate test would depend on ADC and add
    seconds to a suite whose runtime is itself the tripwire for "a test
    started making real calls".

    Patched at `_get_client` rather than at `count`, so the fallback path the
    tests then exercise is the same one a machine without credentials takes.
    A test that wants counting substitutes its own fake client.
    """
    monkeypatch.setattr(tokens.TokenCounter, "_get_client", lambda self: None)
