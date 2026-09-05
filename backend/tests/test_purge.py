"""Permanent dataset deletion: the preview, the purge, and the refusals.

The load-bearing test here is `test_roots_covers_every_dataset_keyed_directory`.
Dataset ids are reused by the next upload, so a directory this module forgets
is not a tidiness problem — it is one survey's labels being served as another's
months later. That test reads the source for directory constants and fails when
one is not accounted for, because nothing else would notice.
"""
import json
import re
from pathlib import Path

import pytest

from app import purge

APP_DIR = Path(__file__).resolve().parent.parent / "app"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# Directories under data/ that are deliberately NOT per-dataset. Each needs a
# reason, because "not per-dataset" is exactly the assumption that goes stale.
GLOBAL_DIRS = {
    # one SQLite file for every dataset; rows go through the ORM
    "survey_analyzer.db",
    # parked manual artifacts, named by dataset+question rather than keyed by
    # dataset, and referenced by no code at all
    "validation_runs",
}


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    """Redirect every per-dataset root into a temp tree.

    Patched on the defining modules, since `purge.ROOTS` resolves each path
    through a callable at call time precisely so this works.
    """
    from app import db as db_module
    from app import induction, lexicon, locations, rulings, subthemes, summary

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(induction, "DATA_DIR", root)
    monkeypatch.setattr(db_module, "UPLOADS_DIR", root / "uploads")
    monkeypatch.setattr(db_module, "EXPORTS_DIR", root / "exports")
    monkeypatch.setattr(induction, "TAXONOMY_DIR", root / "taxonomy")
    monkeypatch.setattr(summary, "LABELS_DIR", root / "labels")
    monkeypatch.setattr(summary, "SUMMARY_DIR", root / "summary")
    monkeypatch.setattr(subthemes, "SUBTHEMES_DIR", root / "subthemes")
    monkeypatch.setattr(lexicon, "LEXICON_DIR", root / "lexicon")
    monkeypatch.setattr(locations, "LOCATIONS_DIR", root / "locations")
    monkeypatch.setattr(rulings, "RULINGS_DIR", root / "rulings")
    # purge imports these names into its own module namespace
    monkeypatch.setattr(purge, "UPLOADS_DIR", root / "uploads")
    monkeypatch.setattr(purge, "EXPORTS_DIR", root / "exports")
    monkeypatch.setattr(purge, "LABELS_DIR", root / "labels")
    monkeypatch.setattr(purge, "SUMMARY_DIR", root / "summary")
    return root


def _seed(data_root, dataset_id="7"):
    """A dataset with something in every root purge knows about."""
    for root in purge.ROOTS:
        target = purge.dataset_dir(root, dataset_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "something.json").write_text("{}", encoding="utf-8")
    # two runs with recorded spend, in the shape the real writers use
    for stage, cost in (("taxonomy", 0.25), ("labels", 0.75)):
        run = purge.dataset_dir(
            next(r for r in purge.ROOTS if r.key == stage), dataset_id) / "3" / "run1"
        run.mkdir(parents=True, exist_ok=True)
        (run / "manifest.json").write_text(
            json.dumps({"tool": f"scripts.{stage}",
                        "usage": {"est_cost_usd": cost}}), encoding="utf-8")
    return dataset_id


# --- the exhaustiveness guard ---------------------------------------------


def test_roots_covers_every_dataset_keyed_directory():
    """Every `DATA_DIR / "name"` in the codebase is purged or explicitly global.

    A directory added to the app and not added to ROOTS survives deletion, and
    because dataset ids are reused it is later adopted by unrelated data. This
    reads the source rather than trusting anyone to remember.
    """
    pattern = re.compile(
        r"""(?:DATA_DIR|_DATA_DIR)\s*/\s*["'](\w+)["']"""
        r"""|REPO_ROOT\s*/\s*["']data["']\s*/\s*["'](\w+)["']""")
    found: set[str] = set()
    for path in list(APP_DIR.glob("*.py")) + list(SCRIPTS_DIR.glob("*.py")):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            found.add(match.group(1) or match.group(2))

    assert found, "the source scan found no directory constants at all"
    covered = {root.key for root in purge.ROOTS} | GLOBAL_DIRS
    missing = found - covered
    assert not missing, (
        f"these data/ directories are keyed by dataset id but are not in "
        f"purge.ROOTS, so deleting a dataset would orphan them: {sorted(missing)}"
    )


def test_every_root_has_a_human_description():
    # the confirm dialog renders these; a blank one shows a bare directory name
    for root in purge.ROOTS:
        assert root.describes.strip(), f"{root.key} has no description"
        assert root.label.strip(), f"{root.key} has no label"


# --- preview ---------------------------------------------------------------


def test_preview_counts_files_bytes_and_recorded_spend(data_root):
    dataset_id = _seed(data_root)
    result = purge.preview(dataset_id)

    assert result.total_files > 0
    assert result.total_bytes > 0
    # 0.25 from the taxonomy manifest + 0.75 from the labels manifest
    assert result.total_spent_usd == pytest.approx(1.0)
    assert {r.key for r in result.roots if r.exists} == {
        r.key for r in purge.ROOTS}


def test_preview_of_an_untouched_dataset_is_all_zeros(data_root):
    result = purge.preview("999")
    assert result.total_files == 0
    assert result.total_spent_usd == 0.0
    assert all(not r.exists for r in result.roots)


def test_preview_reads_only(data_root):
    dataset_id = _seed(data_root)
    purge.preview(dataset_id)
    assert purge.dataset_dir(purge.ROOTS[0], dataset_id).exists()


def test_preview_survives_a_manifest_that_is_not_json(data_root):
    dataset_id = _seed(data_root)
    run = purge.dataset_dir(
        next(r for r in purge.ROOTS if r.key == "labels"), dataset_id) / "3" / "run1"
    (run / "manifest.json").write_text("truncated{", encoding="utf-8")
    # a half-written manifest is a real state after a killed run; it must not
    # turn a read-only preview into a 500
    assert purge.preview(dataset_id).total_spent_usd == pytest.approx(0.25)


# --- purging ---------------------------------------------------------------


def test_purge_removes_every_root(data_root):
    dataset_id = _seed(data_root)
    assert purge.purge_files(dataset_id) == {}
    for root in purge.ROOTS:
        assert not purge.dataset_dir(root, dataset_id).exists(), root.key


def test_purge_leaves_other_datasets_alone(data_root):
    _seed(data_root, "7")
    _seed(data_root, "8")
    purge.purge_files("7")
    for root in purge.ROOTS:
        assert not purge.dataset_dir(root, "7").exists()
        assert purge.dataset_dir(root, "8").exists(), root.key


def test_purging_a_dataset_with_no_files_is_not_an_error(data_root):
    # most datasets legitimately have no review, rulings, or subthemes tree
    assert purge.purge_files("404") == {}


def test_forget_in_memory_drops_the_ask_cache_and_the_job(monkeypatch):
    from app import ask_service, pipeline

    cleared = []
    monkeypatch.setattr(ask_service, "invalidate_context_cache", cleared.append)
    monkeypatch.setattr(pipeline, "_JOBS", {
        "job-for-7": pipeline.Job(job_id="job-for-7", dataset_id="7"),
        "job-for-8": pipeline.Job(job_id="job-for-8", dataset_id="8"),
    })

    purge.forget_in_memory("7")

    assert cleared == ["7"]
    # keyed by job id, so nothing else prunes these by dataset
    assert set(pipeline._JOBS) == {"job-for-8"}


def test_running_job_is_detected_only_while_running(monkeypatch):
    from app import pipeline

    job = pipeline.Job(job_id="j", dataset_id="7", status="running")
    monkeypatch.setattr(pipeline, "latest_job",
                        lambda ds: job if ds == "7" else None)
    assert purge.running_job("7") is job
    assert purge.running_job("8") is None

    job.status = "done"
    # a finished job must not block a delete forever
    assert purge.running_job("7") is None
