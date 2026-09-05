"""Learning cost rates from run manifests.

Most of this file is about what gets EXCLUDED, because every exclusion here
was a real trap in this repo's own data: a manifest written into two trees, a
resumed run missing its expensive phase, a `_review` run with no usage block
at all, and a pile of 50-row smoke tests that outnumber the production runs.
"""
import json

import pytest

from app import calibration


@pytest.fixture()
def manifests(tmp_path, monkeypatch):
    """Empty taxonomy/labels/lexicon/locations trees under a temp root."""
    roots = {}
    for name in ("taxonomy", "labels", "lexicon", "locations"):
        path = tmp_path / name
        path.mkdir()
        roots[name] = path
    monkeypatch.setattr(calibration, "TAXONOMY_DIR", roots["taxonomy"])
    monkeypatch.setattr(calibration, "LABELS_DIR", roots["labels"])
    monkeypatch.setattr(calibration, "LEXICON_DIR", roots["lexicon"])
    monkeypatch.setattr(calibration, "LOCATIONS_DIR", roots["locations"])
    return roots


def write_label_run(roots, dataset, question, run, *, responses, tokens_in,
                    tokens_out, duplicates=0, tool="scripts.label",
                    usage=True):
    run_dir = roots["labels"] / dataset / question / run
    run_dir.mkdir(parents=True)
    manifest = {
        "tool": tool,
        "created_utc": run,
        "source": {"rows_used": responses},
        "report": {"responses_total": responses,
                   "duplicate_responses_collapsed": duplicates},
    }
    if usage:
        manifest["usage"] = {"input_tokens": tokens_in,
                             "output_tokens": tokens_out,
                             "est_cost_usd": 1.0, "calls": 10}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_induce_run(roots, dataset, question, run, *, rows, chunks,
                     tokens_out, calls=10, candidates=None, covers="full_run",
                     tool="scripts.induce"):
    run_dir = roots["taxonomy"] / dataset / question / run
    run_dir.mkdir(parents=True)
    report = {"chunks": {"n_chunks": chunks}}
    if candidates is not None:
        report["candidates_proposed"] = candidates
    (run_dir / "manifest.json").write_text(json.dumps({
        "tool": tool,
        "created_utc": run,
        "source": {"rows_used": rows},
        "run_report": report,
        "usage": {"input_tokens": 1000, "output_tokens": tokens_out,
                  "calls": calls, "covers": covers},
    }), encoding="utf-8")


def write_flat_run(roots, stage, dataset, *, responses, usd, tool):
    path = roots[stage] / dataset
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({
        "tool": tool, "created_utc": "x", "n_responses": responses,
        "usage": {"calls": 1, "est_cost_usd": usd},
    }), encoding="utf-8")


# --- labeling --------------------------------------------------------------


def test_label_rates_are_pooled_per_unique_text(manifests):
    # 1000 responses, 200 of them duplicates -> 800 unique texts see the model
    write_label_run(manifests, "1", "1", "r1", responses=1000,
                    duplicates=200, tokens_in=80_000, tokens_out=40_000)
    measured = calibration.measure()

    # per UNIQUE, not per response: run_labeling collapses duplicate texts
    # before batching, so a per-response rate bakes in one corpus's
    # duplication rate and misprices every other corpus
    assert measured["label"]["input_tokens_per_unique"] == pytest.approx(100.0)
    assert measured["label"]["output_tokens_per_unique"] == pytest.approx(50.0)
    assert measured["label"]["unique_ratio"] == pytest.approx(0.8)
    assert measured["sample"]["label_runs"] == 1


def test_rates_are_pooled_not_averaged(manifests):
    """sum(tokens)/sum(rows), so a 200-row run does not outweigh a 10k one."""
    write_label_run(manifests, "1", "1", "r1", responses=200,
                    tokens_in=200_000, tokens_out=0)     # 1000 tok/response
    write_label_run(manifests, "1", "2", "r1", responses=10_000,
                    tokens_in=100_000, tokens_out=0)     # 10 tok/response
    measured = calibration.measure()
    pooled = 300_000 / 10_200
    assert measured["label"]["input_tokens_per_unique"] == pytest.approx(
        pooled, abs=0.01)
    # a mean of the per-run rates would be ~505, which the big run's evidence
    # does not support
    assert measured["label"]["input_tokens_per_unique"] < 100


def test_tiny_runs_are_excluded(manifests):
    write_label_run(manifests, "1", "1", "r1", responses=50,
                    tokens_in=999_999, tokens_out=999_999)
    measured = calibration.measure()
    # the synthetic 50-row datasets produce rates several times off the real
    # ones and outnumber the production runs
    assert measured["sample"]["label_runs"] == 0
    assert measured["label"] == calibration.BUILTIN["label"]


def test_incremental_manifests_are_excluded_by_tool(manifests):
    # scripts.label_incremental writes ONE manifest into BOTH the taxonomy and
    # the labels tree, so globbing by directory double-counts it and credits
    # induction with labeling tokens
    write_label_run(manifests, "1", "1", "r1", responses=5000,
                    tokens_in=1, tokens_out=1, tool="scripts.label_incremental")
    assert calibration.measure()["sample"]["label_runs"] == 0


def test_review_runs_without_usage_fall_back_to_the_previous_run(manifests):
    write_label_run(manifests, "1", "1", "2026-01-01T00-00-00Z_a",
                    responses=1000, tokens_in=50_000, tokens_out=25_000)
    # sorts last and carries no usage block at all
    write_label_run(manifests, "1", "1", "2026-02-01T00-00-00Z_review",
                    responses=1000, tokens_in=0, tokens_out=0,
                    tool="scripts.review", usage=False)

    measured = calibration.measure()
    assert measured["sample"]["label_runs"] == 1
    assert measured["label"]["input_tokens_per_unique"] == pytest.approx(50.0)


def test_only_the_newest_run_per_question_counts(manifests):
    # a question re-run seven times while tuning a prompt must not outvote six
    # questions run once
    write_label_run(manifests, "1", "1", "2026-01-01T00-00-00Z_a",
                    responses=1000, tokens_in=999_000, tokens_out=0)
    write_label_run(manifests, "1", "1", "2026-02-01T00-00-00Z_b",
                    responses=1000, tokens_in=50_000, tokens_out=0)
    measured = calibration.measure()
    assert measured["sample"]["label_runs"] == 1
    assert measured["label"]["input_tokens_per_unique"] == pytest.approx(50.0)


# --- induction -------------------------------------------------------------


def test_induce_output_is_measured_per_chunk(manifests):
    write_induce_run(manifests, "1", "1", "r1", rows=1000, chunks=10,
                     tokens_out=25_000, calls=17, candidates=200)
    measured = calibration.measure()
    assert measured["induce"]["output_tokens_per_chunk"] == pytest.approx(2500.0)
    assert measured["induce"]["calls_per_chunk"] == pytest.approx(1.7)
    assert measured["induce"]["candidates_per_response"] == pytest.approx(0.2)


def test_resumed_runs_are_excluded(manifests):
    # covers="consolidation_only" means the MAP spend is missing entirely, so
    # this run's tokens-per-chunk is meaningless rather than merely noisy
    write_induce_run(manifests, "1", "1", "r1", rows=1000, chunks=10,
                     tokens_out=10, covers="consolidation_only")
    assert calibration.measure()["sample"]["induce_runs"] == 0


def test_a_missing_covers_key_counts_as_a_full_run(manifests):
    run_dir = manifests["taxonomy"] / "1" / "1" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "tool": "scripts.induce", "created_utc": "r1",
        "source": {"rows_used": 1000},
        "run_report": {"chunks": {"n_chunks": 10}},
        # older scripts.induce manifests predate the key
        "usage": {"input_tokens": 1, "output_tokens": 20_000, "calls": 10},
    }), encoding="utf-8")
    assert calibration.measure()["sample"]["induce_runs"] == 1


# --- the flat stages -------------------------------------------------------


def test_flat_stage_cost_interpolates_against_corpus_size(manifests):
    write_flat_run(manifests, "locations", "1", responses=100, usd=0.001,
                   tool="scripts.build_locations")
    write_flat_run(manifests, "locations", "2", responses=10_000, usd=0.1,
                   tool="scripts.build_locations")
    cal = calibration.Calibration(calibration.measure())

    # interpolated in log10, so the midpoint of 100 and 10,000 is 1,000
    assert cal.flat_stage_usd("locations", 1000) == pytest.approx(0.0505, abs=1e-4)
    # and clamped rather than extrapolated past the evidence
    assert cal.flat_stage_usd("locations", 1) == pytest.approx(0.001)
    assert cal.flat_stage_usd("locations", 10_000_000) == pytest.approx(0.1)


def test_repeat_runs_at_one_size_collapse_to_their_median(manifests):
    cal = calibration.Calibration({
        "lexicon": {"points": [[150, 0.001], [150, 0.002], [150, 0.009]]},
    })
    # duplicate x values would otherwise make a zero-width segment whose value
    # depends on iteration order
    assert cal.flat_stage_usd("lexicon", 150) == pytest.approx(0.002)


def test_a_single_observation_is_used_flat(manifests):
    cal = calibration.Calibration({"lexicon": {"points": [[150, 0.004]]}})
    assert cal.flat_stage_usd("lexicon", 1) == pytest.approx(0.004)
    assert cal.flat_stage_usd("lexicon", 100_000) == pytest.approx(0.004)


# --- the active calibration ------------------------------------------------


def test_no_history_means_builtin_rates(manifests, monkeypatch):
    monkeypatch.setattr(calibration.config, "calibration", lambda: {})
    cal = calibration.active()
    assert cal.source == "builtin"
    assert cal.label_input_per_unique == calibration.BUILTIN[
        "label"]["input_tokens_per_unique"]


def test_a_stored_calibration_wins_and_is_labelled_measured(monkeypatch):
    monkeypatch.setattr(calibration.config, "calibration", lambda: {
        "source": "measured",
        "label": {"input_tokens_per_unique": 12.0,
                  "output_tokens_per_unique": 6.0, "unique_ratio": 0.9},
    })
    cal = calibration.active()
    assert cal.source == "measured"
    assert cal.label_input_per_unique == 12.0
    # a stage the stored payload does not mention keeps its built-in rate
    assert cal.induce_output_per_chunk == calibration.BUILTIN[
        "induce"]["output_tokens_per_chunk"]


def test_a_corrupt_stored_rate_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(calibration.config, "calibration", lambda: {
        "source": "measured",
        "label": {"input_tokens_per_unique": "lots"},
    })
    cal = calibration.active()
    # read on the cost path; a bad value must degrade, not raise
    assert cal.label_input_per_unique == calibration.BUILTIN[
        "label"]["input_tokens_per_unique"]


def test_a_nonsensical_spread_is_ignored(monkeypatch):
    for bad in ({"low": 2.0, "high": 1.0}, {"low": -1, "high": 5},
                {"low": "a", "high": "b"}):
        monkeypatch.setattr(calibration.config, "calibration",
                            lambda bad=bad: {"source": "measured", "spread": bad})
        low, high = calibration.active().spread
        # the band must always contain the headline figure
        assert low <= 1.0 <= high
