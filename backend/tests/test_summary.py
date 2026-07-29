"""Phase 4 summary: latest-run resolution, real counts, prompt rendering."""
import json

import pytest

from app import summary


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


TAXONOMY = {
    "dataset_id": "1", "question_id": "2", "question_text": "what feels unsafe?",
    "parents": [{"parent_id": "2_P01", "name": "safety", "description": "d"}],
    "labels": [
        {"label_id": "2_001", "parent_id": "2_P01", "name": "Theft",
         "description": "stolen things"},
        {"label_id": "2_002", "parent_id": None, "name": "Dark streets",
         "description": "lighting"},
    ],
}

ASSIGNMENTS = [
    {"response_key": "1:2:0", "label_ids": ["2_001"], "uncategorized": False},
    {"response_key": "1:2:1", "label_ids": ["2_001", "2_002"], "uncategorized": False},
    {"response_key": "1:2:2", "label_ids": [], "uncategorized": True},
    {"response_key": "1:2:3", "label_ids": ["2_999"], "uncategorized": False},
]


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "LABELS_DIR", tmp_path / "labels")
    monkeypatch.setattr(summary, "TAXONOMY_DIR", tmp_path / "taxonomy")
    monkeypatch.setattr(summary, "LEXICON_DIR", tmp_path / "lexicon")
    monkeypatch.setattr(summary, "LOCATIONS_DIR", tmp_path / "locations")
    return tmp_path


def test_counts_and_unknown_ids(data_dirs):
    _write(data_dirs / "taxonomy/1/2/2026-01-01T00-00-00Z_aaaa/candidate_taxonomy.json",
           TAXONOMY)
    run = data_dirs / "labels/1/2/2026-01-01T00-05-00Z_bbbb"
    _write(run / "assignments.json", ASSIGNMENTS)
    _write(run / "manifest.json", {"taxonomy_run": "2026-01-01T00-00-00Z_aaaa"})

    q = summary.summarize_question("1", "2")
    counts = {e["label_id"]: e["count"] for e in q["entries"]}
    assert counts == {"2_001": 2, "2_002": 1}
    assert q["n_responses"] == 4
    assert q["n_uncategorized"] == 1
    assert q["unknown_assignment_ids"] == {"2_999": 1}
    assert q["entries"][0]["parent_name"] == "safety"      # sorted by count desc
    assert q["entries"][1]["parent_name"] is None


def test_review_run_pairs_by_same_run_id(data_dirs):
    # older plain run + newer review run on BOTH sides; review must win and
    # pair with its own taxonomy, not the manifest of the older run
    _write(data_dirs / "taxonomy/1/2/2026-01-01T00-00-00Z_aaaa/candidate_taxonomy.json",
           TAXONOMY)
    review_tax = dict(TAXONOMY, labels=TAXONOMY["labels"][:1])
    _write(data_dirs / "taxonomy/1/2/2026-01-02T00-00-00Z_review/candidate_taxonomy.json",
           review_tax)
    old = data_dirs / "labels/1/2/2026-01-01T00-05-00Z_bbbb"
    _write(old / "assignments.json", ASSIGNMENTS)
    _write(old / "manifest.json", {"taxonomy_run": "2026-01-01T00-00-00Z_aaaa"})
    new = data_dirs / "labels/1/2/2026-01-02T00-00-00Z_review"
    _write(new / "assignments.json", ASSIGNMENTS[:2])
    _write(new / "manifest.json", {"tool": "scripts.review"})

    q = summary.summarize_question("1", "2")
    assert q["labels_run"] == "2026-01-02T00-00-00Z_review"
    assert q["taxonomy_run"] == "2026-01-02T00-00-00Z_review"
    assert q["taxonomy_resolved_via"] == "same_run_id"
    assert len(q["entries"]) == 1


def test_render_and_index():
    s = {
        "dataset_description": "a survey",
        "questions": [{
            "question_id": "2", "question_text": "what feels unsafe?",
            "n_responses": 4, "n_uncategorized": 1,
            "entries": [{"label_id": "2_001", "name": "Theft",
                         "parent_name": "safety", "description": "stolen things",
                         "count": 2}],
        }],
        "lexicon_concepts": [{"name": "public transit", "n_terms": 3}],
    }
    text = summary.render_summary(s)
    assert '2_001 | safety > Theft — stolen things (n=2)' in text
    assert "public transit" in text
    assert summary.valid_label_ids(s) == {"2_001"}
    assert summary.label_index(s)["2_001"]["question_id"] == "2"


def test_render_location_block():
    s = {
        "dataset_description": "",
        "questions": [],
        "lexicon_concepts": [],
        "location_concepts": [
            {"name": "downtown", "kind": "named", "n_spans": 3, "count": 61},
            {"name": "streets", "kind": "type", "n_spans": 4, "count": 94},
        ],
        "location_coverage": {
            "2": {"responses": 222, "responses_with_location": 75}},
    }
    text = summary.render_summary(s)
    assert "named places: downtown (n=61)" in text
    assert "place types: streets (n=94)" in text
    assert "q2: 75/222" in text
    assert "denominator" in text


def test_build_summary_reads_location_artifact(data_dirs):
    _write(data_dirs / "taxonomy/1/2/2026-01-01T00-00-00Z_aaaa/candidate_taxonomy.json",
           TAXONOMY)
    run = data_dirs / "labels/1/2/2026-01-01T00-05-00Z_bbbb"
    _write(run / "assignments.json", ASSIGNMENTS)
    _write(run / "manifest.json", {"taxonomy_run": "2026-01-01T00-00-00Z_aaaa"})
    _write(data_dirs / "locations/1/locations.json", {"concepts": [
        {"name": "downtown", "kind": "named", "spans": ["downtown"]}]})
    _write(data_dirs / "locations/1/manifest.json", {
        "concept_counts": {"downtown": 14},
        "labeled_location_coverage": {
            "2": {"responses": 4, "responses_with_location": 2}}})

    s = summary.build_summary("1")
    assert s["location_concepts"] == [
        {"name": "downtown", "kind": "named", "n_spans": 1, "count": 14}]
    assert s["location_coverage"]["2"]["responses_with_location"] == 2
    assert "downtown (n=14)" in summary.render_summary(s)
