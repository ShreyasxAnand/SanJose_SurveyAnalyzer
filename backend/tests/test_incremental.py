"""Incremental labeling: pool selection, overlay merge, taxonomy extension.

Fake client routes on each prompt's required JSON shape (the test_induction
convention), so a reworded prompt can't silently change routing. Relabel calls
are told apart from the first labeling pass by the extended taxonomy's _i ids
appearing in the prompt.
"""
import json

import pytest

from app import incremental, induction, labeling
from app.induction import ResponseRow
from app.llm import Usage

TAXONOMY = {
    "schema_version": 1,
    "status": "candidate_for_review",
    "dataset_id": "9",
    "question_id": "7",
    "question_text": "What would improve your parks?",
    "parents": [
        {
            "parent_id": "7_P01",
            "name": "Public space",
            "description": "Shared outdoor space.",
            "child_label_ids": ["7_001"],
        }
    ],
    "labels": [
        {
            "label_id": "7_001",
            "parent_id": "7_P01",
            "name": "Parks",
            "description": "Park improvements.",
            "include": [],
            "exclude": [],
            "examples": [],
            "chunk_support": 2,
            "singleton": False,
            "needs_review": False,
            "provenance": {},
        }
    ],
}

PRIOR = [
    {"response_key": f"9:7:{i}", "respondent_key": f"9:{i}", "label_ids": ["7_001"],
     "uncategorized": False, "fit": 3, "locations": [], "time_context": [],
     "actionability": None, "event_occurred": None}
    for i in range(3)
]

ROWS = [ResponseRow(f"9:7:{i}", f"old text {i}") for i in range(3)] + [
    ResponseRow("9:7:6", "dog park please"),
    ResponseRow("9:7:7", "street racing at night"),
    ResponseRow("9:7:8", "street racing at night"),
]

# First labeling pass over the 2 unique new texts: n1 fits 7_001, n2 doesn't.
LABEL_OUT = json.dumps({"responses": [
    {"n": 1, "l": ["7_001"], "f": 3},
    {"n": 2, "l": [], "f": 1, "uncategorized": True},
]})

# Pool MAP: one genuinely new proposal + one that name-matches an existing
# label ("parks!" normalizes to "parks") and must be dropped.
MAP_OUT = json.dumps({"categories": [
    {"name": "parks!", "description": "dupe of an existing label",
     "include": [], "exclude": [], "evidence": [1]},
    {"name": "Street racing", "description": "Cars racing in or near parks.",
     "include": ["racing", "sideshows"], "exclude": [], "evidence": [1, 2]},
]})

# ASSIGN attaches the surviving proposal (cid c00_01) to the existing parent.
ASSIGN_OUT = json.dumps({"assignments": [
    {"id": "c00_01", "theme": "Public space"},
]})
ASSIGN_NONE_OUT = json.dumps({"assignments": [
    {"id": "c00_01", "theme": "none"},
]})

# Relabel of the pool against the extended taxonomy (1 unique text).
RELABEL_OUT = json.dumps({"responses": [
    {"n": 1, "l": ["7_i001"], "f": 3},
]})


class FakeClient:
    model_id = "fake-model"

    def __init__(self, assign_out=ASSIGN_OUT):
        self.usage = Usage()
        self.calls: list[str] = []
        self._assign_out = assign_out

    def complete(self, system: str, user: str) -> str:
        self.usage.add(100, 50)
        if '{"assignments":' in system:
            self.calls.append("assign")
            return self._assign_out
        if "FROZEN taxonomy" in system:
            if "7_i001" in system:
                self.calls.append("relabel")
                return RELABEL_OUT
            self.calls.append("label")
            return LABEL_OUT
        self.calls.append("map")
        return MAP_OUT


def test_never_labeled_pairs_picks_only_new_keys():
    pairs = incremental.never_labeled_pairs(ROWS, PRIOR)
    assert [k for k, _ in pairs] == ["9:7:6", "9:7:7", "9:7:8"]


def test_next_incremental_seq_scans_existing_ids():
    assert incremental.next_incremental_seq(TAXONOMY) == 1
    tax = json.loads(json.dumps(TAXONOMY))
    tax["labels"].append({"label_id": "7_i004", "name": "x"})
    assert incremental.next_incremental_seq(tax) == 5


def test_extend_taxonomy_collision_raises():
    with pytest.raises(ValueError):
        incremental.extend_taxonomy(TAXONOMY, [{"label_id": "7_001", "name": "x"}])


def test_run_incremental_full_flow():
    client = FakeClient()
    new_tax, merged, report = incremental.run_incremental(
        ROWS, TAXONOMY, PRIOR, client, min_pool=2, workers=1
    )

    assert client.calls == ["label", "map", "assign", "relabel"]

    # Prior records carried over verbatim, in order.
    assert merged[:3] == PRIOR

    by_key = {a["response_key"]: a for a in merged}
    assert len(merged) == 6

    covered = by_key["9:7:6"]
    assert covered["label_ids"] == ["7_001"]
    assert covered["labeled_incrementally"] is True
    assert "relabelled_incremental" not in covered

    for k in ("9:7:7", "9:7:8"):
        pooled = by_key[k]
        assert pooled["label_ids"] == ["7_i001"]
        assert pooled["labeled_incrementally"] is True
        assert pooled["relabelled_incremental"] is True

    # Taxonomy extended, never mutated: the input dict is untouched.
    assert [l["label_id"] for l in TAXONOMY["labels"]] == ["7_001"]
    added = [l for l in new_tax["labels"] if l["label_id"] == "7_i001"]
    assert len(added) == 1
    lab = added[0]
    assert lab["name"] == "Street racing"
    assert lab["parent_id"] == "7_P01"
    assert lab["needs_review"] is True
    assert lab["chunk_support"] is None
    assert lab["provenance"]["source"] == "incremental"
    parent = next(p for p in new_tax["parents"] if p["parent_id"] == "7_P01")
    assert "7_i001" in parent["child_label_ids"]

    assert report["n_new_rows"] == 3
    assert report["n_pool"] == 2
    assert report["new_label_ids"] == ["7_i001"]
    # The name-matching proposal was dropped, and disclosed.
    assert report["pool_induction"]["dropped_name_matches_existing"] == 1


def test_assign_none_leaves_orphan_for_review():
    client = FakeClient(assign_out=ASSIGN_NONE_OUT)
    new_tax, _merged, report = incremental.run_incremental(
        ROWS, TAXONOMY, PRIOR, client, min_pool=2, workers=1
    )
    lab = next(l for l in new_tax["labels"] if l["label_id"] == "7_i001")
    assert lab["parent_id"] is None
    assert lab["needs_review"] is True
    parent = next(p for p in new_tax["parents"] if p["parent_id"] == "7_P01")
    assert "7_i001" not in parent["child_label_ids"]


def test_pool_below_floor_skips_induction_and_discloses():
    client = FakeClient()
    new_tax, merged, report = incremental.run_incremental(
        ROWS, TAXONOMY, PRIOR, client, min_pool=5, workers=1
    )
    assert client.calls == ["label"]  # no map, no assign, no relabel
    assert new_tax is TAXONOMY  # unchanged, not even copied
    assert "skipped" in report["pool_induction"]["outcome"]
    assert report["new_label_ids"] == []
    # Pool rows stay uncategorized — a correct result, not a failure.
    by_key = {a["response_key"]: a for a in merged}
    assert by_key["9:7:7"]["uncategorized"] is True


def test_no_new_rows_raises_instead_of_writing():
    rows = [ResponseRow(f"9:7:{i}", f"old text {i}") for i in range(3)]
    with pytest.raises(ValueError):
        incremental.run_incremental(rows, TAXONOMY, PRIOR, FakeClient(), workers=1)


def test_malformed_map_output_is_contained():
    # The live 2026-08-03 failure: the pool-induction MAP call returned
    # malformed JSON and the whole run crashed AFTER the paid labeling
    # succeeded. Contained now, mirroring induction's failed-chunk policy:
    # the labeling work survives, the pool stays uncategorized, disclosed.
    class BadMapClient(FakeClient):
        def complete(self, system: str, user: str) -> str:
            if '{"assignments":' not in system and "FROZEN taxonomy" not in system:
                self.usage.add(100, 50)
                self.calls.append("map")
                return '{"categories": [{"name": "truncated"'  # malformed
            return super().complete(system, user)

    client = BadMapClient()
    new_tax, merged, report = incremental.run_incremental(
        ROWS, TAXONOMY, PRIOR, client, min_pool=2, workers=1
    )
    assert client.calls == ["label", "map"]  # no assign, no relabel
    assert new_tax is TAXONOMY  # unchanged
    assert report["new_label_ids"] == []
    assert "pool induction failed" in report["pool_induction"]["outcome"]
    # The paid labeling work is kept: prior + all 3 fresh rows.
    assert len(merged) == 6
    by_key = {a["response_key"]: a for a in merged}
    assert by_key["9:7:6"]["label_ids"] == ["7_001"]
    assert by_key["9:7:7"]["uncategorized"] is True


def test_malformed_assign_output_is_contained():
    # A malformed ASSIGN response must not discard MAP's successful proposals:
    # they are kept as orphans (parent_id None) for review, with a warning.
    client = FakeClient(assign_out='{"assignments": [{"id":')  # malformed
    new_tax, merged, report = incremental.run_incremental(
        ROWS, TAXONOMY, PRIOR, client, min_pool=2, workers=1
    )
    assert client.calls == ["label", "map", "assign", "relabel"]
    lab = next(l for l in new_tax["labels"] if l["label_id"] == "7_i001")
    assert lab["parent_id"] is None
    assert lab["needs_review"] is True
    warnings = report["pool_induction"]["assign_warnings"]
    assert any("ASSIGN failed" in w for w in warnings)
    # The relabel still ran against the extended taxonomy.
    by_key = {a["response_key"]: a for a in merged}
    assert by_key["9:7:7"]["label_ids"] == ["7_i001"]


def test_failed_pool_rows_do_not_enter_induction():
    # A row the model never returned is uncategorized+not_returned; it must
    # not seed new labels (its text was never actually judged uncoverable).
    class OmitClient(FakeClient):
        def complete(self, system: str, user: str) -> str:
            self.usage.add(100, 50)
            if "FROZEN taxonomy" in system:
                self.calls.append("label")
                return json.dumps({"responses": [{"n": 1, "l": ["7_001"], "f": 3}]})
            raise AssertionError("pool induction should not run")

    client = OmitClient()
    _tax, merged, report = incremental.run_incremental(
        ROWS, TAXONOMY, PRIOR, client, min_pool=1, workers=1
    )
    assert report["n_pool"] == 0
    assert "skipped" in report["pool_induction"]["outcome"]
