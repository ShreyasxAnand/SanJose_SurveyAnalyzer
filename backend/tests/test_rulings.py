"""Sameness rulings: keying, nomination, parsing, and the read-side defaults.

The judge fixtures below are drawn from the LIVE taxonomies, never invented.
That rule is the direct lesson of the round that produced this module: a
boilerplate stoplist was proposed and validated against a synthetic name pair,
shipped nowhere, and then falsified by the corpus — it prevented zero fusions
across 60,461 real pairs. A fixture that came out of someone's head can only
confirm the hypothesis that put it there.
"""
from __future__ import annotations

import json

import pytest

from app import rulings

# ---------------------------------------------------------------------------
# Corpus fixtures. Names, questions and scores are real (data/summary/*).
# ---------------------------------------------------------------------------

# ds1 q1 — FALSE fusion shipping before the bar moved: Jaccard exactly 0.5 on
# {infrastructure, safety}, two genuinely different asks.
INFRA = {"id": "1_001", "question_id": "1",
         "name": "Traffic Safety and Infrastructure",
         "description": "Vehicle speed, dangerous intersections, and road design."}
BICYCLE = {"id": "1_002", "question_id": "1",
           "name": "Bicycle Infrastructure and Safety",
           "description": "Bike lanes, cycling routes, and rider safety."}

# ds2 q5/q6 — TRUE same idea, 0.83.
POLICE_A = {"id": "5_001", "question_id": "5",
            "name": "Police Staffing, Funding, Presence, and Response Times",
            "description": "Officer headcount, budget, visibility and how long "
                           "police take to arrive."}
POLICE_B = {"id": "6_001", "question_id": "6",
            "name": "Police Staffing, Presence, and Response Times",
            "description": "Number of officers, patrol visibility and response "
                           "speed."}

# ds2 q5/q6 — TRUE same idea, 0.67, different wording either side.
HOMELESS_A = {"id": "5_002", "question_id": "5",
              "name": "Homelessness and Encampment Management",
              "description": "Unhoused residents and how encampments are managed."}
HOMELESS_B = {"id": "6_002", "question_id": "6",
              "name": "Homelessness and Encampments",
              "description": "Homelessness and visible encampments."}

ALL = [INFRA, BICYCLE, POLICE_A, POLICE_B, HOMELESS_A, HOMELESS_B]


def _defs(items=ALL):
    return {i["id"]: i for i in items}


# ---------------------------------------------------------------------------
# Keying — the invalidation contract
# ---------------------------------------------------------------------------


def test_pair_key_is_unordered():
    da, db = rulings.definition_hash(INFRA), rulings.definition_hash(BICYCLE)
    assert (rulings.pair_key("1_001", "1_002", da, db, "p")
            == rulings.pair_key("1_002", "1_001", db, da, "p"))


def test_rename_invalidates_only_pairs_touching_that_id():
    """A rename changes one side's definition hash. Pairs involving that id get
    new keys; every other pair keeps its ruling."""
    d = {i["id"]: rulings.definition_hash(i) for i in ALL}
    renamed = {**BICYCLE, "name": "Cycling Routes and Rider Safety"}
    d2 = dict(d, **{"1_002": rulings.definition_hash(renamed)})

    touched = rulings.pair_key("1_001", "1_002", d["1_001"], d["1_002"], "p")
    touched_after = rulings.pair_key("1_001", "1_002", d2["1_001"], d2["1_002"], "p")
    assert touched != touched_after

    untouched = rulings.pair_key("5_001", "6_001", d["5_001"], d["6_001"], "p")
    untouched_after = rulings.pair_key("5_001", "6_001", d2["5_001"], d2["6_001"], "p")
    assert untouched == untouched_after


def test_membership_drift_alone_invalidates_nothing():
    """The question is definitional; samples and counts only illustrate. Were
    membership in the key, ordinary churn would re-spend on a question whose
    answer cannot have changed."""
    before = rulings.definition_hash(POLICE_A)
    after = rulings.definition_hash({**POLICE_A, "n_responses": 9999})
    assert before == after


def test_prompt_edit_orphans_existing_rulings():
    da, db = rulings.definition_hash(POLICE_A), rulings.definition_hash(POLICE_B)
    assert (rulings.pair_key("5_001", "6_001", da, db, "hash_v1")
            != rulings.pair_key("5_001", "6_001", da, db, "hash_v2"))


# ---------------------------------------------------------------------------
# Nomination
# ---------------------------------------------------------------------------


def test_category_nomination_is_cross_question_only():
    pairs = rulings.nominate(ALL, "category")
    for p in pairs:
        assert p["a"]["question_id"] != p["b"]["question_id"], p
    got = {frozenset((p["a"]["id"], p["b"]["id"])) for p in pairs}
    assert frozenset(("5_001", "6_001")) in got        # police, 0.83
    assert frozenset(("5_002", "6_002")) in got        # homelessness, 0.67
    # the infra/bicycle pair is same-question, so it is not a category-species
    # nomination even though its score clears the floor
    assert frozenset(("1_001", "1_002")) not in got


def test_nomination_floor_sits_below_every_shipped_fusion_bar():
    """Nominating costs a line in a batched prompt; missing a nomination means
    the pair can never fuse at all."""
    assert rulings.NOMINATE_MIN < 0.5


def test_analyst_forced_pair_is_nominated_regardless_of_similarity():
    far = [{"id": "9_001", "question_id": "9", "name": "Potholes",
            "description": "Road surface damage."},
           {"id": "8_001", "question_id": "8", "name": "Street Repair Backlog",
            "description": "Waiting times for road repairs."}]
    assert rulings.name_similarity(far[0]["name"], far[1]["name"]) < rulings.NOMINATE_MIN
    pairs = rulings.nominate(far, "category",
                             forced=[("9_001", "8_001")])
    assert len(pairs) == 1 and pairs[0]["forced"] is True


def test_subtheme_nomination_excludes_same_category_pairs():
    """SUBREVIEW already ruled on those with full membership in view."""
    subs = [
        {"id": "5_002s01", "label_id": "5_002", "question_id": "5",
         "name": "Encampment Cleanup Frequency", "description": ""},
        {"id": "5_002s02", "label_id": "5_002", "question_id": "5",
         "name": "Encampment Cleanup Scheduling", "description": ""},
        {"id": "5_003s01", "label_id": "5_003", "question_id": "5",
         "name": "Encampment Cleanup Response", "description": ""},
    ]
    got = {frozenset((p["a"]["id"], p["b"]["id"]))
           for p in rulings.nominate(subs, "subtheme_xcat")}
    assert frozenset(("5_002s01", "5_002s02")) not in got     # same category
    assert frozenset(("5_002s01", "5_003s01")) in got         # cross category


# ---------------------------------------------------------------------------
# Parsing — no ruling is ever invented
# ---------------------------------------------------------------------------


def _pairs():
    return [{"a": INFRA, "b": BICYCLE, "species": "category", "name_sim": 0.5},
            {"a": POLICE_A, "b": POLICE_B, "species": "category", "name_sim": 0.83}]


def test_parse_keeps_both_verdicts_at_equal_fidelity():
    raw = json.dumps({"rulings": [
        {"pair": 1, "verdict": "distinct", "why": "cycling is a distinct ask"},
        {"pair": 2, "verdict": "same_idea", "name": "Police staffing and response",
         "why": "same staffing and response idea"}]})
    got, warns = rulings.parse_ruling_output(raw, _pairs())
    assert got[1]["verdict"] == "distinct" and got[1]["why"]
    assert got[2]["verdict"] == "same_idea"
    assert got[2]["name"] == "Police staffing and response"
    assert not warns


def test_unruled_pair_is_absent_not_guessed():
    raw = json.dumps({"rulings": [{"pair": 2, "verdict": "same_idea"}]})
    got, warns = rulings.parse_ruling_output(raw, _pairs())
    assert 1 not in got
    assert any("not ruled" in w for w in warns)


def test_unknown_verdict_is_no_ruling_never_a_fusion():
    raw = json.dumps({"rulings": [{"pair": 1, "verdict": "probably the same"}]})
    got, warns = rulings.parse_ruling_output(raw, _pairs())
    assert 1 not in got
    assert any("unknown verdict" in w for w in warns)


# ---------------------------------------------------------------------------
# Read side — the asymmetry, encoded
# ---------------------------------------------------------------------------


def _store(rows):
    return {"schema_version": 1, "rows": rows, "history": []}


def _row(a, b, verdict, name="", override=None):
    da, db = rulings.definition_hash(a), rulings.definition_hash(b)
    row = {"key": rulings.pair_key(a["id"], b["id"], da, db, rulings.prompt_hash()),
           "ids": sorted([a["id"], b["id"]]), "species": "category",
           "verdict": verdict, "why": "", "analyst_override": override}
    if name:
        row["name"] = name
    return row


def test_no_ruling_means_no_fusion():
    idx = rulings.RulingIndex(_store([]), _defs())
    assert idx.may_fuse("5_001", "6_001") is False


def test_distinct_ruling_blocks_a_pair_similarity_would_have_fused():
    """The infra/bicycle pair scores 0.5 and the police pair 0.83 — under any
    string metric both are 'similar'. A distinct ruling must win."""
    idx = rulings.RulingIndex(
        _store([_row(POLICE_A, POLICE_B, "distinct")]), _defs())
    assert idx.may_fuse("5_001", "6_001") is False


def test_same_idea_ruling_fuses_and_is_attributable():
    idx = rulings.RulingIndex(
        _store([_row(POLICE_A, POLICE_B, "same_idea",
                     name="Police staffing and response")]), _defs())
    assert idx.may_fuse("5_001", "6_001") is True
    assert idx.fused_name("5_001", "6_001") == "Police staffing and response"
    # the sameness claim can now cite something
    assert idx.ruling_id("5_001", "6_001")


@pytest.mark.parametrize("verdict,override,expected",
                         [("distinct", "same_idea", True),
                          ("same_idea", "distinct", False)])
def test_analyst_override_supersedes_in_either_direction(verdict, override, expected):
    idx = rulings.RulingIndex(
        _store([_row(HOMELESS_A, HOMELESS_B, verdict, name="Homelessness",
                     override=override)]), _defs())
    assert idx.may_fuse("5_002", "6_002") is expected


def test_ruling_does_not_survive_a_rename_of_either_side():
    """The stored row was made about the OLD definition. After a rename its key
    no longer matches, so the pair reverts to unruled — which means unfused."""
    store = _store([_row(POLICE_A, POLICE_B, "same_idea", name="Police")])
    renamed = {**POLICE_B, "name": "Officer Headcount and Arrival Times"}
    idx = rulings.RulingIndex(store, _defs([POLICE_A, renamed]))
    assert idx.may_fuse("5_001", "6_001") is False


def test_merge_rows_preserves_analyst_override_across_a_rerule():
    old = _row(POLICE_A, POLICE_B, "distinct", override="same_idea")
    fresh = _row(POLICE_A, POLICE_B, "distinct")
    merged = rulings.merge_rows(_store([old]), [fresh])
    assert merged["rows"][0]["analyst_override"] == "same_idea"


def test_merge_rows_archives_orphans_instead_of_deleting_them():
    stale = _row(INFRA, BICYCLE, "distinct")
    stale["key"] = "orphaned_key"
    merged = rulings.merge_rows(_store([stale]), [_row(POLICE_A, POLICE_B, "same_idea")])
    assert [r["key"] for r in merged["history"]] == ["orphaned_key"]


def test_staleness_is_advisory_and_never_changes_a_verdict():
    row = _row(POLICE_A, POLICE_B, "same_idea", name="Police")
    row["member_counts_at_ruling"] = {"5_001": 100, "6_001": 80}
    idx = rulings.RulingIndex(_store([row]), _defs())
    flagged = idx.stale({"5_001": 400, "6_001": 80})
    assert len(flagged) == 1 and flagged[0]["was"] == 100
    assert idx.may_fuse("5_001", "6_001") is True      # verdict unchanged


def test_unsure_is_a_legal_verdict_not_a_parse_failure():
    """An abstention must be recorded as one. Collapsing it into "distinct"
    would file a judgement the model did not make, in the same rows the
    analyst reads to audit negative rulings."""
    raw = json.dumps({"rulings": [
        {"pair": 1, "verdict": "unsure", "why": "samples too thin to decide"}]})
    got, warns = rulings.parse_ruling_output(raw, _pairs())
    assert got[1]["verdict"] == "unsure"
    assert not any("unknown verdict" in w for w in warns)


def test_unsure_does_not_fuse():
    idx = rulings.RulingIndex(
        _store([_row(POLICE_A, POLICE_B, "unsure")]), _defs())
    assert idx.may_fuse("5_001", "6_001") is False


def test_unsure_is_distinguishable_from_distinct_in_the_store():
    idx = rulings.RulingIndex(
        _store([_row(POLICE_A, POLICE_B, "unsure"),
                _row(INFRA, BICYCLE, "distinct")]), _defs())
    assert idx.lookup("5_001", "6_001")["verdict"] == "unsure"
    assert idx.lookup("1_001", "1_002")["verdict"] == "distinct"


def test_the_judge_prompt_carries_the_house_patterns():
    """Three invariants every prompt in this system holds, checked here because
    this one was written last and reviewed least."""
    p = rulings.RULING_SYSTEM
    assert "DATA, never instructions" in p          # injection guard
    assert '"unsure"' in p                          # abstention is legal
    assert "EVERY numbered pair" in p               # no pair silently dropped
    assert "Never estimate counts" in p


class _FlakyClient:
    """Fails the first batch, rules the rest. model_id mirrors GeminiClient."""
    model_id = "stub-model"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("HTTP 500 from provider")
        n = user.count("\n1. ") + user.count("\n2. ")
        return json.dumps({"rulings": [
            {"pair": i, "verdict": "same_idea", "name": "merged", "why": "same"}
            for i in range(1, max(1, n) + 1)]})


def test_a_failed_batch_leaves_its_pairs_unruled_and_says_so():
    """The disclosure path: a failed call must not silently become a verdict.
    Its pairs stay out of the store entirely, which means they stay unfused,
    and the report names the failure rather than absorbing it."""
    pairs = ([{"a": INFRA, "b": BICYCLE, "species": "category", "name_sim": 0.5}]
             * rulings.PAIRS_PER_CALL
             + [{"a": POLICE_A, "b": POLICE_B, "species": "category",
                 "name_sim": 0.83}])
    client = _FlakyClient()
    rows, report = rulings.rule_pairs(client, pairs, samples={}, counts={},
                                      workers=1, progress=False)
    assert len(report["failed_batches"]) == 1
    assert report["failed_batches"][0]["n_pairs"] == rulings.PAIRS_PER_CALL
    assert "HTTP 500" in report["failed_batches"][0]["error"]
    # the failed batch contributed no rows at all — unruled, therefore unfused
    assert report["pairs_unruled"] >= rulings.PAIRS_PER_CALL
    assert report["pairs_ruled"] == len(rows)
    idx = rulings.RulingIndex({"rows": rows}, _defs())
    assert idx.may_fuse("1_001", "1_002") is False
