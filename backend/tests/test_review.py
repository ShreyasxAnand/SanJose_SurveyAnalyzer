"""Review layer: deterministic diagnostics + validated edit application."""
import pytest

from app import review


def _tax():
    return {
        "dataset_id": "1", "question_id": "9", "question_text": "test q",
        "status": "candidate",
        "parents": [
            {"parent_id": "P1", "name": "crime", "description": "",
             "rationale": "", "child_label_ids": ["L1", "L2", "L3"], "absorbed": []},
        ],
        "labels": [
            {"label_id": "L1", "parent_id": "P1", "name": "Car Break-ins",
             "description": "d1"},
            {"label_id": "L2", "parent_id": "P1", "name": "Carjackings",
             "description": "d2"},
            {"label_id": "L3", "parent_id": "P1", "name": "car break-ins",
             "description": "dupe name"},
            {"label_id": "L4", "parent_id": None, "name": "Drug Use",
             "description": "orphan"},
            {"label_id": "L5", "parent_id": None, "name": "Ghost",
             "description": "never assigned"},
        ],
    }


def _asg():
    # L1 has 5 members; L2 has 4, all shared with L1 -> overlap 1.0
    # L4 has 4 disjoint members; L3/L5 empty; two uncovered responses
    out = []
    for i in range(5):
        out.append({"response_key": f"k{i}", "label_ids": ["L1"] + (["L2"] if i < 4 else []),
                    "uncategorized": False, "fit": 3, "sentiment": "negative"})
    for i in range(5, 9):
        out.append({"response_key": f"k{i}", "label_ids": ["L4"],
                    "uncategorized": False, "fit": 2, "sentiment": "negative"})
    out.append({"response_key": "k9", "label_ids": [], "uncategorized": True,
                "fit": None, "sentiment": None})
    out.append({"response_key": "k10", "label_ids": ["L1"], "uncategorized": False,
                "fit": 1, "sentiment": "negative"})
    return out


# ---------------------------------------------------------------- diagnostics

def test_diagnose_finds_overlap_pair():
    rep = review.diagnose(_tax(), _asg())
    pairs = {(c["label_a"], c["label_b"]) for c in rep["merge_candidates"]}
    assert ("L1", "L2") in pairs or ("L2", "L1") in pairs
    top = rep["merge_candidates"][0]
    assert top["overlap"] == 1.0 and top["shared"] == 4 and top["same_parent"]


def test_diagnose_ignores_small_labels():
    asg = _asg()
    # L4 members drop below MIN_MEMBERS -> no L4 pairs possible
    asg = [a for a in asg if a["label_ids"] != ["L4"]][: -1] + asg[-2:]
    rep = review.diagnose(_tax(), asg)
    assert all("L4" not in (c["label_a"], c["label_b"])
               for c in rep["merge_candidates"])


def test_diagnose_duplicate_names_case_insensitive():
    rep = review.diagnose(_tax(), _asg())
    assert rep["duplicate_names"] == [
        {"name": "car break-ins", "label_ids": ["L1", "L3"]}]


def test_diagnose_orphans_and_zero_counts():
    rep = review.diagnose(_tax(), _asg())
    assert [o["label_id"] for o in rep["orphan_labels"]] == ["L4", "L5"]
    assert {z["label_id"] for z in rep["zero_count_labels"]} == {"L3", "L5"}


def test_diagnose_pool_is_uncategorized_plus_fit1():
    rep = review.diagnose(_tax(), _asg())
    assert set(rep["missing_category_pool"]) == {"k9", "k10"}


def test_suggest_edits_covers_mechanical_cases():
    rep = review.diagnose(_tax(), _asg())
    ops = review.suggest_edits(rep)
    kinds = {(o["op"], o.get("from") or o.get("label_id")) for o in ops}
    assert ("merge", "L3") in kinds          # duplicate name
    assert ("merge", "L2") in kinds          # 100% same-parent overlap, smaller
    assert ("delete", "L5") in kinds         # zero count
    assert all(o["op"] in review.VALID_OPS for o in ops)


def test_render_report_contains_evidence():
    rep = review.diagnose(_tax(), _asg())
    md = review.render_report(_tax(), rep, {"k9": "Homeless", "k10": "thieves"})
    assert "Car Break-ins" in md and "Homeless" in md and "Drug Use" in md


# ---------------------------------------------------------------- apply_edits

def test_merge_rewrites_ids_dedupes_and_updates_tree():
    tax, asg, log = review.apply_edits(
        _tax(), _asg(), [{"op": "merge", "from": "L2", "into": "L1"}])
    ids = {l["label_id"] for l in tax["labels"]}
    assert "L2" not in ids
    assert "L2" not in tax["parents"][0]["child_label_ids"]
    for a in asg:
        assert "L2" not in a["label_ids"]
        assert len(a["label_ids"]) == len(set(a["label_ids"]))
    survivor = next(l for l in tax["labels"] if l["label_id"] == "L1")
    assert survivor["merged_in_review"][0]["label_id"] == "L2"
    assert any("merge" in line for line in log)


def test_merge_chain_resolves_transitively():
    tax, asg, _ = review.apply_edits(_tax(), _asg(), [
        {"op": "merge", "from": "L2", "into": "L3"},
        {"op": "merge", "from": "L3", "into": "L1"},
    ])
    assert {l["label_id"] for l in tax["labels"]} == {"L1", "L4", "L5"}
    assert all("L2" not in a["label_ids"] and "L3" not in a["label_ids"]
               for a in asg)


def test_merge_unknown_or_self_raises():
    with pytest.raises(ValueError):
        review.apply_edits(_tax(), _asg(), [{"op": "merge", "from": "NOPE", "into": "L1"}])
    with pytest.raises(ValueError):
        review.apply_edits(_tax(), _asg(), [{"op": "merge", "from": "L1", "into": "L1"}])


def test_delete_empties_become_uncategorized():
    tax, asg, _ = review.apply_edits(
        _tax(), _asg(), [{"op": "delete", "label_id": "L4"}])
    assert all(l["label_id"] != "L4" for l in tax["labels"])
    for a in asg:
        if a["response_key"] in {"k5", "k6", "k7", "k8"}:
            assert a["label_ids"] == [] and a["uncategorized"]


def test_add_parent_add_label_reparent_chain():
    tax, asg, _ = review.apply_edits(_tax(), _asg(), [
        {"op": "add_parent", "parent_id": "P2", "name": "homelessness",
         "description": "presence of homelessness"},
        {"op": "add_label", "label_id": "L9", "parent_id": "P2",
         "name": "Homeless Presence", "description": "d"},
        {"op": "reparent", "label_id": "L4", "parent_id": "P2"},
    ])
    p2 = next(p for p in tax["parents"] if p["parent_id"] == "P2")
    assert set(p2["child_label_ids"]) == {"L9", "L4"}
    l4 = next(l for l in tax["labels"] if l["label_id"] == "L4")
    assert l4["parent_id"] == "P2"
    l9 = next(l for l in tax["labels"] if l["label_id"] == "L9")
    assert l9["provenance"]["source"] == "review_edit"


def test_add_label_existing_id_or_unknown_parent_raises():
    with pytest.raises(ValueError):
        review.apply_edits(_tax(), _asg(), [
            {"op": "add_label", "label_id": "L1", "name": "x"}])
    with pytest.raises(ValueError):
        review.apply_edits(_tax(), _asg(), [
            {"op": "add_label", "label_id": "L9", "parent_id": "NOPE", "name": "x"}])


def test_rename_label_and_parent():
    tax, _, _ = review.apply_edits(_tax(), _asg(), [
        {"op": "rename", "label_id": "L1", "name": "Vehicle Crime",
         "description": "break-ins, theft, vandalism"},
        {"op": "rename", "parent_id": "P1", "name": "property crime"},
    ])
    assert next(l for l in tax["labels"] if l["label_id"] == "L1")["name"] == "Vehicle Crime"
    assert tax["parents"][0]["name"] == "property crime"


def test_inputs_not_mutated_and_status_set():
    tax_in, asg_in = _tax(), _asg()
    tax, asg, _ = review.apply_edits(
        tax_in, asg_in, [{"op": "merge", "from": "L2", "into": "L1"}])
    assert any(l["label_id"] == "L2" for l in tax_in["labels"])
    assert any("L2" in a["label_ids"] for a in asg_in)
    assert tax["status"] == "reviewed"
    assert tax_in["status"] == "candidate"


def test_unknown_op_raises_and_new_label_ids():
    with pytest.raises(ValueError):
        review.apply_edits(_tax(), _asg(), [{"op": "explode"}])
    edits = [{"op": "add_label", "label_id": "L9", "name": "x"},
             {"op": "merge", "from": "L2", "into": "L1"}]
    assert review.new_label_ids(edits) == ["L9"]
