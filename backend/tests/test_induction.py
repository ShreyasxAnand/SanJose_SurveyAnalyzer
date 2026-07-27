"""Offline end-to-end test of taxonomy induction with a fake model client.

Proves: chunking, evidence resolution + invalid-citation guard, exact-name
merge, theme sorting, per-theme dedup and cross-theme dedup with their bad-input
guards, chunk-failure resilience, artifact writing — all without an API key.
"""
import json

import pytest

from app import induction
from app.induction import ResponseRow
from app.llm import Usage

MAP_CHUNK_A = json.dumps({
    "categories": [
        {"name": "Poor street lighting", "description": "Respondent cites dark or unlit streets.",
         "include": ["mentions dark streets"], "exclude": ["general night fear without lighting mention"],
         "evidence": [1, 2]},
        {"name": "Car break-ins", "description": "Vehicles broken into or windows smashed.",
         "include": ["broken car windows"], "exclude": ["car theft (whole vehicle)"],
         "evidence": [3, 99]},  # 99 is an invalid citation -> must be dropped & counted
        {"name": "Aggressive panhandling", "description": "Being aggressively approached for money.",
         "include": ["asked for money aggressively"], "exclude": [],
         "evidence": [4]},
    ]
})

MAP_CHUNK_B = json.dumps({
    "categories": [
        {"name": "poor street lighting!", "description": "Dark streets at night.",
         "include": ["unlit areas"], "exclude": [], "evidence": [1]},  # exact-name merge with A
        {"name": "Vehicle break-ins", "description": "Cars getting broken into.",
         "include": ["smashed windows"], "exclude": [], "evidence": [2]},
        {"name": "Loose dogs", "description": "Unleashed or stray dogs.",
         "include": ["stray dogs"], "exclude": [], "evidence": [5]},
    ]
})

# Step 1: sort every candidate into a theme. c01_02 ("Loose dogs") is
# deliberately left out, to prove an unsorted candidate is not lost.
ASSIGN_OUT = json.dumps({
    "parents": [
        {"name": "Neighborhood crime", "description": "Crime against people and property.",
         "members": ["c00_01", "c01_01", "c00_02"]},
        {"name": "Street conditions", "description": "Physical conditions of the street.",
         "members": ["c00_00", "zzz_bogus"]},
    ]
})

# Step 2: one dedup call per theme, keyed by the theme name in the prompt.
DEDUP_OUT = {
    "Neighborhood crime": json.dumps({
        "groups": [
            {"ids": ["c00_01", "c01_01"], "name": "Vehicle break-ins"},
            {"ids": ["c00_02"], "name": "Aggressive panhandling"},
        ],
        "too_broad": [],
    }),
    # single-member theme after the bogus id is dropped -> never reaches a call
}

# Step 3: nothing duplicated across themes here — the common, correct answer.
CROSS_OUT = json.dumps({"groups": []})


class FakeClient:
    model_id = "fake-model"

    def __init__(self):
        self.usage = Usage()
        self._map_outputs = [MAP_CHUNK_A, MAP_CHUNK_B]

    def complete(self, system: str, user: str) -> str:
        self.usage.add(100, 50)
        # route on each prompt's required JSON shape rather than prose wording,
        # so rewording a prompt can't silently turn it into a MAP call
        if '{"parents":' in system:
            return ASSIGN_OUT
        if "across DIFFERENT themes" in system:
            return CROSS_OUT
        if '{"groups":' in system:
            theme = system.split("Theme: ", 1)[1].split("\n", 1)[0].strip()
            return DEDUP_OUT[theme]
        return self._map_outputs.pop(0)


def _rows(n=12):
    return [ResponseRow(response_key=f"ds:q4oe:{i}", text=f"response text {i}") for i in range(n)]


def test_chunks_are_disjoint_and_exhaustive():
    rows = _rows(13)
    chunks = induction.make_chunks(rows, chunk_size=6, seed=7)
    keys = [r.response_key for c in chunks for r in c]
    assert len(keys) == 13 and len(set(keys)) == 13  # no sampling, no dupes
    assert len(chunks) == 3


def test_sentinel_filter_keeps_meaningful_negatives():
    assert induction._is_sentinel("n/a")
    assert induction._is_sentinel("  . ")
    assert induction._is_sentinel("IDK!")
    assert not induction._is_sentinel("None")      # meaningful for "what makes you unsafe"
    assert not induction._is_sentinel("nothing.")  # ditto
    assert not induction._is_sentinel("nothing makes me feel unsafe")


def test_full_pipeline_offline(tmp_path):
    rows = _rows(12)
    meta = {"dataset_id": "ds", "question_id": "q4oe",
            "question_text": "What makes you feel unsafe in San Jose?",
            "rows_for_question": 12, "rows_empty": 0,
            "rows_sentinel_filtered": 0, "rows_used": 12}
    client = FakeClient()
    taxonomy, report = induction.run_induction(rows, meta, client, chunk_size=6, seed=7)

    # 6 candidates -> exact-name merge (lighting) -> 5 -> dedup (break-ins) -> 4
    assert report["candidates_proposed"] == 6
    assert report["labels_after_exact_merge"] == 5
    assert report["labels_final"] == 4
    assert report["invalid_evidence_citations"] == 1

    by_name = {l["name"]: l for l in taxonomy["labels"]}
    lighting = by_name["Poor street lighting"]
    assert lighting["chunk_support"] == 2
    assert len(lighting["provenance"]["merged_from"]) == 2

    breakins = by_name["Vehicle break-ins"]
    assert breakins["chunk_support"] == 2
    mechanisms = {m["mechanism"] for m in report["merge_log"]}
    assert mechanisms == {"exact_name", "llm_dedup"}
    dedup_entries = [m for m in report["merge_log"] if m["mechanism"] == "llm_dedup"]
    assert len(dedup_entries) == 1 and dedup_entries[0]["rationale"]
    assert set(dedup_entries[0]["member_ids"]) == {"c00_01", "c01_01"}

    # an id the sort step invented is ignored, not sorted
    assert any("zzz_bogus" in w for w in report["merge_warnings"])
    assert "Aggressive panhandling" in by_name

    # singletons flagged, never dropped
    assert by_name["Aggressive panhandling"]["singleton"]
    assert by_name["Loose dogs"]["singleton"]

    # --- two-level tree ---
    assert [p["name"] for p in taxonomy["parents"]] == ["Neighborhood crime", "Street conditions"]
    crime, street = taxonomy["parents"]
    assert breakins["parent_id"] == crime["parent_id"]
    assert by_name["Aggressive panhandling"]["parent_id"] == crime["parent_id"]
    assert breakins["label_id"] in crime["child_label_ids"]
    # a theme with a single member keeps its parent rather than discarding
    # the sort step's answer
    assert lighting["parent_id"] == street["parent_id"]

    # a candidate the sort step forgot stays top-level and is flagged
    assert by_name["Loose dogs"]["parent_id"] is None
    assert by_name["Loose dogs"]["needs_review"]
    assert report["labels_unparented"] == ["Loose dogs"]
    assert any("not sorted into any theme" in w for w in report["merge_warnings"])
    assert report["parents_final"] == 2

    # every child_label_id points at a real label; no label claims a ghost parent
    label_ids = {l["label_id"] for l in taxonomy["labels"]}
    parent_ids = {p["parent_id"] for p in taxonomy["parents"]}
    for p in taxonomy["parents"]:
        assert set(p["child_label_ids"]) <= label_ids
    for l in taxonomy["labels"]:
        assert l["parent_id"] is None or l["parent_id"] in parent_ids

    # every example resolves to a real source row (no hallucinated quotes)
    real_keys = {r.response_key for r in rows}
    for label in taxonomy["labels"]:
        for ex in label["examples"]:
            assert ex["response_key"] in real_keys

    manifest = {"run_id": "testrun_0000", "run_report": report}
    out_dir = induction.write_artifacts(taxonomy, manifest, out_root=tmp_path)
    tax_file = out_dir / "candidate_taxonomy.json"
    assert tax_file.exists() and (out_dir / "manifest.json").exists()
    reloaded = json.loads(tax_file.read_text(encoding="utf-8"))
    assert reloaded["status"] == "candidate_for_review"
    assert "NOT corpus prevalence" in reloaded["labels"][0]["chunk_support_note"]


def _plabel(pid, name, chunk=0):
    cand = induction.Candidate(cid=pid, chunk_index=chunk, name=name, description=name,
                               include=[], exclude=[], evidence=[])
    return induction.ProvisionalLabel(pid=pid, name=name, description=name,
                                      include=[], exclude=[], members=[cand])


def test_absorbed_broad_label_is_dropped_but_never_lost():
    """The one case where a label does not survive: a candidate that only
    restates its theme. Its provenance must land on the parent."""
    labels = [
        _plabel("c00_00", "Vehicle Crime and Theft"),          # the broad one
        _plabel("c00_01", "Car break-ins"),
        _plabel("c01_00", "Catalytic converter theft", chunk=1),
    ]
    raw = json.dumps({
        "groups": [{"ids": ["c00_01"], "name": "Car break-ins"},
                   {"ids": ["c01_00"], "name": "Catalytic converter theft"}],
        "too_broad": ["c00_00"],
    })
    kept, absorbed, merge_log, warnings = induction.apply_dedup(raw, labels, "Vehicle Crime")

    assert [l.name for l in kept] == ["Car break-ins", "Catalytic converter theft"]
    assert [c.name for c in absorbed] == ["Vehicle Crime and Theft"]
    entry = next(m for m in merge_log if m["mechanism"] == "absorbed_into_parent")
    assert entry["member_names"] == ["Vehicle Crime and Theft"]
    assert entry["result_name"] == "Vehicle Crime"
    assert not warnings


def test_dedup_combines_members_without_losing_include_or_exclude():
    a, b = _plabel("c00_00", "Car break-ins"), _plabel("c01_00", "Auto burglary", chunk=1)
    a.include, a.exclude = ["smashed windows"], ["whole vehicle stolen"]
    b.include, b.exclude = ["broken into car"], ["whole vehicle stolen"]   # dupe exclude
    raw = json.dumps({"groups": [{"ids": ["c00_00", "c01_00"], "name": "Vehicle break-ins"}]})
    kept, _, merge_log, warnings = induction.apply_dedup(raw, [a, b], "Property crime")

    assert len(kept) == 1
    assert kept[0].name == "Vehicle break-ins"
    assert kept[0].include == ["smashed windows", "broken into car"]
    assert kept[0].exclude == ["whole vehicle stolen"]         # union, deduped
    assert len(kept[0].members) == 2                            # both chunks' provenance
    assert merge_log[0]["mechanism"] == "llm_dedup"
    assert not warnings


def test_id_the_model_forgot_survives_as_its_own_label():
    labels = [_plabel(f"c00_0{i}", f"L{i}") for i in range(3)]
    raw = json.dumps({"groups": [{"ids": ["c00_00", "c00_01"], "name": "Merged"}]})
    kept, _, _, warnings = induction.apply_dedup(raw, labels, "Theme")
    assert [l.name for l in kept] == ["Merged", "L2"]           # L2 not dropped
    assert any("not mentioned, kept as-is" in w for w in warnings)


def test_id_claimed_by_two_dedup_groups_goes_to_the_first():
    labels = [_plabel(f"c00_0{i}", f"L{i}") for i in range(4)]
    raw = json.dumps({"groups": [
        {"ids": ["c00_00", "c00_01"], "name": "First"},
        {"ids": ["c00_01", "c00_02", "c00_03"], "name": "Second"},
    ]})
    kept, _, _, warnings = induction.apply_dedup(raw, labels, "Theme")
    assert [sorted(m.cid for m in l.members) for l in kept] == [
        ["c00_00", "c00_01"], ["c00_02", "c00_03"]
    ]
    assert any("already grouped" in w for w in warnings)


def test_theme_with_everything_marked_too_broad_keeps_its_members():
    """Absorbing every member would silently delete the theme's contents."""
    labels = [_plabel("c00_00", "A"), _plabel("c00_01", "B")]
    raw = json.dumps({"groups": [], "too_broad": ["c00_00", "c00_01"]})
    kept, absorbed, merge_log, warnings = induction.apply_dedup(raw, labels, "Theme")
    assert [l.name for l in kept] == ["A", "B"]
    assert absorbed == [] and merge_log == []
    assert any("absorption skipped" in w for w in warnings)


def _parent(name, pids):
    return induction.ProvisionalParent(name=name, description="d", child_pids=list(pids),
                                       rationale="r")


def test_cross_theme_merge_moves_the_child_off_the_losing_parent():
    labels = [_plabel("c00_00", "Soft on crime policies"),
              _plabel("c01_00", "Lenient laws and lack of prosecution", chunk=1),
              _plabel("c02_00", "Gang activity", chunk=2)]
    parents = [_parent("policing and justice", ["c00_00", "c02_00"]),
               _parent("property crime", ["c01_00"])]
    theme_of = {"c00_00": "policing and justice", "c02_00": "policing and justice",
                "c01_00": "property crime"}
    raw = json.dumps({"groups": [
        {"ids": ["c00_00", "c01_00"], "name": "Lenient laws and lack of prosecution"}
    ]})
    out, log, warnings = induction.apply_cross_merges(raw, labels, parents, theme_of)

    assert [l.name for l in out] == ["Lenient laws and lack of prosecution", "Gang activity"]
    assert len(out[0].members) == 2                       # provenance from both themes
    assert parents[0].child_pids == ["c00_00", "c02_00"]  # winner keeps its parent
    assert parents[1].child_pids == []                    # loser's parent drops the child
    assert log[0]["mechanism"] == "llm_cross_theme"
    assert log[0]["member_themes"] == ["policing and justice", "property crime"]
    assert not warnings


def test_cross_theme_will_not_relitigate_a_single_theme():
    """Same-theme pairs were already judged with the full theme in view."""
    labels = [_plabel("c00_00", "A"), _plabel("c00_01", "B")]
    parents = [_parent("one theme", ["c00_00", "c00_01"])]
    theme_of = {"c00_00": "one theme", "c00_01": "one theme"}
    raw = json.dumps({"groups": [{"ids": ["c00_00", "c00_01"], "name": "Merged"}]})
    out, log, warnings = induction.apply_cross_merges(raw, labels, parents, theme_of)
    assert [l.name for l in out] == ["A", "B"]            # untouched
    assert log == []
    assert any("all in one theme, rejected" in w for w in warnings)


def test_cross_theme_empty_result_is_a_no_op():
    labels = [_plabel("c00_00", "A"), _plabel("c01_00", "B", chunk=1)]
    parents = [_parent("x", ["c00_00"]), _parent("y", ["c01_00"])]
    out, log, warnings = induction.apply_cross_merges(
        json.dumps({"groups": []}), labels, parents, {"c00_00": "x", "c01_00": "y"})
    assert out is labels and log == [] and warnings == []
    assert parents[0].child_pids == ["c00_00"] and parents[1].child_pids == ["c01_00"]


def test_sort_step_never_loses_a_candidate():
    labels = [_plabel(f"c00_0{i}", f"L{i}") for i in range(4)]
    raw = json.dumps({"parents": [
        {"name": "Theme A", "description": "d", "members": ["c00_00", "ghost"]},
        {"name": "Theme B", "description": "d", "members": ["c00_01", "c00_00"]},
    ]})
    groups, warnings = induction.parse_assignment(raw, labels)
    assert groups[0] == ("Theme A", "d", ["c00_00"])
    assert groups[1] == ("Theme B", "d", ["c00_01"])        # c00_00 not double-sorted
    assert groups[-1] == ("", "", ["c00_02", "c00_03"])     # unsorted bucket
    assert any("ghost" in w for w in warnings)
    assert any("already sorted elsewhere" in w for w in warnings)


def test_one_bad_chunk_does_not_kill_the_run():
    """A chunk whose output stays unparseable through the retry is skipped and
    disclosed, not fatal — and chunk_support is measured against the chunks
    that actually answered."""
    class FlakyClient(FakeClient):
        """Chunk 0 answers; chunk 1 returns junk on the first call AND the
        retry, which is the case that used to abort the whole run."""

        def __init__(self):
            super().__init__()
            self.map_calls = 0

        def complete(self, system, user):
            if "candidate coding taxonomy" in system:
                self.usage.add(100, 50)
                self.map_calls += 1
                return MAP_CHUNK_A if self.map_calls == 1 else "{ this is not json"
            return super().complete(system, user)

    taxonomy, report = induction.run_induction(
        _rows(12), {"dataset_id": "ds", "question_id": "q4oe",
                    "question_text": "q?", "rows_for_question": 12, "rows_empty": 0,
                    "rows_sentinel_filtered": 0, "rows_used": 12},
        FlakyClient(), chunk_size=6, seed=7)

    assert report["chunks"]["n_chunks"] == 2
    assert report["chunks"]["n_chunks_succeeded"] == 1
    assert [f["chunk"] for f in report["chunks"]["failed_chunks"]] == [1]
    assert report["chunks"]["failed_chunks"][0]["n_responses"] == 6
    # chunk 0's three candidates still produce a usable taxonomy
    assert report["labels_final"] == 3
    # with one surviving chunk nothing can be a "singleton" — support is 1 of 1
    assert all(l["chunk_support"] == 1 and not l["singleton"] for l in taxonomy["labels"])


def test_all_chunks_failing_raises_rather_than_writing_an_empty_taxonomy():
    class DeadClient(FakeClient):
        def complete(self, system, user):
            self.usage.add(1, 1)
            return "not json at all"

    with pytest.raises(RuntimeError, match="All 2 chunks failed"):
        induction.run_induction(
            _rows(12), {"dataset_id": "ds", "question_id": "q", "question_text": "q?",
                        "rows_for_question": 12, "rows_empty": 0,
                        "rows_sentinel_filtered": 0, "rows_used": 12},
            DeadClient(), chunk_size=6, seed=7)


def test_dataset_description_reaches_every_prompt_stage_and_versions_the_run():
    desc = "This survey is the Community Focus Area survey for San José."
    rows = _rows(3)
    labels = [_plabel("c00_00", "A"), _plabel("c00_01", "B")]

    stages = [
        induction.build_map_prompts("q?", rows, desc)[0],
        induction.build_assign_prompts("q?", labels, desc)[0],
        induction.build_dedup_prompts("q?", "theme", labels, desc)[0],
        induction.build_cross_prompts("q?", labels, {}, desc)[0],
    ]
    for system in stages:
        assert desc in system and system.startswith("Survey context:")

    # no description -> prompts identical to having none at all, so existing
    # runs stay reproducible
    assert not induction.build_map_prompts("q?", rows)[0].startswith("Survey context:")
    assert induction.context_block("   ") == ""

    # the description shapes output, so it must version the run like a prompt
    assert induction.prompt_hash(desc) != induction.prompt_hash("")
    assert induction.prompt_hash(desc) == induction.prompt_hash(desc)


def test_extract_json_tolerates_fences_and_prose():
    assert induction.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert induction.extract_json('Here you go: {"a": 1} hope that helps') == {"a": 1}
