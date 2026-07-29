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

# Step 1a: the vocabulary call fixes the theme list (themes only, no members).
VOCAB_OUT = json.dumps({
    "themes": [
        {"name": "Neighborhood crime", "description": "Crime against people and property."},
        {"name": "Street conditions", "description": "Physical conditions of the street."},
    ]
})

# Step 1b: assign batches classify each id against the frozen vocabulary.
# c01_02 ("Loose dogs") answers "none", to prove an unsorted candidate is not
# lost; "neighborhood crime" (lowercase) exercises canonicalization; the
# bogus id exercises the unknown-id guard.
ASSIGN_BATCH_OUT = json.dumps({
    "assignments": [
        {"id": "c00_01", "theme": "Neighborhood crime"},
        {"id": "c01_01", "theme": "neighborhood crime"},
        {"id": "c00_02", "theme": "Neighborhood crime"},
        {"id": "c00_00", "theme": "Street conditions"},
        {"id": "zzz_bogus", "theme": "Street conditions"},
        {"id": "c01_02", "theme": "none"},
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
        if '{"themes":' in system:
            return VOCAB_OUT
        if '{"assignments":' in system:
            return ASSIGN_BATCH_OUT
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


def _write_parquet(tmp_path, records, drop_columns=()):
    import pandas as pd

    df = pd.DataFrame(records)
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    path = tmp_path / "responses.parquet"
    df.to_parquet(path)
    return path


def test_load_questions_bulk_matches_load_question(tmp_path):
    records = [
        {"dataset_id": "1", "question_id": "1", "question_label": "Better city?",
         "source_row_index": i, "response_key": f"1:1:{i}", "response_text": t,
         "is_nonanswer": t.strip().lower() in {"n/a"}}
        for i, t in enumerate(["More parks", "n/a", "", "Lower rent"])
    ] + [
        {"dataset_id": "1", "question_id": "2", "question_label": "Unsafe?",
         "source_row_index": i, "response_key": f"1:2:{i}", "response_text": t,
         "is_nonanswer": False}
        for i, t in enumerate(["Dark streets", "Speeding"])
    ]
    path = _write_parquet(tmp_path, records)

    bulk = induction.load_questions_bulk(path)
    assert set(bulk) == {"1", "2"}
    for q in ("1", "2"):
        rows_b, meta_b, filtered_b = bulk[q]
        rows_s, meta_s, filtered_s = induction.load_question(path, q)
        assert [(r.response_key, r.text) for r in rows_b] == \
               [(r.response_key, r.text) for r in rows_s]
        assert meta_b == meta_s
        assert filtered_b == filtered_s
    rows, meta, filtered = bulk["1"]
    assert meta["rows_used"] == 2 and meta["rows_empty"] == 1
    assert meta["rows_sentinel_filtered"] == 1
    assert meta["sentinel_source"] == "parquet"


def test_load_question_prefers_is_nonanswer_column(tmp_path):
    # The flag wins over the text: a row the export marked as a non-answer is
    # filtered even though the predicate wouldn't catch it, and vice versa.
    records = [
        {"dataset_id": "1", "question_id": "1", "question_label": "Better city?",
         "source_row_index": 0, "response_key": "1:1:0",
         "response_text": "maybe later", "is_nonanswer": True},
        {"dataset_id": "1", "question_id": "1", "question_label": "Better city?",
         "source_row_index": 1, "response_key": "1:1:1",
         "response_text": "n/a", "is_nonanswer": False},
    ]
    path = _write_parquet(tmp_path, records)
    rows, meta, filtered = induction.load_question(path, "1")
    assert [r.response_key for r in rows] == ["1:1:1"]
    assert [f["response_key"] for f in filtered] == ["1:1:0"]
    assert meta["sentinel_source"] == "parquet"

    # Old-schema parquet without the column: predicate fallback still works.
    old_path = tmp_path / "old"
    old_path.mkdir()
    path2 = _write_parquet(old_path, records, drop_columns=("is_nonanswer",))
    rows2, meta2, filtered2 = induction.load_question(path2, "1")
    assert [r.response_key for r in rows2] == ["1:1:0"]
    assert [f["response_key"] for f in filtered2] == ["1:1:1"]
    assert meta2["sentinel_source"] == "computed"


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


def test_parse_assign_batch_guards():
    batch = [_plabel(f"c00_0{i}", f"L{i}") for i in range(4)]
    raw = json.dumps({"assignments": [
        {"id": "c00_00", "theme": "theme a"},          # canonicalized
        {"id": "c00_00", "theme": "Theme B"},          # answered twice: first kept
        {"id": "ghost", "theme": "Theme A"},           # unknown id ignored
        {"id": "c00_01", "theme": "Theme C"},          # unknown theme -> unsorted
        {"id": "c00_02", "theme": "none"},             # explicit no-fit -> unsorted
        # c00_03 never mentioned -> unsorted
    ]})
    mapping, warnings = induction.parse_assign_batch(
        raw, batch, ["Theme A", "Theme B"])
    assert mapping == {"c00_00": "Theme A"}
    assert any("ghost" in w for w in warnings)
    assert any("answered twice" in w for w in warnings)
    assert any("unknown theme 'Theme C'" in w for w in warnings)


def test_parse_vocab_guards():
    raw = json.dumps({"themes": [
        {"name": "Crime", "description": "d"},
        {"name": "crime", "description": "dup, dropped"},
        {"name": "", "description": "blank, dropped"},
        {"name": "Transit", "description": "d"},
    ]})
    themes, warnings = induction.parse_vocab(raw)
    assert [t["name"] for t in themes] == ["Crime", "Transit"]
    assert any("duplicate theme" in w for w in warnings)

    many = json.dumps({"themes": [{"name": f"T{i}", "description": ""}
                                  for i in range(15)]})
    themes, warnings = induction.parse_vocab(many)
    assert len(themes) == 12
    assert any("keeping the first 12" in w for w in warnings)

    with pytest.raises(ValueError):
        induction.parse_vocab(json.dumps({"themes": []}))


def test_assign_batches_reassemble_and_contain_failures(monkeypatch):
    """Multi-batch assign: results merge across batches deterministically,
    and a failed batch's candidates end up unsorted and disclosed — never
    lost."""
    monkeypatch.setattr(induction, "ASSIGN_BATCH_SIZE", 2)

    class BatchClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.assign_calls = 0

        def complete(self, system, user):
            if '{"assignments":' in system:
                self.usage.add(100, 50)
                self.assign_calls += 1
                # batches of 2 over 5 provisional labels: [c00_00, c00_01],
                # [c00_02, c01_01], [c01_02]. Second batch dies both times.
                if "c00_02" in user:
                    raise RuntimeError("boom")
                out = {"assignments": []}
                for pid, theme in [("c00_00", "Street conditions"),
                                   ("c00_01", "Neighborhood crime"),
                                   ("c01_02", "none")]:
                    if f"id={pid}" in user:
                        out["assignments"].append({"id": pid, "theme": theme})
                return json.dumps(out)
            if "Theme: Unsorted" in system:
                # dedup of the unsorted bucket: mention nothing — the
                # forgotten-id guard keeps every member as-is
                self.usage.add(100, 50)
                return json.dumps({"groups": [], "too_broad": []})
            return super().complete(system, user)

    rows = _rows(12)
    meta = {"dataset_id": "ds", "question_id": "q4oe",
            "question_text": "What makes you feel unsafe?",
            "rows_for_question": 12, "rows_empty": 0,
            "rows_sentinel_filtered": 0, "rows_used": 12}
    client = BatchClient()
    taxonomy, report = induction.run_induction(rows, meta, client,
                                               chunk_size=6, seed=7)

    assert report["assign"]["batch_size"] == 2
    assert report["assign"]["n_batches"] == 3
    fails = [f for f in report["consolidation_failures"] if f["stage"] == "assign"]
    assert len(fails) == 1 and fails[0]["n_candidates"] == 2
    # the failed batch's candidates (Aggressive panhandling, Vehicle
    # break-ins) survive as unparented labels
    by_name = {l["name"]: l for l in taxonomy["labels"]}
    assert by_name["Aggressive panhandling"]["parent_id"] is None
    assert by_name["Vehicle break-ins"]["parent_id"] is None
    assert len(taxonomy["labels"]) == 5      # nothing lost


def test_checkpoint_round_trip_and_resume_equivalence(tmp_path):
    rows = _rows(12)
    meta = {"dataset_id": "ds", "question_id": "q4oe",
            "question_text": "What makes you feel unsafe in San Jose?",
            "rows_for_question": 12, "rows_empty": 0,
            "rows_sentinel_filtered": 0, "rows_used": 12}

    # uninterrupted run, checkpointing as it goes
    taxonomy_full, report_full = induction.run_induction(
        rows, meta, FakeClient(), chunk_size=6, seed=7, checkpoint_dir=tmp_path)
    ckpt = tmp_path / "candidates_checkpoint.json"
    assert ckpt.exists()

    # resume: consolidation over the loaded checkpoint must reproduce the
    # same taxonomy (modulo nothing — FakeClient is deterministic)
    cands, map_report, meta2, sha = induction.load_candidates_checkpoint(ckpt)
    assert sha == induction.prompt_hash("")
    assert meta2 == meta
    assert [c.cid for c in cands] == [
        c["cid"] for c in json.loads(ckpt.read_text(encoding="utf-8"))["candidates"]]
    taxonomy_resumed, report_resumed = induction.run_consolidation(
        cands, map_report, meta2, FakeClient())
    assert taxonomy_resumed["labels"] == taxonomy_full["labels"]
    assert taxonomy_resumed["parents"] == taxonomy_full["parents"]
    assert report_resumed["labels_final"] == report_full["labels_final"]


def test_vocab_failure_names_the_resume_path(tmp_path):
    class VocabDeadClient(FakeClient):
        def complete(self, system, user):
            if '{"themes":' in system:
                self.usage.add(1, 1)
                raise RuntimeError("MAX_TOKENS")
            return super().complete(system, user)

    meta = {"dataset_id": "ds", "question_id": "q4oe", "question_text": "q?",
            "rows_for_question": 12, "rows_empty": 0,
            "rows_sentinel_filtered": 0, "rows_used": 12}
    with pytest.raises(RuntimeError, match="--resume"):
        induction.run_induction(_rows(12), meta, VocabDeadClient(),
                                chunk_size=6, seed=7, checkpoint_dir=tmp_path)
    # the MAP spend is safe on disk
    assert (tmp_path / "candidates_checkpoint.json").exists()


def test_cross_failure_is_contained():
    class CrossDeadClient(FakeClient):
        def complete(self, system, user):
            if "across DIFFERENT themes" in system:
                self.usage.add(1, 1)
                raise RuntimeError("HTTP 500")
            return super().complete(system, user)

    meta = {"dataset_id": "ds", "question_id": "q4oe", "question_text": "q?",
            "rows_for_question": 12, "rows_empty": 0,
            "rows_sentinel_filtered": 0, "rows_used": 12}
    taxonomy, report = induction.run_induction(_rows(12), meta,
                                               CrossDeadClient(),
                                               chunk_size=6, seed=7)
    # taxonomy still built, skip disclosed
    assert report["labels_final"] == 4
    assert report["cross_theme_skipped"]["error"].startswith("RuntimeError")
    assert any(f["stage"] == "cross" for f in report["consolidation_failures"])


def test_dedup_theme_two_rounds_merge_across_sub_batches(monkeypatch):
    """Duplicates split across round-1 sub-batches become co-visible in the
    survivors round; a failing sub-batch passes through unmerged."""
    monkeypatch.setattr(induction, "DEDUP_MAX_SINGLE", 3)
    monkeypatch.setattr(induction, "DEDUP_SUB_BATCH", 2)
    # sub-batches: [A-dup1, B], [C, A-dup2] — the two A's never share a
    # round-1 call
    members = [_plabel("c00_00", "Car break-ins"),
               _plabel("c00_01", "Loud parties"),
               _plabel("c01_00", "Gang activity", chunk=1),
               _plabel("c01_01", "Vehicle break-ins", chunk=1)]

    class DedupClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.dedup_calls = []

        def complete(self, system, user):
            if '{"groups":' in system and "across DIFFERENT themes" not in system:
                self.usage.add(100, 50)
                ids = [line.split("id=")[1].split(" ")[0]
                       for line in user.splitlines() if "id=" in line]
                self.dedup_calls.append(ids)
                if ids == ["c00_00", "c00_01"]:       # round 1, batch 0
                    return json.dumps({"groups": [{"ids": ["c00_00"], "name": "Car break-ins"},
                                                  {"ids": ["c00_01"], "name": "Loud parties"}],
                                       "too_broad": []})
                if ids == ["c01_00", "c01_01"]:       # round 1, batch 1
                    raise RuntimeError("boom")        # pass through unmerged
                # round 2 over survivors: now both break-ins are visible
                return json.dumps({"groups": [
                    {"ids": ["c00_00", "c01_01"], "name": "Vehicle break-ins"},
                    {"ids": ["c00_01"], "name": "Loud parties"},
                    {"ids": ["c01_00"], "name": "Gang activity"}], "too_broad": []})
            return super().complete(system, user)

    client = DedupClient()
    kept, absorbed, log, warnings, failures = induction.dedup_theme(
        client, "q?", "Crime", members, "")

    assert [l.name for l in kept] == ["Vehicle break-ins", "Loud parties", "Gang activity"]
    merged = next(m for m in log if m["result_name"] == "Vehicle break-ins")
    assert set(merged["member_ids"]) == {"c00_00", "c01_01"}
    assert merged["round"] == 2
    assert len(failures) == 1 and failures[0]["stage"] == "dedup"
    assert failures[0]["batch"] == "round1:1"
    # three calls: two round-1 sub-batches (one failed) + one survivors round
    assert len(client.dedup_calls) == 3

    # concurrent sub-batches produce the identical result — reassembly is by
    # batch index, never completion order
    kept_p, absorbed_p, log_p, warnings_p, failures_p = induction.dedup_theme(
        DedupClient(), "q?", "Crime", members, "", workers=4)
    assert [l.name for l in kept_p] == [l.name for l in kept]
    assert [f["batch"] for f in failures_p] == ["round1:1"]


def test_run_induction_parallel_dedup_matches_serial(monkeypatch):
    """Theme-level dedup concurrency must not change the taxonomy."""
    rows = _rows(12)
    meta = {"dataset_id": "ds", "question_id": "q4oe",
            "question_text": "What makes you feel unsafe in San Jose?",
            "rows_for_question": 12, "rows_empty": 0,
            "rows_sentinel_filtered": 0, "rows_used": 12}
    tax_serial, _ = induction.run_induction(rows, meta, FakeClient(),
                                            chunk_size=6, seed=7, workers=1)
    tax_parallel, _ = induction.run_induction(rows, meta, FakeClient(),
                                              chunk_size=6, seed=7, workers=8)
    assert tax_parallel["labels"] == tax_serial["labels"]
    assert tax_parallel["parents"] == tax_serial["parents"]


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

    themes = [{"name": "t", "description": "d"}]
    stages = [
        induction.build_map_prompts("q?", rows, desc)[0],
        induction.build_vocab_prompts("q?", labels, desc)[0],
        induction.build_assign_batch_prompts("q?", themes, labels, desc)[0],
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
