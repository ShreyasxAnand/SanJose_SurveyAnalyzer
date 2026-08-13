"""Two-step ask endpoints: proposal, server-side gates, answer synthesis.
Model calls are faked; the filesystem context is a tmp_path fixture."""
import json

import pytest
from fastapi.testclient import TestClient

from app import ask_api, ask_service
from app.llm import Usage
from app.main import app

# Captured before any fixture monkeypatches the module attribute, so the cache
# tests below can exercise the REAL loader rather than ask_ctx's stub.
_REAL_LOAD_CONTEXT = ask_service.load_context


class FakeClient:
    """Returns canned JSON per call, in order, and records usage the way a
    real client does — the manifest's per-model split is derived from it."""
    model_id = "fake-model"

    def __init__(self, *replies, model_id: str | None = None):
        self.replies = list(replies)
        self.usage = Usage()
        if model_id:
            self.model_id = model_id

    def complete(self, system: str, user: str) -> str:
        self.usage.add(100, 20)
        return self.replies.pop(0)


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
    {"response_key": "1:2:0", "label_ids": ["2_001"], "uncategorized": False,
     "actionability": "general", "event_occurred": True},
    {"response_key": "1:2:1", "label_ids": ["2_001", "2_002"],
     "uncategorized": False, "actionability": "specific",
     "event_occurred": False},
    {"response_key": "1:2:2", "label_ids": [], "uncategorized": True,
     "actionability": None, "event_occurred": False},
]

ROUTE_REPLY = json.dumps({
    "answerable": True, "route": "retrieval", "reason": "asks what people say",
    "candidates": [
        {"label_id": "2_001", "relevance": "high", "rationale": "direct match"},
        {"label_id": "2_002", "relevance": "medium", "rationale": "related"},
    ],
    "groups": [], "lexicon_concepts": [],
})

SYNTH_REPLY = json.dumps({
    "answer_markdown": "People report theft [1] and break-ins [2]."
})


@pytest.fixture
def ask_ctx(tmp_path, monkeypatch):
    """A minimal on-disk pipeline state + a stubbed load_context that skips
    the parquet (texts come from a dict) but keeps everything else real."""
    from app import summary

    def _write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    from app import ask_cache

    monkeypatch.setattr(summary, "LABELS_DIR", tmp_path / "labels")
    monkeypatch.setattr(summary, "TAXONOMY_DIR", tmp_path / "taxonomy")
    monkeypatch.setattr(summary, "LEXICON_DIR", tmp_path / "lexicon")
    monkeypatch.setattr(summary, "LOCATIONS_DIR", tmp_path / "locations")
    monkeypatch.setattr(summary, "SUMMARY_DIR", tmp_path / "summary")
    monkeypatch.setattr(ask_service, "ANSWERS_DIR", tmp_path / "answers")
    monkeypatch.setattr(ask_cache, "CACHE_DIR", tmp_path / "answers")

    _write(tmp_path / "taxonomy/1/2/2026-01-01T00-00-00Z_aaaa/candidate_taxonomy.json",
           TAXONOMY)
    run = tmp_path / "labels/1/2/2026-01-01T00-05-00Z_bbbb"
    _write(run / "assignments.json", ASSIGNMENTS)
    _write(run / "manifest.json", {"taxonomy_run": "2026-01-01T00-00-00Z_aaaa"})

    texts = {"1:2:0": "my car got stolen", "1:2:1": "break-ins and it is dark",
             "1:2:2": "nothing"}
    real_load = ask_service.load_context

    def fake_load(dataset_id, parquet=None, description=""):
        s = summary.build_summary(dataset_id, description)
        summary.write_summary(s)
        import app.router as approuter
        members = {}
        assignments = json.loads((run / "assignments.json").read_text(encoding="utf-8"))
        members.update(approuter.members_by_label(assignments))
        actionability = {a["response_key"]: a["actionability"]
                         for a in assignments if a.get("actionability")}
        counts: dict[str, int] = {}
        for v in actionability.values():
            counts[v] = counts.get(v, 0) + 1
        coded = {a["response_key"] for a in assignments
                 if not a.get("not_returned") and not a.get("batch_failed")}
        events = {a["response_key"] for a in assignments
                  if a.get("event_occurred")}
        return ask_service.AskContext(
            dataset_id=dataset_id, summary=s,
            summary_text=summary.render_summary(s),
            index=summary.label_index(s), valid_ids=summary.valid_label_ids(s),
            question_ids=["2"], question_totals={"2": len(assignments)},
            lexicon={}, valid_concepts=set(), texts=texts,
            keys_by_question={"2": list(texts)}, members=members,
            actionability=actionability, actionability_counts=counts,
            events=events, event_coded=coded)

    monkeypatch.setattr(ask_service, "load_context", fake_load)
    return tmp_path


@pytest.fixture
def client(ask_ctx, monkeypatch):
    # Safety net: every model entry point starts out fatal, so a test that
    # forgets _fake_gemini fails loudly instead of quietly billing a real API
    # call. (A missed _synth_client patch did exactly that once.)
    def _no_real_calls(_name):
        def boom():
            raise AssertionError(
                f"ask_api.{_name}() was called without _fake_gemini() — this "
                "test would have made a real, billed model call")
        return boom

    monkeypatch.setattr(ask_api, "_client", _no_real_calls("_client"))
    monkeypatch.setattr(ask_api, "_synth_client", _no_real_calls("_synth_client"))
    return TestClient(app)


def _fake_gemini(monkeypatch, *replies):
    """Patch BOTH model entry points at once. Routing and synthesis are
    separate clients in production (different models); in tests they share one
    reply queue, since no single request uses both."""
    fake = FakeClient(*replies)
    monkeypatch.setattr(ask_api, "_client", lambda: fake)
    monkeypatch.setattr(ask_api, "_synth_client", lambda: fake)
    return fake


def test_context_key_sees_every_dataset_change(ask_ctx, monkeypatch):
    """The persistent ask cache keys on this fingerprint — every way a
    dataset's answer-relevant state can change MUST change it. Hand edits
    are the treacherous case: taxonomies invite in-place editing, which
    creates no new run directory."""
    from app import induction, subthemes, summary

    monkeypatch.setattr(induction, "DATA_DIR", ask_ctx)
    monkeypatch.setattr(subthemes, "SUBTHEMES_DIR", ask_ctx / "subthemes")

    def key():
        return ask_service.context_cache_key("1", "desc")

    def touch(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    k0 = key()
    # 1. hand-editing the taxonomy in place (no new run dir)
    tax = ask_ctx / "taxonomy/1/2/2026-01-01T00-00-00Z_aaaa/candidate_taxonomy.json"
    touch(tax, tax.read_text(encoding="utf-8") + " ")
    k1 = key()
    assert k1 != k0, "in-place taxonomy edit must invalidate"
    # 2. hand-editing assignments in place
    asg = ask_ctx / "labels/1/2/2026-01-01T00-05-00Z_bbbb/assignments.json"
    touch(asg, asg.read_text(encoding="utf-8") + " ")
    k2 = key()
    assert k2 != k1, "in-place assignments edit must invalidate"
    # 3. a new labels run directory (normal pipeline path)
    touch(ask_ctx / "labels/1/2/2026-02-02T00-00-00Z_cccc/assignments.json", "[]")
    k3 = key()
    assert k3 != k2, "new labels run must invalidate"
    # 4. a sub-themes run appearing, then being hand-edited
    sub = ask_ctx / "subthemes/1/2/2026-02-03T00-00-00Z_dddd/sub_taxonomy.json"
    touch(sub, json.dumps({"categories": []}))
    k4 = key()
    assert k4 != k3, "new sub-themes run must invalidate"
    touch(sub, json.dumps({"categories": [{"x": 1}]}))
    k5 = key()
    assert k5 != k4, "in-place sub-taxonomy edit must invalidate"
    # 5. exports in EITHER form — responses.parquet or reshaped.parquet
    touch(ask_ctx / "exports/1/responses.parquet", "v1")
    k6 = key()
    assert k6 != k5, "export write must invalidate"
    touch(ask_ctx / "exports/1/reshaped.parquet", "v1")
    k7 = key()
    assert k7 != k6, "reshaped-form export must invalidate too"
    # 6. locations / lexicon artifacts
    touch(ask_ctx / "locations/1/locations.json", "{}")
    k8 = key()
    assert k8 != k7, "locations change must invalidate"
    touch(ask_ctx / "lexicon/1/lexicon.json", "{}")
    k9 = key()
    assert k9 != k8, "lexicon change must invalidate"
    # 7. dataset description edits
    assert ask_service.context_cache_key("1", "different description") != k9
    # 8. and a different dataset id never shares a key
    assert ask_service.context_cache_key("2", "desc") != k9


def test_answer_repair_fires_only_on_violations(client, monkeypatch):
    """A draft with an invented count triggers exactly one repair call; the
    served answer is the corrected one and the verification record says so."""
    bad = json.dumps({"answer_markdown":
                      "**999 responses** report theft [1]."})
    fixed = json.dumps({"answer_markdown":
                        "**2 responses** report theft [1]."})
    fake = _fake_gemini(monkeypatch, ROUTE_REPLY, bad, fixed)
    client.post("/api/datasets/1/ask/route", json={"question": "verify me?"})
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "verify me?", "route": "retrieval", "reason": "r",
        "selected": [{"label_id": "2_001", "relevance": "high", "rationale": "x"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert not fake.replies, "route + synth + one repair, nothing more"
    assert "**2 responses**" in body["answer_markdown"]
    assert "999" not in body["answer_markdown"]
    v = body["verification"]
    assert v["checked"] and v["repaired"] and v["residual"] == []
    assert v["violations"][0]["value"] == 999


def test_ask_cache_serves_stored_route_and_answer(client, monkeypatch):
    """Identical request against identical data returns the stored response
    (cached=True, same run_id) with ZERO further model calls — the FakeClient
    carries exactly one reply per step, so a second real call would raise."""
    fake = _fake_gemini(monkeypatch, ROUTE_REPLY, SYNTH_REPLY)
    q = {"question": "what about theft, cached?"}
    r1 = client.post("/api/datasets/1/ask/route", json=q)
    assert r1.status_code == 200 and r1.json()["cached"] is False
    r2 = client.post("/api/datasets/1/ask/route", json=q)
    assert r2.status_code == 200 and r2.json()["cached"] is True
    assert {k: v for k, v in r2.json().items() if k != "cached"} \
        == {k: v for k, v in r1.json().items() if k != "cached"}

    body = {
        "question": "what about theft, cached?", "route": "retrieval",
        "reason": "r",
        "selected": [{"label_id": "2_001", "relevance": "high", "rationale": "x"}],
    }
    a1 = client.post("/api/datasets/1/ask/answer", json=body)
    assert a1.status_code == 200 and a1.json()["cached"] is False
    a2 = client.post("/api/datasets/1/ask/answer", json=body)
    assert a2.status_code == 200 and a2.json()["cached"] is True
    assert a2.json()["run_id"] == a1.json()["run_id"]
    assert not fake.replies, "every canned reply should have been consumed exactly once"

    # an edited selection is a DIFFERENT request — it must MISS the cache and
    # reach the model, which the empty reply queue turns into an IndexError
    edited = {**body, "selected": body["selected"]
              + [{"label_id": "2_002", "relevance": "low", "rationale": "y"}]}
    with pytest.raises(IndexError):
        client.post("/api/datasets/1/ask/answer", json=edited)


def test_route_returns_enriched_candidates(client, monkeypatch):
    _fake_gemini(monkeypatch, ROUTE_REPLY)
    r = client.post("/api/datasets/1/ask/route", json={"question": "what about theft?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True
    assert body["route"] == "retrieval"
    c = body["candidates"][0]
    assert c["label_id"] == "2_001" and c["name"] == "Theft"
    assert c["count"] == 2                       # real count, computed
    assert c["question_text"] == "what feels unsafe?"


def test_route_exposes_the_whole_taxonomy_tree(client, monkeypatch):
    """The review screen shows every parent category, not only the proposed
    ones, so the analyst can add as well as remove."""
    _fake_gemini(monkeypatch, json.dumps({
        "answerable": True, "route": "retrieval", "reason": "r",
        "candidates": [{"label_id": "2_001", "relevance": "high",
                        "rationale": "direct match"}],
        "groups": [], "lexicon_concepts": [],
    }))
    r = client.post("/api/datasets/1/ask/route", json={"question": "theft?"})
    groups = r.json()["available_categories"]
    assert {c["label_id"] for g in groups for c in g["children"]} == {"2_001", "2_002"}
    # the parent holding the proposal comes first — the proposal stays the
    # thing the analyst reads, with the rest of the taxonomy behind it
    assert groups[0]["parent_name"] == "safety"
    assert groups[0]["n_proposed"] == 1
    theft = groups[0]["children"][0]
    assert theft["proposed"] is True
    assert theft["relevance"] == "high" and theft["rationale"] == "direct match"
    assert theft["count"] == 2                     # real count, computed
    assert theft["description"] == "stolen things"
    # a category the router did not pick carries no relevance or rationale —
    # nothing invented to make it look endorsed
    other = next(c for g in groups for c in g["children"]
                 if c["label_id"] == "2_002")
    assert other["proposed"] is False
    assert other["relevance"] == "" and other["rationale"] == ""
    assert groups[-1]["n_proposed"] == 0


def test_parent_count_is_a_union_not_a_sum_of_children():
    """Summing child counts would double-count every multi-label response and
    put a number on screen that no operation over the data produced."""
    ctx = ask_service.AskContext(
        dataset_id="1",
        summary={"questions": [{
            "question_id": "2", "question_text": "q",
            "entries": [
                {"label_id": "2_001", "name": "A", "parent_name": "safety",
                 "description": "", "count": 2},
                {"label_id": "2_002", "name": "B", "parent_name": "safety",
                 "description": "", "count": 2},
            ]}]},
        summary_text="", index={}, valid_ids={"2_001", "2_002"},
        question_ids=["2"], question_totals={"2": 3},
        lexicon={}, valid_concepts=set(), texts={}, keys_by_question={},
        members={"2_001": ["r1", "r2"], "2_002": ["r2", "r3"]},
    )
    groups = ask_api._available_categories({"candidates": []}, ctx)
    assert len(groups) == 1
    # r2 carries both children: 3 responses, not 2 + 2
    assert groups[0].count_unique_responses == 3


def test_answer_logs_a_category_the_router_never_proposed(client, monkeypatch,
                                                          ask_ctx):
    """Additions are the mirror of deselections: one signals bad recall in the
    router, the other a miscoded category. Both belong in the log."""
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}, {"label_id": "2_002"}],
        "proposed_label_ids": ["2_001"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == ["2_002"]
    assert body["deselected"] == []
    log = (ask_ctx / "answers/1/selection_log.jsonl").read_text(encoding="utf-8")
    assert json.loads(log.splitlines()[0])["added"] == ["2_002"]


def test_route_unanswerable_is_200(client, monkeypatch):
    _fake_gemini(monkeypatch, json.dumps(
        {"answerable": False, "route": "retrieval", "reason": "wrong domain",
         "candidates": [], "groups": [], "lexicon_concepts": []}))
    r = client.post("/api/datasets/1/ask/route", json={"question": "irrelevant?"})
    assert r.status_code == 200
    assert r.json()["answerable"] is False
    assert r.json()["candidates"] == []


def test_answer_happy_path_and_artifacts(client, monkeypatch, ask_ctx):
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "what about theft?", "route": "retrieval", "reason": "r",
        "selected": [{"label_id": "2_001", "relevance": "high", "rationale": "x"}],
        "proposed_label_ids": ["2_001", "2_002"],
    })
    assert r.status_code == 200
    body = r.json()
    assert "theft [1]" in body["answer_markdown"]
    # sources are structured data, not embedded markdown
    assert "Sources" not in body["answer_markdown"]
    assert body["counts"] == {"2_001": 2}
    assert body["stats"] == {
        "categories_searched": 1, "categories_total": 2,
        "unique_responses": 2, "quotes_shown": 2, "quotes_cited": 2,
        # coverage guardrails: the fixture searches 2 of the question's 3
        # coded responses; any base under SMALL_BASE_N flags small_base
        "scope_total": 3, "scope_coverage": 0.667, "small_base": True,
    }
    # the process note is computed, never model-written — check its figures
    assert body["process_note"].startswith("Routed as retrieval")
    assert "Searched 1 of 2 categories (Theft)" in body["process_note"]
    assert "covering 2 unique responses" in body["process_note"]
    assert "2 of them are cited" in body["process_note"]
    assert [s["response_key"] for s in body["sources"]] == ["1:2:0", "1:2:1"]
    assert body["deselected"] == ["2_002"]

    run_dir = ask_ctx / "answers/1" / body["run_id"]
    # the on-disk document keeps the embedded Sources section (CLI/audit form)
    answer_doc = (run_dir / "answer.md").read_text(encoding="utf-8")
    assert "**Sources**" in answer_doc and "`1:2:0`" in answer_doc
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["deselected"] == ["2_002"]
    log = (ask_ctx / "answers/1/selection_log.jsonl").read_text(encoding="utf-8")
    assert json.loads(log.splitlines()[0])["deselected"] == ["2_002"]


def test_answer_rejects_unknown_label_id(client, monkeypatch):
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "9_999"}],
    })
    assert r.status_code == 422
    assert "9_999" in r.json()["detail"]


def test_answer_requires_selection(client):
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval", "selected": [],
    })
    assert r.status_code == 422


def test_answer_location_args_ignored_without_location_layer(client, monkeypatch):
    """group_by/location_filter are guarded server-side: with no location
    artifact they fall back to category grouping instead of erroring."""
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "where?", "route": "aggregate",
        "selected": [{"label_id": "2_001"}],
        "group_by": "location", "location_filter": ["downtown"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["location_counts"] == []
    assert body["location_denominator"] is None
    assert body["counts"] == {"2_001": 2}      # unfiltered — filter dropped
    assert body["location_filter"] == []       # echoed as dropped, not kept
    assert body["counts_unfiltered"] == {}


def test_route_exposes_actionability_availability(client, monkeypatch):
    """The review screen can only offer the filter if it knows the corpus
    carries the field — and what ticking it would cost."""
    _fake_gemini(monkeypatch, ROUTE_REPLY)
    r = client.post("/api/datasets/1/ask/route", json={"question": "what about theft?"})
    body = r.json()
    assert body["available_actionability"] == {"general": 1, "specific": 1}
    assert body["actionability_filter"] == ""      # this reply asked for none


def test_answer_actionability_filter_restricts_evidence(client, monkeypatch,
                                                        ask_ctx):
    _fake_gemini(monkeypatch, json.dumps(
        {"answer_markdown": "One concrete ask: better lighting [1]."}))
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "what should the city do?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}],
        "actionability_filter": "specific",
    })
    assert r.status_code == 200
    body = r.json()
    # 2_001 has two members; only 1:2:1 is marked specific
    assert body["counts"] == {"2_001": 1}
    assert body["counts_unfiltered"] == {"2_001": 2}
    assert body["actionability_filter"] == "specific"
    assert body["actionability_denominator"] == {
        "in_scope": 2, "coded": 2, "matching": 1}
    assert [s["response_key"] for s in body["sources"]] == ["1:2:1"]
    assert 'marked "specific": 1 of 2 in-scope responses' in body["process_note"]
    manifest = json.loads(
        (ask_ctx / "answers/1" / body["run_id"] / "manifest.json")
        .read_text(encoding="utf-8"))
    assert manifest["evidence"]["actionability_filter"] == "specific"
    assert manifest["route"]["actionability_filter"] == "specific"


def test_answer_rejects_unknown_actionability_filter(client, monkeypatch):
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}],
        "actionability_filter": "somewhat",
    })
    assert r.status_code == 422
    assert "somewhat" in r.json()["detail"]


def test_answer_actionability_ignored_when_labels_lack_the_field(
        client, monkeypatch):
    """An older labels run carries no actionability — the filter is dropped
    rather than silently answering from zero responses."""
    real_load = ask_service.load_context

    def uncoded(dataset_id, parquet=None, description=""):
        ctx = real_load(dataset_id, parquet, description)
        ctx.actionability = {}
        ctx.actionability_counts = {}
        return ctx

    monkeypatch.setattr(ask_service, "load_context", uncoded)
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}],
        "actionability_filter": "specific",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["actionability_filter"] == ""
    assert body["actionability_denominator"] is None
    assert body["counts"] == {"2_001": 2}       # unfiltered
    assert body["counts_unfiltered"] == {}


def test_answer_event_filter_restricts_to_first_hand_incidents(
        client, monkeypatch, ask_ctx):
    _fake_gemini(monkeypatch, json.dumps(
        {"answer_markdown": "One respondent had their car broken into [1]."}))
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "what have people actually experienced?",
        "route": "retrieval", "selected": [{"label_id": "2_001"}],
        "event_filter": "reported",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"2_001": 1}          # only 1:2:0 reports one
    assert body["counts_unfiltered"] == {"2_001": 2}
    assert body["event_filter"] == "reported"
    assert body["event_denominator"] == {"in_scope": 2, "coded": 2, "matching": 1}
    assert [s["response_key"] for s in body["sources"]] == ["1:2:0"]
    assert "not the same as nothing having happened" in body["process_note"]


def test_answer_rejects_negative_event_filter(client, monkeypatch):
    """There is no 'responses without an incident' filter — asking for one is
    a 422, not a silent inversion."""
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}], "event_filter": "not_reported",
    })
    assert r.status_code == 422
    assert "not_reported" in r.json()["detail"]


def test_event_denominator_excludes_failed_batch_rows(client, monkeypatch,
                                                      ask_ctx):
    """A row the labeling pass never returned carries event_occurred=False by
    default; counting it as 'no incident' would overstate the denominator."""
    real_load = ask_service.load_context

    def with_failed_row(dataset_id, parquet=None, description=""):
        ctx = real_load(dataset_id, parquet, description)
        ctx.event_coded.discard("1:2:1")      # pretend that batch failed
        return ctx

    monkeypatch.setattr(ask_service, "load_context", with_failed_row)
    _fake_gemini(monkeypatch, json.dumps({"answer_markdown": "An incident [1]."}))
    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "what happened to people?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}], "event_filter": "reported",
    })
    body = r.json()
    assert body["event_denominator"] == {"in_scope": 2, "coded": 1, "matching": 1}


def test_answer_uses_synth_client_and_records_both_models(client, monkeypatch,
                                                          ask_ctx):
    """Synthesis runs on its own (stronger) model; the manifest records both
    ids, and the routing client — which /answer never calls — must not appear
    in the per-model usage as though it had been used."""
    route_fake = FakeClient(model_id="route-model")
    synth_fake = FakeClient(SYNTH_REPLY, model_id="synth-model")
    monkeypatch.setattr(ask_api, "_client", lambda: route_fake)
    monkeypatch.setattr(ask_api, "_synth_client", lambda: synth_fake)

    r = client.post("/api/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "2_001"}]})
    assert r.status_code == 200
    assert route_fake.usage.calls == 0 and synth_fake.usage.calls == 1

    manifest = json.loads(
        (ask_ctx / "answers/1" / r.json()["run_id"] / "manifest.json")
        .read_text(encoding="utf-8"))
    assert manifest["model_id"] == "route-model"
    assert manifest["synth_model_id"] == "synth-model"
    assert manifest["usage"]["by_model"] == {
        "synth-model": {"calls": 1, "input_tokens": 100, "output_tokens": 20,
                        "thinking_tokens": 0, "est_cost_usd": None}}
    assert manifest["usage"]["calls"] == 1
    # a model with no published rate is named, not silently priced at zero
    assert manifest["usage"]["unpriced_models"] == ["synth-model"]


def test_usage_block_merges_same_model_and_drops_unused():
    a = FakeClient(model_id="m1")
    b = FakeClient(model_id="m1")
    unused = FakeClient(model_id="m2")
    a.usage.add(10, 1)
    b.usage.add(20, 2, thinking_tokens=1)
    block = ask_service.usage_block([a, b, unused, a], elapsed=1.5)
    # same model id merges; the same client passed twice is counted once
    assert block["by_model"] == {
        "m1": {"calls": 2, "input_tokens": 30, "output_tokens": 3,
               "thinking_tokens": 1, "est_cost_usd": None}}
    assert block["calls"] == 2 and block["input_tokens"] == 30
    assert block["unpriced_models"] == ["m1"]


def test_usage_block_prices_known_models_at_published_rates():
    """Thinking tokens arrive folded into output_tokens, so pricing them at
    the output rate is the whole point — the bill is not the visible text."""
    route = FakeClient(model_id="gemini-3.5-flash-lite")
    synth = FakeClient(model_id="gemini-3.6-flash")
    route.usage.add(1_000_000, 1_000_000)     # $0.30 + $2.50
    synth.usage.add(1_000_000, 1_000_000)     # $1.50 + $7.50
    block = ask_service.usage_block([route, synth], elapsed=1.0)
    assert block["by_model"]["gemini-3.5-flash-lite"]["est_cost_usd"] == 2.80
    assert block["by_model"]["gemini-3.6-flash"]["est_cost_usd"] == 9.00
    assert block["est_cost_usd"] == 11.80
    assert "unpriced_models" not in block


def test_dataset_parquet_ignores_a_newer_other_dataset(tmp_path, monkeypatch):
    """Regression: discover_parquet returns the most recent export across all
    datasets, so uploading dataset 3 silently repointed dataset 2's asks at a
    30-row corpus and 500'd them. Resolution must be by dataset id."""
    from app import induction

    monkeypatch.setattr(induction, "DATA_DIR", tmp_path)
    for ds in ("2", "3"):
        d = tmp_path / "exports" / ds
        d.mkdir(parents=True)
        (d / "responses.parquet").write_bytes(b"x")
    # make dataset 3 the most recent, the way a fresh upload would
    import os
    os.utime(tmp_path / "exports/2/responses.parquet", (1, 1))

    assert ask_service.dataset_parquet("2") == tmp_path / "exports/2/responses.parquet"
    assert ask_service.dataset_parquet("3") == tmp_path / "exports/3/responses.parquet"


def test_dataset_parquet_missing_export_raises_not_another_corpus(tmp_path,
                                                                  monkeypatch):
    """A dataset with no export must fail, never silently borrow another
    dataset's responses — a wrong answer beats no answer only never."""
    from app import induction

    monkeypatch.setattr(induction, "DATA_DIR", tmp_path)
    d = tmp_path / "exports" / "2"
    d.mkdir(parents=True)
    (d / "responses.parquet").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="dataset 4"):
        ask_service.dataset_parquet("4")


def test_route_no_pipeline_state_is_409(client, monkeypatch):
    def boom(dataset_id, parquet=None, description=""):
        raise FileNotFoundError("no labels for dataset 99")
    monkeypatch.setattr(ask_service, "load_context", boom)
    r = client.post("/api/datasets/99/ask/route", json={"question": "q?"})
    assert r.status_code == 409


def _two_question_ctx():
    """Minimal in-memory context with two survey questions, for scope tests."""
    from app import summary

    s = {"dataset_description": "d", "lexicon_concepts": [],
         "generated_utc": "2026-01-01T00:00:00Z",
         "questions": [
             {"question_id": "2", "question_text": "what feels unsafe?",
              "n_responses": 3, "n_uncategorized": 0, "labels_run": "r",
              "entries": [{"label_id": "2_001", "parent_name": "safety",
                           "name": "Theft", "description": "d", "count": 2}]},
             {"question_id": "3", "question_text": "what feels unclean?",
              "n_responses": 3, "n_uncategorized": 0, "labels_run": "r",
              "entries": [{"label_id": "3_001", "parent_name": "clean",
                           "name": "Litter", "description": "d", "count": 2}]},
         ]}
    return ask_service.AskContext(
        dataset_id="1", summary=s, summary_text=summary.render_summary(s),
        index=summary.label_index(s), valid_ids=summary.valid_label_ids(s),
        question_ids=["2", "3"], question_totals={"2": 3, "3": 3},
        lexicon={}, valid_concepts=set(), texts={}, keys_by_question={},
        members={"2_001": ["1:2:0"], "3_001": ["1:3:0"]})


class _RecordingClient(FakeClient):
    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        return super().complete(system, user)


def test_propose_question_scope_enforced_in_code():
    """A scoped proposal cannot contain an out-of-scope category: the router
    never sees the other questions' summary, and an id it invents anyway is
    dropped by validation — the scope is enforced in code, not prose."""
    ctx = _two_question_ctx()
    reply = json.dumps({
        "answerable": True, "route": "retrieval", "reason": "",
        "candidates": [
            {"label_id": "2_001", "relevance": "high", "rationale": ""},
            {"label_id": "3_001", "relevance": "high", "rationale": "out of scope"},
        ]})
    client = _RecordingClient(reply)
    route, stats = ask_service.propose(client, "q?", ctx,
                                       question_scope=["2"])
    assert [c["label_id"] for c in route["candidates"]] == ["2_001"]
    assert stats["invalid_label_ids"] == 1
    assert route["question_scope"] == ["2"]
    assert "what feels unsafe?" in client.last_system
    assert "what feels unclean?" not in client.last_system   # never shown


def test_propose_unknown_scope_raises():
    ctx = _two_question_ctx()
    with pytest.raises(ValueError, match="Unknown question ids"):
        ask_service.propose(FakeClient("{}"), "q?", ctx, question_scope=["9"])


# --- context cache: the same load must not be paid twice per question ------


@pytest.fixture
def real_ctx(ask_ctx, monkeypatch):
    """ask_ctx's stub replaces load_context wholesale; these tests need the
    real one. Restore it and stub only the corpus read underneath."""
    from app import induction

    monkeypatch.setattr(ask_service, "load_context", _REAL_LOAD_CONTEXT)
    monkeypatch.setattr(ask_service, "dataset_parquet",
                        lambda ds, explicit=None: ask_ctx / "export.parquet")
    rows = [induction.ResponseRow(response_key=k, text=t) for k, t in
            {"1:2:0": "my car got stolen",
             "1:2:1": "break-ins and it is dark",
             "1:2:2": "nothing"}.items()]
    monkeypatch.setattr(
        induction, "load_questions_bulk",
        lambda pq, qs: {q: (rows, {"question_id": q, "question_text": "t"}, [])
                        for q in qs})
    ask_service.invalidate_context_cache()
    yield ask_ctx
    ask_service.invalidate_context_cache()


def _load(**kw):
    return _REAL_LOAD_CONTEXT("1", **kw)


def test_context_cache_serves_the_second_load_and_discloses_it(
        real_ctx, monkeypatch):
    """The stateless two-step flow loads context for /route and again for
    /answer. Nothing between them can change it, so the second must be free —
    and must say it came from the cache rather than quietly looking fast."""
    import app.summary as summary_module
    calls = {"n": 0}
    orig = summary_module.build_summary

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(summary_module, "build_summary", counting)

    a = _load(description="d")
    b = _load(description="d")
    assert calls["n"] == 1                    # the corpus was read once
    assert a.context_source == "computed"
    assert b.context_source == "cache"
    assert b.load_seconds == 0.0
    assert b.members == a.members             # same evidence, not a rebuild


def test_context_cache_misses_when_a_new_labels_run_appears(real_ctx):
    """A CLI label/review/incremental run in another process writes a NEW run
    dir; the key is a readdir, so it cannot be answered around."""
    import app.summary as summary_module
    _load(description="d")
    assert _load(description="d").context_source == "cache"

    newer = summary_module.LABELS_DIR / "1" / "2" / "2026-06-06T00-00-00Z_zzzz"
    newer.mkdir(parents=True)
    (newer / "assignments.json").write_text(json.dumps(ASSIGNMENTS),
                                            encoding="utf-8")
    assert _load(description="d").context_source == "computed"


def test_context_cache_misses_on_a_different_description(real_ctx):
    """The description feeds the summary the router reads, so it is part of
    the key — a cached context must never be served under another one."""
    _load(description="one")
    assert _load(description="two").context_source == "computed"


def test_explicit_parquet_never_served_from_cache(real_ctx):
    """`--parquet` is the CLI steering at a specific file; a cache keyed on the
    dataset's own export must not answer for it."""
    _load(description="d")
    assert _load(parquet=str(real_ctx / "other.parquet"),
                 description="d").context_source == "computed"


def test_invalidate_drops_the_entry(real_ctx):
    """_write_exports calls this: an append or re-export must never be
    answered around, whatever the clock says about mtime."""
    _load(description="d")
    assert _load(description="d").context_source == "cache"
    ask_service.invalidate_context_cache("1")
    assert _load(description="d").context_source == "computed"
