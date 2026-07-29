"""Two-step ask endpoints: proposal, server-side gates, answer synthesis.
Model calls are faked; the filesystem context is a tmp_path fixture."""
import json

import pytest
from fastapi.testclient import TestClient

from app import ask_api, ask_service
from app.llm import Usage
from app.main import app


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

    monkeypatch.setattr(summary, "LABELS_DIR", tmp_path / "labels")
    monkeypatch.setattr(summary, "TAXONOMY_DIR", tmp_path / "taxonomy")
    monkeypatch.setattr(summary, "LEXICON_DIR", tmp_path / "lexicon")
    monkeypatch.setattr(summary, "LOCATIONS_DIR", tmp_path / "locations")
    monkeypatch.setattr(summary, "SUMMARY_DIR", tmp_path / "summary")
    monkeypatch.setattr(ask_service, "ANSWERS_DIR", tmp_path / "answers")

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


def test_route_returns_enriched_candidates(client, monkeypatch):
    _fake_gemini(monkeypatch, ROUTE_REPLY)
    r = client.post("/datasets/1/ask/route", json={"question": "what about theft?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True
    assert body["route"] == "retrieval"
    c = body["candidates"][0]
    assert c["label_id"] == "2_001" and c["name"] == "Theft"
    assert c["count"] == 2                       # real count, computed
    assert c["question_text"] == "what feels unsafe?"


def test_route_unanswerable_is_200(client, monkeypatch):
    _fake_gemini(monkeypatch, json.dumps(
        {"answerable": False, "route": "retrieval", "reason": "wrong domain",
         "candidates": [], "groups": [], "lexicon_concepts": []}))
    r = client.post("/datasets/1/ask/route", json={"question": "irrelevant?"})
    assert r.status_code == 200
    assert r.json()["answerable"] is False
    assert r.json()["candidates"] == []


def test_answer_happy_path_and_artifacts(client, monkeypatch, ask_ctx):
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval",
        "selected": [{"label_id": "9_999"}],
    })
    assert r.status_code == 422
    assert "9_999" in r.json()["detail"]


def test_answer_requires_selection(client):
    r = client.post("/datasets/1/ask/answer", json={
        "question": "q?", "route": "retrieval", "selected": [],
    })
    assert r.status_code == 422


def test_answer_location_args_ignored_without_location_layer(client, monkeypatch):
    """group_by/location_filter are guarded server-side: with no location
    artifact they fall back to category grouping instead of erroring."""
    _fake_gemini(monkeypatch, SYNTH_REPLY)
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/route", json={"question": "what about theft?"})
    body = r.json()
    assert body["available_actionability"] == {"general": 1, "specific": 1}
    assert body["actionability_filter"] == ""      # this reply asked for none


def test_answer_actionability_filter_restricts_evidence(client, monkeypatch,
                                                        ask_ctx):
    _fake_gemini(monkeypatch, json.dumps(
        {"answer_markdown": "One concrete ask: better lighting [1]."}))
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/answer", json={
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
    r = client.post("/datasets/1/ask/answer", json={
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

    r = client.post("/datasets/1/ask/answer", json={
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


def test_route_no_pipeline_state_is_409(client, monkeypatch):
    def boom(dataset_id, parquet=None, description=""):
        raise FileNotFoundError("no labels for dataset 99")
    monkeypatch.setattr(ask_service, "load_context", boom)
    r = client.post("/datasets/99/ask/route", json={"question": "q?"})
    assert r.status_code == 409
