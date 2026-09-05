"""Pipeline runner: the estimate gate, the one-job-at-a-time lock, and stage
failure containment. No subprocess actually runs a model — `_run_stage` is
stubbed, so these are about the orchestration, not the CLI stages themselves."""
import json

import pytest
from fastapi.testclient import TestClient

from app import pipeline, pipeline_api
from app.main import app


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "JOBS_DIR", tmp_path / "jobs")
    # a clean registry per test — the lock and registry are module-level
    monkeypatch.setattr(pipeline, "_JOBS", {})
    monkeypatch.setattr(pipeline, "_ACTIVE", None)
    return tmp_path / "jobs"


@pytest.fixture
def two_questions(monkeypatch, tmp_path):
    """Pretend dataset 7 has an export with two questions."""
    pq = tmp_path / "exports" / "7" / "responses.parquet"
    pq.parent.mkdir(parents=True)
    pq.write_bytes(b"x")
    monkeypatch.setattr(pipeline.ask_service, "dataset_parquet",
                        lambda ds, explicit=None: pq)
    monkeypatch.setattr(pipeline.induction, "list_questions",
                        lambda p: [{"question_id": "1", "n_rows": 10,
                                    "question_text": "a"},
                                   {"question_id": "2", "n_rows": 20,
                                    "question_text": "b"}])
    return pq


def test_stages_cover_every_question_then_the_shared_layers(jobs_dir, two_questions):
    stages = pipeline._build_stages("7", two_questions, ["1", "2"], batch_size=60)
    assert [s.key for s in stages] == [
        "induce:1", "induce:2", "label", "subthemes",
        "lexicon", "locations", "summary"]
    # sub-themes read the latest labels run to find which categories cleared
    # the member floor, so they must come after labeling and before the
    # summary that renders them
    sub = next(s for s in stages if s.key == "subthemes")
    assert sub.command[:2] == ["-m", "scripts.subthemes"]
    assert "--all" in sub.command
    # labeling runs once with --all, at the batch size the 30k run proved
    # reliable (80 produced deterministic malformed-JSON failures)
    label = next(s for s in stages if s.key == "label")
    assert "--all" in label.command
    assert label.command[label.command.index("--batch-size") + 1] == "60"


def test_only_one_job_runs_at_a_time(jobs_dir, two_questions, monkeypatch):
    """Two concurrent runs would double the request rate against the shared
    per-process rate limiter and race on the same run directories."""
    started = []

    def never_finish(job, stage, log_path):
        started.append(stage.key)
        stage.status = "running"
        return False          # stop the worker without touching the filesystem

    monkeypatch.setattr(pipeline, "_run_stage", never_finish)
    monkeypatch.setattr(pipeline.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: None})())

    pipeline.start_job("7")
    with pytest.raises(RuntimeError, match="already in progress"):
        pipeline.start_job("7")


def test_failed_stage_skips_the_rest_and_marks_the_job_failed(jobs_dir,
                                                              two_questions,
                                                              monkeypatch):
    """Labeling against a missing taxonomy would fail again with a worse
    message, so the run stops at the first failure instead of grinding on."""
    def fail_second(job, stage, log_path):
        if stage.key == "induce:1":
            stage.status = "done"
            return True
        stage.status = "failed"
        stage.error = "exit code 1"
        return False

    monkeypatch.setattr(pipeline, "_run_stage", fail_second)
    job = pipeline.Job(job_id="j1", dataset_id="7", created_utc="t",
                       stages=pipeline._build_stages("7", two_questions,
                                                     ["1", "2"], 60))
    pipeline._worker(job)

    assert job.status == "failed"
    assert "induce:2" in job.error or "Induce" in job.error
    by_key = {s.key: s.status for s in job.stages}
    assert by_key["induce:1"] == "done"
    assert by_key["induce:2"] == "failed"
    assert by_key["label"] == "skipped"
    assert by_key["summary"] == "skipped"
    # the lock is released even on failure, or the next run is a permanent 409
    assert pipeline._ACTIVE is None


def test_worker_writes_status_json_for_a_killed_server(jobs_dir, two_questions,
                                                       monkeypatch):
    """A --reload restart kills the run; status.json on disk is then the only
    record of how far it got."""
    monkeypatch.setattr(pipeline, "_run_stage",
                        lambda job, stage, log: (setattr(stage, "status", "done"),
                                                 True)[1])
    job = pipeline.Job(job_id="j2", dataset_id="7", created_utc="t",
                       stages=pipeline._build_stages("7", two_questions, ["1"], 60))
    pipeline._worker(job)

    status = json.loads((jobs_dir / "7" / "j2" / "status.json")
                        .read_text(encoding="utf-8"))
    assert status["status"] == "done"
    assert [s["status"] for s in status["stages"]] == ["done"] * len(job.stages)
    # the exact command line stays in the on-disk record — it is what makes an
    # interrupted stage reproducible by hand
    assert status["stages"][0]["command"][:2] == ["-m", "scripts.induce"]


def test_api_does_not_leak_server_paths_to_the_browser(jobs_dir, two_questions):
    """status.json keeps the command line for auditing; the HTTP response does
    not — the browser has no use for absolute paths on the server."""
    job = pipeline.Job(job_id="j3", dataset_id="7", created_utc="t",
                       stages=pipeline._build_stages("7", two_questions, ["1"], 60))
    out = pipeline_api._job_out(job).model_dump()
    assert "command" not in out["stages"][0]
    assert out["stages"][0]["key"] == "induce:1"


def test_estimate_marks_planned_and_projected_separately(jobs_dir, two_questions,
                                                         monkeypatch):
    """A projection presented as a plan is an invented number. Induction's
    figure is planned; labeling's is extrapolated on a dataset with no
    taxonomy, because the prompts it would count do not exist yet."""
    monkeypatch.setattr(pipeline.induction, "resolve_description",
                        lambda d, p: "desc")
    monkeypatch.setattr(pipeline.induction, "load_questions_bulk",
                        lambda p, qs: _rows_by_q({q: [f"7:{q}:{i}"
                                                      for i in range(4)]
                                                 for q in qs}))
    monkeypatch.setattr(pipeline.induction, "plan_dry_run", _fake_plan)
    monkeypatch.setattr(pipeline, "_taxonomy_exists", lambda ds, q: q == "2")
    # q2 is "already induced" for the disclosure check, but no taxonomy file
    # exists to plan labeling against, so both labeling rows stay projected
    monkeypatch.setattr(pipeline, "_latest_taxonomy", lambda ds, q: None)

    est = pipeline.estimate_dataset("7")
    basis = {(i["stage"], i["basis"]) for i in est["items"]}
    assert ("induce", "planned") in basis
    assert ("label", "projected") in basis
    assert ("lexicon", "projected") in basis
    # the headline is exactly what the rows add up to — no separate arithmetic
    assert est["est_total_usd"] == pytest.approx(
        sum(i["est_cost_usd"] for i in est["items"]), abs=1e-6)
    # and it sits inside the band, which is a band and not a point
    assert est["est_low_usd"] <= est["est_total_usd"] <= est["est_high_usd"]
    # a question that already has a taxonomy is disclosed, not silently redone
    assert est["questions_with_existing_taxonomy"] == ["2"]


def test_estimate_counts_labeling_prompts_when_a_taxonomy_exists(
        jobs_dir, two_questions, monkeypatch):
    """The one basis upgrade this rework buys: with a taxonomy on disk the
    labeling prompts are fully determined, so they are built and counted
    instead of multiplied by a rate."""
    monkeypatch.setattr(pipeline.induction, "resolve_description",
                        lambda d, p: "desc")
    monkeypatch.setattr(pipeline.induction, "load_questions_bulk",
                        lambda p, qs: _rows_by_q({q: [f"7:{q}:{i}"
                                                      for i in range(4)]
                                                 for q in qs}))
    monkeypatch.setattr(pipeline.induction, "plan_dry_run", _fake_plan)
    monkeypatch.setattr(pipeline, "_taxonomy_exists", lambda ds, q: True)
    monkeypatch.setattr(pipeline, "_latest_taxonomy", lambda ds, q: {
        "question_text": "q?",
        "labels": [{"label_id": "1_001", "name": "parks",
                    "description": "green space"}],
    })

    est = pipeline.estimate_dataset("7")
    label_rows = [i for i in est["items"] if i["stage"] == "label"]
    assert label_rows, "expected a labeling row per question"
    for row in label_rows:
        # no credentials in the offline suite, so the counter falls back —
        # but it fell back on REAL prompts, not on a per-response rate
        assert row["input_basis"] == "heuristic"
        assert row["est_input_tokens"] > 0
    # every row's tokens roll up into the dataset totals
    assert est["est_input_tokens"] == sum(
        i["est_input_tokens"] for i in est["items"])


def test_run_endpoint_is_409_when_a_job_is_in_flight(jobs_dir, two_questions,
                                                     monkeypatch):
    monkeypatch.setattr(pipeline, "start_job",
                        lambda ds, batch_size=60, mode="full": (_ for _ in ()).throw(
                            RuntimeError("A pipeline run is already in progress")))
    client = TestClient(app)
    r = client.post("/api/datasets/7/pipeline/run", json={})
    assert r.status_code == 409
    assert "already in progress" in r.json()["detail"]


def test_status_is_null_before_any_run(jobs_dir, monkeypatch):
    client = TestClient(app)
    r = client.get("/api/datasets/7/pipeline/status")
    assert r.status_code == 200
    assert r.json() is None


# --- incremental mode ------------------------------------------------------

class _Row:
    def __init__(self, key):
        self.response_key = key
        self.text = "t"


def _rows_by_q(spec):
    """{qid: [response_keys]} -> the load_questions_bulk shape."""
    return {q: ([_Row(k) for k in keys],
                {"question_id": q, "question_text": f"q{q}"}, [])
            for q, keys in spec.items()}


def _fake_plan(rows, meta, cs, seed, pi, po, desc, **kwargs):
    """Stand-in for induction.plan_dry_run.

    Takes **kwargs because the real one now also accepts `counter` and `cal`;
    returns the full key set `_priced_item` reads, including the token counts
    it prices from (the stubbed est_cost_usd is deliberately ignored by the
    caller, which re-prices from tokens at the configured rate).
    """
    return {
        "n_chunks": 1,
        "total_calls": 3,
        "responses_usable": 500,
        "responses_sentinel_filtered": 0,
        "responses_empty": 0,
        "est_input_tokens": 20_000,
        "est_output_tokens": 5_000,
        "est_cost_usd": 0.05,
        "input_basis": "counted",
        "count_calls": 1,
    }


def test_incremental_stages_pick_the_right_branch_per_question(
        jobs_dir, two_questions, monkeypatch):
    """q1: taxonomy+labels with new rows -> label_incr. q2: taxonomy only ->
    full label for that question. q3: nothing -> induce + label."""
    monkeypatch.setattr(pipeline, "_taxonomy_exists",
                        lambda ds, q: q in {"1", "2"})
    monkeypatch.setattr(pipeline, "_prior_label_keys",
                        lambda ds, q: {"7:1:0"} if q == "1" else None)

    rows = _rows_by_q({"1": ["7:1:0", "7:1:9"], "2": ["7:2:0"], "3": ["7:3:0"]})
    stages = pipeline._build_incremental_stages("7", two_questions, rows, 60)
    assert [s.key for s in stages] == [
        "label_incr:1", "label:2", "induce:3", "label:3",
        "subthemes", "lexicon", "locations", "summary"]
    incr = stages[0]
    assert incr.command[:2] == ["-m", "scripts.label_incremental"]
    assert incr.command[incr.command.index("--batch-size") + 1] == "60"


def test_incremental_stage_skipped_when_nothing_is_new(
        jobs_dir, two_questions, monkeypatch):
    monkeypatch.setattr(pipeline, "_taxonomy_exists", lambda ds, q: True)
    monkeypatch.setattr(pipeline, "_prior_label_keys",
                        lambda ds, q: {"7:1:0", "7:2:0"})
    rows = _rows_by_q({"1": ["7:1:0"], "2": ["7:2:0"]})
    with pytest.raises(ValueError, match="already labelled"):
        pipeline._build_incremental_stages("7", two_questions, rows, 60)


def test_incremental_estimate_is_all_projected(jobs_dir, two_questions,
                                               monkeypatch):
    """The pool size is unknown until run time, so no labeling row in an
    incremental estimate may claim basis='planned'. Scoped to questions that
    already have a taxonomy (_taxonomy_exists is stubbed True below) — a
    brand-new column is the one case that still gets a genuinely planned full
    induction, pinned separately by the test beneath this one."""
    monkeypatch.setattr(pipeline.induction, "resolve_description",
                        lambda d, p: "desc")
    monkeypatch.setattr(pipeline.induction, "load_questions_bulk",
                        lambda p, qs: _rows_by_q({"1": ["7:1:0", "7:1:8", "7:1:9"],
                                                  "2": ["7:2:0"]}))
    monkeypatch.setattr(pipeline, "_taxonomy_exists", lambda ds, q: True)
    monkeypatch.setattr(pipeline, "_prior_label_keys",
                        lambda ds, q: {"7:1:0", "7:2:0"})

    est = pipeline.estimate_dataset("7", mode="incremental")
    assert est["mode"] == "incremental"
    assert all(i["basis"] == "projected" for i in est["items"])
    incr_items = [i for i in est["items"] if i["stage"] == "label_incr"]
    assert [i["question_id"] for i in incr_items] == ["1"]  # q2 has nothing new
    assert incr_items[0]["responses"] == 2
    assert est["n_new_responses"] == 2


def test_incremental_estimate_plans_a_brand_new_column(jobs_dir, two_questions,
                                                       monkeypatch):
    """The documented exception: a question with no taxonomy gets full
    induction, whose chunking and prompt sizes are real — so it is `planned`
    while every labeling row around it stays `projected`."""
    monkeypatch.setattr(pipeline.induction, "resolve_description",
                        lambda d, p: "desc")
    monkeypatch.setattr(pipeline.induction, "load_questions_bulk",
                        lambda p, qs: _rows_by_q({"1": ["7:1:0", "7:1:8"],
                                                  "2": ["7:2:0"]}))
    monkeypatch.setattr(pipeline.induction, "plan_dry_run", _fake_plan)
    # q2 is the new column: no taxonomy, no labels
    monkeypatch.setattr(pipeline, "_taxonomy_exists", lambda ds, q: q != "2")
    monkeypatch.setattr(pipeline, "_prior_label_keys",
                        lambda ds, q: None if q == "2" else {"7:1:0"})

    est = pipeline.estimate_dataset("7", mode="incremental")
    planned = [i for i in est["items"] if i["basis"] == "planned"]
    assert [(i["stage"], i["question_id"]) for i in planned] == [("induce", "2")]
    assert all(i["basis"] == "projected"
               for i in est["items"] if i["stage"] != "induce")


def _labels_manifest(labels_dir, question, run_id, cost, created_utc):
    run = labels_dir / "7" / question / run_id
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"created_utc": created_utc,
                    "usage": {"est_cost_usd": cost}}), encoding="utf-8")
    (run / "assignments.json").write_text("[]", encoding="utf-8")
    return run


def test_stage_cost_reads_per_question_manifest_for_label_incr(
        jobs_dir, tmp_path, monkeypatch):
    import app.summary as summary_module
    labels_dir = tmp_path / "labels"
    _labels_manifest(labels_dir, "1", "2026-01-01T00-00-00Z_incr", 0.042,
                     "2026-01-01T00-00-05Z")
    monkeypatch.setattr(summary_module, "LABELS_DIR", labels_dir)

    job = pipeline.Job(job_id="j", dataset_id="7",
                       created_utc="2026-01-01T00-00-00Z")
    stage = pipeline.Stage(key="label_incr:1", label="x", command=[])
    assert pipeline._stage_cost(job, stage) == 0.042


def test_stage_cost_ignores_runs_this_job_did_not_write(
        jobs_dir, tmp_path, monkeypatch):
    """A `label --all` that dies partway leaves later questions holding only
    their PREVIOUS runs. Summing those would bill this job for spend it never
    made, so a manifest older than the job start does not count."""
    import app.summary as summary_module
    labels_dir = tmp_path / "labels"
    # q1 was relabelled by this job; q2's newest run predates it
    _labels_manifest(labels_dir, "1", "2026-06-01T12-00-05Z_abcd1234", 0.10,
                     "2026-06-01T12-00-09Z")
    _labels_manifest(labels_dir, "2", "2026-01-01T00-00-00Z_beefcafe", 5.00,
                     "2026-01-01T00-00-02Z")
    monkeypatch.setattr(summary_module, "LABELS_DIR", labels_dir)

    job = pipeline.Job(job_id="j", dataset_id="7",
                       created_utc="2026-06-01T12-00-00Z")
    stage = pipeline.Stage(key="label", label="x", command=[])
    assert pipeline._stage_cost(job, stage) == 0.10

    # and an unstamped manifest is not attributed either
    stale = pipeline.Job(job_id="j2", dataset_id="7",
                         created_utc="2026-12-01T00-00-00Z")
    assert pipeline._stage_cost(stale, stage) is None
