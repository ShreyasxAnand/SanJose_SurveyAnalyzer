"""Run the induce -> label -> lexicon -> locations pipeline from the browser.

Until now every stage after ingest was CLI-only: an analyst could upload a file
through the UI, get an export, and then hit a 409 on the Ask tab forever,
because nothing in the browser could produce a taxonomy or labels. This module
is the missing half — one background job per dataset, driven by a button.

Two design choices worth keeping:

**Stages are the real CLI commands, run as subprocesses.** `scripts.induce`,
`scripts.label` and friends are the validated, measured code paths; re-implementing
them behind the API would create a second pipeline that silently drifts from the
one the repo documents. Running them as `sys.executable -m scripts.X` means the
browser and the terminal cannot diverge, and stdout lands verbatim in the job
log for auditing. Cost comes from each stage's own run manifest, never from
scraping stdout.

**Cost is estimated before anything is spent.** `estimate_dataset` runs only
free code (`induction.plan_dry_run` plus a projection for labeling) so the
analyst approves a figure before the first billed call — the same "human gates
at the expensive irreversible steps" rule the taxonomy review follows. Which
figures are *planned* and which are *projected* is marked per row and must stay
marked: a projection presented as a plan is exactly the kind of invented number
this project refuses to produce.

Job state is in memory with `status.json` mirrored to disk at every transition.
A server restart (e.g. uvicorn --reload) kills an in-flight run, and the status
file is then the record of how far it got; induction checkpoints its MAP phase,
so re-running skips the expensive part it already paid for. Do not use --reload
while a real run is in flight.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ask_service, induction

BACKEND_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = induction.DATA_DIR / "jobs"

# Measured on the test dataset and the 30k production run: ~$0.10 per 500
# responses for labeling. Used only for the pre-run
# projection, and only when no taxonomy exists yet to plan against — labeling's
# own dry-run needs one. Always surfaced as "projected", never as a plan.
LABEL_USD_PER_RESPONSE = 0.10 / 500
# One grouping call each, bounded by MAX_CANDIDATES / MAX_SPANS rather than by
# corpus size, so these do not scale with the file.
FLAT_STAGE_USD = 0.001

VALID_STATUS = {"pending", "running", "done", "failed", "skipped"}


@dataclass
class Stage:
    """One CLI invocation. `cost_usd` is read from the stage's own manifest
    after it finishes — None means the stage does not report a cost (or did
    not get far enough to write one)."""
    key: str
    label: str
    command: list[str]
    status: str = "pending"
    detail: str = ""
    cost_usd: float | None = None
    seconds: float | None = None
    error: str = ""


@dataclass
class Job:
    job_id: str
    dataset_id: str
    status: str = "running"
    stages: list[Stage] = field(default_factory=list)
    created_utc: str = ""
    finished_utc: str = ""
    error: str = ""
    cancel_requested: bool = False

    @property
    def cost_usd(self) -> float:
        return round(sum(s.cost_usd or 0.0 for s in self.stages), 6)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "created_utc": self.created_utc,
            "finished_utc": self.finished_utc,
            "error": self.error,
            "cost_usd": self.cost_usd,
            "stages": [asdict(s) for s in self.stages],
        }


# One job at a time, process-wide. Two concurrent runs would double the request
# rate against the shared per-process rate limiter in llm.py and race on the
# same run directories.
_LOCK = threading.Lock()
_JOBS: dict[str, Job] = {}
_ACTIVE: str | None = None


def _job_dir(dataset_id: str, job_id: str) -> Path:
    return JOBS_DIR / str(dataset_id) / job_id


def _write_status(job: Job) -> None:
    d = _job_dir(job.dataset_id, job.job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(
        json.dumps(job.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Estimation — free, no API calls, nothing written
# ---------------------------------------------------------------------------


def _taxonomy_exists(dataset_id: str, question_id: str) -> bool:
    from .summary import latest_run_dir

    return latest_run_dir(induction.TAXONOMY_DIR / str(dataset_id) / str(question_id),
                          "candidate_taxonomy.json") is not None


def _prior_label_keys(dataset_id: str, question_id: str) -> set[str] | None:
    """response_keys covered by the question's latest labels run, or None when
    no labels run exists at all (a different state than "run with zero rows")."""
    from .summary import LABELS_DIR, latest_run_dir

    run = latest_run_dir(LABELS_DIR / str(dataset_id) / str(question_id),
                         "assignments.json")
    if run is None:
        return None
    try:
        assignments = json.loads(
            (run / "assignments.json").read_text(encoding="utf-8"))
        return {a.get("response_key") for a in assignments}
    except (ValueError, OSError):
        return None


def _incremental_plan(dataset_id: str, rows_by_q: dict) -> dict[str, dict]:
    """Per-question facts the incremental mode branches on: whether a taxonomy
    and a labels run exist, and how many rows the latest labels run has never
    seen. Shared by estimate and stage building so the plan the analyst
    approves and the job that runs cannot disagree."""
    plan: dict[str, dict] = {}
    for qid, (rows, _meta, _filtered) in rows_by_q.items():
        prior = _prior_label_keys(dataset_id, qid)
        plan[qid] = {
            "has_taxonomy": _taxonomy_exists(dataset_id, qid),
            "has_labels": prior is not None,
            "n_rows": len(rows),
            "n_new": (len(rows) if prior is None
                      else sum(1 for r in rows if r.response_key not in prior)),
        }
    return plan


def estimate_dataset(dataset_id: str, chunk_size: int = 120, seed: int = 7,
                     price_in: float = 0.30, price_out: float = 2.50,
                     description: str = "", mode: str = "full") -> dict:
    """The plan the analyst approves. Runs only free code.

    In full mode, induction's plan is exact (real chunking, real prompt sizes)
    while labeling is a *projection* from a measured rate, because its real
    dry-run needs a taxonomy that does not exist yet on a fresh dataset —
    flagged as such in `basis` so the UI can say which is which.

    In incremental mode (after an append), every labeling figure is a
    projection: the new rows are labeled at the measured rate, and the
    pool-induction step's size — the uncovered rows among them — is only
    determined at run time. The ONE exception is a question that has no
    taxonomy at all (a newly selected column): that gets full induction, whose
    plan is exact for the same reason it is in full mode, so it is correctly
    tagged `basis: "planned"`. Nothing else in incremental mode may claim it.
    """
    if mode not in {"full", "incremental"}:
        raise ValueError(f"Unknown pipeline mode: {mode!r}")

    parquet = ask_service.dataset_parquet(dataset_id)
    description = induction.resolve_description(description or None, parquet)
    questions = induction.list_questions(parquet)
    if not questions:
        raise ValueError(f"Dataset {dataset_id} has no questions in its export")

    rows_by_q = induction.load_questions_bulk(
        parquet, [q["question_id"] for q in questions])

    items: list[dict] = []
    total = 0.0
    n_new_responses = 0
    if mode == "incremental":
        inc_plan = _incremental_plan(dataset_id, rows_by_q)
        for q in questions:
            qid = q["question_id"]
            rows, meta, _filtered = rows_by_q[qid]
            facts = inc_plan[qid]
            if not facts["has_taxonomy"]:
                # a newly selected column: full induce + full label, as ever
                plan = induction.plan_dry_run(rows, meta, chunk_size, seed,
                                              price_in, price_out, description)
                items.append({
                    "stage": "induce", "question_id": qid,
                    "question_text": meta["question_text"],
                    "detail": f"no taxonomy yet — full induction, "
                              f"{plan['n_chunks']} chunk(s)",
                    "responses": plan["responses_usable"],
                    "est_cost_usd": plan["est_cost_usd"],
                    "basis": "planned",
                })
                total += plan["est_cost_usd"]
            if not facts["has_labels"]:
                cost = round(facts["n_rows"] * LABEL_USD_PER_RESPONSE, 4)
                items.append({
                    "stage": "label", "question_id": qid,
                    "question_text": meta["question_text"],
                    "detail": f"no labels run yet — all {facts['n_rows']} "
                              f"responses at the measured rate (~$0.10 per 500)",
                    "responses": facts["n_rows"],
                    "est_cost_usd": cost,
                    "basis": "projected",
                })
                total += cost
                n_new_responses += facts["n_rows"]
            elif facts["n_new"] > 0:
                cost = round(facts["n_new"] * LABEL_USD_PER_RESPONSE, 4)
                items.append({
                    "stage": "label_incr", "question_id": qid,
                    "question_text": meta["question_text"],
                    "detail": f"{facts['n_new']} new responses at the measured "
                              f"rate; rows the taxonomy can't place (count known "
                              f"only at run time, at most {facts['n_new']}) add "
                              f"one proposal call and a relabel of that pool",
                    "responses": facts["n_new"],
                    "est_cost_usd": cost,
                    "basis": "projected",
                })
                total += cost
                n_new_responses += facts["n_new"]
    else:
        for q in questions:
            qid = q["question_id"]
            rows, meta, _filtered = rows_by_q[qid]
            plan = induction.plan_dry_run(rows, meta, chunk_size, seed,
                                          price_in, price_out, description)
            items.append({
                "stage": "induce", "question_id": qid,
                "question_text": meta["question_text"],
                "detail": f"{plan['n_chunks']} chunk(s), ~{plan['total_calls']} calls",
                "responses": plan["responses_usable"],
                "est_cost_usd": plan["est_cost_usd"],
                "basis": "planned",
            })
            total += plan["est_cost_usd"]

            n = plan["responses_usable"]
            label_cost = round(n * LABEL_USD_PER_RESPONSE, 4)
            items.append({
                "stage": "label", "question_id": qid,
                "question_text": meta["question_text"],
                "detail": f"{n} responses at the measured rate "
                          f"(~$0.10 per 500)",
                "responses": n,
                "est_cost_usd": label_cost,
                "basis": "projected",
            })
            total += label_cost

    for key, label in (("lexicon", "keyword lexicon"),
                       ("locations", "location concepts")):
        items.append({
            "stage": key, "question_id": "", "question_text": "",
            "detail": "one grouping call, does not scale with the file",
            "responses": 0, "est_cost_usd": FLAT_STAGE_USD, "basis": "projected",
        })
        total += FLAT_STAGE_USD

    already = [q["question_id"] for q in questions
               if _taxonomy_exists(dataset_id, q["question_id"])]
    return {
        "dataset_id": str(dataset_id),
        "dataset_description": description,
        "parquet": str(parquet),
        "mode": mode,
        "n_questions": len(questions),
        "n_responses": sum(len(rows) for rows, _m, _f in rows_by_q.values()),
        "n_new_responses": n_new_responses,
        "items": items,
        "est_total_usd": round(total, 4),
        # a question that already has a taxonomy will be induced again, adding a
        # new versioned run — say so rather than letting the analyst assume the
        # button is a no-op on a processed dataset
        "questions_with_existing_taxonomy": already,
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _build_stages(dataset_id: str, parquet: Path, question_ids: list[str],
                  batch_size: int) -> list[Stage]:
    pq = str(parquet)
    stages = [
        Stage(key=f"induce:{q}", label=f"Induce taxonomy — question {q}",
              command=["-m", "scripts.induce", "--parquet", pq, "--question", str(q)])
        for q in question_ids
    ]
    stages.append(Stage(
        key="label", label="Label every response",
        # batch 60 is the reliable setting: batch 80 produced deterministic
        # malformed-JSON failures on the 30k run
        command=["-m", "scripts.label", "--parquet", pq, "--all",
                 "--batch-size", str(batch_size)]))
    _append_cheap_stages(stages, pq)
    return stages


def _append_cheap_stages(stages: list[Stage], pq: str) -> None:
    stages.append(Stage(
        key="lexicon", label="Build the keyword lexicon",
        command=["-m", "scripts.build_lexicon", "--parquet", pq]))
    stages.append(Stage(
        key="locations", label="Canonicalize place mentions",
        command=["-m", "scripts.build_locations", "--parquet", pq]))
    stages.append(Stage(
        key="summary", label="Rebuild the summary artifact",
        command=["-m", "scripts.summarize"]))


def _build_incremental_stages(dataset_id: str, parquet: Path,
                              rows_by_q: dict, batch_size: int) -> list[Stage]:
    """Stage list for a post-append run. Per question: incremental labeling
    when the question has both a taxonomy and a labels run (skipped entirely
    when nothing is new); full labeling when only the taxonomy exists; full
    induce + label when the column is brand new."""
    pq = str(parquet)
    inc_plan = _incremental_plan(dataset_id, rows_by_q)
    stages: list[Stage] = []
    for qid, facts in inc_plan.items():
        if not facts["has_taxonomy"]:
            stages.append(Stage(
                key=f"induce:{qid}", label=f"Induce taxonomy — question {qid}",
                command=["-m", "scripts.induce", "--parquet", pq,
                         "--question", str(qid)]))
        if not facts["has_labels"]:
            stages.append(Stage(
                key=f"label:{qid}", label=f"Label question {qid} (first run)",
                command=["-m", "scripts.label", "--parquet", pq,
                         "--question", str(qid), "--batch-size", str(batch_size)]))
        elif facts["n_new"] > 0:
            stages.append(Stage(
                key=f"label_incr:{qid}",
                label=f"Label {facts['n_new']} new responses — question {qid}",
                command=["-m", "scripts.label_incremental", "--parquet", pq,
                         "--question", str(qid), "--batch-size", str(batch_size)]))
    if not stages:
        raise ValueError(
            "Nothing to process incrementally — every question's responses are "
            "already labelled.")
    _append_cheap_stages(stages, pq)
    return stages


def _manifest_cost(path: Path, not_before: str) -> float | None:
    """A manifest's recorded cost, but only if this job wrote it.

    `created_utc` is compared against the job's start (both are
    `induction.utc_now()` strings, so lexicographic order is chronological). A
    manifest older than the job belongs to a previous run: counting it would
    bill this job for spend it never made — which happens the moment a stage
    fails partway through `--all`, leaving later questions holding only their
    older runs. An unstamped manifest is not counted for the same reason.
    """
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    created = str(m.get("created_utc") or "")
    if not created or created < not_before:
        return None
    return (m.get("usage") or {}).get("est_cost_usd")


def _stage_cost(job: Job, stage: Stage) -> float | None:
    """Read the cost the stage itself recorded. Deliberately not parsed from
    stdout — the manifest is the artifact of record. Only manifests written
    since this job started count (see _manifest_cost)."""
    from .summary import LABELS_DIR, LEXICON_DIR, LOCATIONS_DIR, latest_run_dir

    ds = str(job.dataset_id)
    since = job.created_utc
    try:
        if stage.key.startswith("induce:"):
            qid = stage.key.split(":", 1)[1]
            run = latest_run_dir(induction.TAXONOMY_DIR / ds / qid, "manifest.json")
            if run:
                return _manifest_cost(run / "manifest.json", since)
        if stage.key.startswith(("label:", "label_incr:")):
            # per-question labeling stage (incremental mode): the run the
            # stage just wrote is the question's latest
            qid = stage.key.split(":", 1)[1]
            run = latest_run_dir(LABELS_DIR / ds / qid, "manifest.json")
            if run:
                return _manifest_cost(run / "manifest.json", since)
        if stage.key == "label":
            total, seen = 0.0, False
            qdir = LABELS_DIR / ds
            if qdir.is_dir():
                for q in sorted(qdir.iterdir()):
                    run = latest_run_dir(q, "manifest.json")
                    if run:
                        c = _manifest_cost(run / "manifest.json", since)
                        if c is not None:
                            total += c
                            seen = True
            return round(total, 6) if seen else None
        if stage.key in {"lexicon", "locations"}:
            root = LEXICON_DIR if stage.key == "lexicon" else LOCATIONS_DIR
            return _manifest_cost(root / ds / "manifest.json", since)
    except (ValueError, OSError):
        return None
    return None


def _run_stage(job: Job, stage: Stage, log_path: Path) -> bool:
    """Run one CLI stage. Returns False if the job should stop."""
    stage.status = "running"
    _write_status(job)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\n$ python {' '.join(stage.command)}\n{'=' * 70}\n")
        log.flush()
        try:
            proc = subprocess.run(
                [sys.executable, *stage.command],
                cwd=str(BACKEND_DIR), stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
            code = proc.returncode
        except OSError as exc:
            stage.status = "failed"
            stage.error = f"could not start: {exc}"
            stage.seconds = round(time.time() - started, 1)
            _write_status(job)
            return False

    stage.seconds = round(time.time() - started, 1)
    stage.cost_usd = _stage_cost(job, stage)
    if code != 0:
        stage.status = "failed"
        # the log holds the traceback; surface its tail so the UI says
        # something more useful than "exit 1"
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-600:]
        except OSError:
            pass
        stage.error = f"exit code {code}"
        stage.detail = tail.strip().splitlines()[-1] if tail.strip() else ""
        _write_status(job)
        return False
    stage.status = "done"
    _write_status(job)
    return True


def _worker(job: Job) -> None:
    global _ACTIVE
    log_path = _job_dir(job.dataset_id, job.job_id) / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for stage in job.stages:
            if job.cancel_requested:
                for s in job.stages:
                    if s.status == "pending":
                        s.status = "skipped"
                job.status = "failed"
                job.error = "cancelled"
                break
            if not _run_stage(job, stage, log_path):
                # a failed stage stops the run: labeling against a missing
                # taxonomy, or asking against missing labels, would just fail
                # again further down with a less obvious message
                for s in job.stages:
                    if s.status == "pending":
                        s.status = "skipped"
                job.status = "failed"
                job.error = f"{stage.label}: {stage.error}"
                break
        else:
            job.status = "done"
    except Exception as exc:                      # never leave a job "running"
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished_utc = induction.utc_now()
        _write_status(job)
        with _LOCK:
            _ACTIVE = None


def start_job(dataset_id: str, batch_size: int = 60, mode: str = "full") -> Job:
    """Kick off the pipeline. Raises RuntimeError if one is already running —
    the API turns that into a 409 rather than queueing, so the analyst is never
    surprised by a second billed run they didn't watch start."""
    global _ACTIVE
    if mode not in {"full", "incremental"}:
        raise ValueError(f"Unknown pipeline mode: {mode!r}")
    parquet = ask_service.dataset_parquet(dataset_id)
    questions = induction.list_questions(parquet)
    if not questions:
        raise ValueError(f"Dataset {dataset_id} has no questions in its export")

    if mode == "incremental":
        rows_by_q = induction.load_questions_bulk(
            parquet, [q["question_id"] for q in questions])
        stages = _build_incremental_stages(dataset_id, parquet, rows_by_q,
                                           batch_size)
    else:
        stages = _build_stages(dataset_id, parquet,
                               [q["question_id"] for q in questions], batch_size)

    with _LOCK:
        if _ACTIVE is not None:
            running = _JOBS.get(_ACTIVE)
            raise RuntimeError(
                f"A pipeline run is already in progress for dataset "
                f"{running.dataset_id if running else '?'} (job {_ACTIVE}). "
                f"Wait for it to finish.")
        job = Job(
            job_id=f"{induction.utc_now()}_{uuid.uuid4().hex[:6]}",
            dataset_id=str(dataset_id),
            created_utc=induction.utc_now(),
            stages=stages,
        )
        _JOBS[job.job_id] = job
        _ACTIVE = job.job_id
    _write_status(job)
    threading.Thread(target=_worker, args=(job,), daemon=True,
                     name=f"pipeline-{job.job_id}").start()
    return job


def get_job(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def latest_job(dataset_id: str) -> Job | None:
    """Most recent job for this dataset in this process. Jobs from before a
    restart are not resurrected — their status.json on disk is the record."""
    jobs = [j for j in _JOBS.values() if j.dataset_id == str(dataset_id)]
    return max(jobs, key=lambda j: j.created_utc) if jobs else None


def cancel_job(job_id: str) -> bool:
    """Ask a job to stop after its current stage. The running subprocess is
    left to finish — killing a stage mid-write is how you get a half-written
    assignments.json that later reads as real data."""
    job = _JOBS.get(job_id)
    if job is None or job.status != "running":
        return False
    job.cancel_requested = True
    _write_status(job)
    return True


def is_processed(dataset_id: str) -> bool:
    """Whether the Ask tab will work for this dataset."""
    from .summary import LABELS_DIR

    ds_dir = LABELS_DIR / str(dataset_id)
    if not ds_dir.is_dir():
        return False
    return any((d / "assignments.json").exists()
               for q in ds_dir.iterdir() if q.is_dir()
               for d in q.iterdir() if d.is_dir())
