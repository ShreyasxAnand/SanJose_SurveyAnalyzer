"""Run the induce -> label -> subthemes -> lexicon -> locations pipeline from
the browser.

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
free code — prompt tokens come from Vertex's countTokens, which does not run
the model, and everything else from rates `calibration` measured over runs
that already finished here. The analyst approves a figure before the first
billed call. Which figures are *planned* and which are *projected* is marked
per row and must stay marked: a projection presented as a plan is exactly the
kind of invented number this project refuses to produce.

**The sub-theme pass is part of processing.** It was CLI-only for a while,
which meant a dataset processed by this button answered with less depth than
one processed from a terminal, and its cost was invisible in the plan. It runs
after labeling (it reads the latest labels run to find which categories
cleared the member floor) and costs nothing on a corpus where no category
clears it.

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

from . import (ask_service, calibration, induction, labeling, llm,
               subthemes, tokens)

BACKEND_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = induction.DATA_DIR / "jobs"

# countTokens round-trips one whole estimate may spend, across every question
# and all three per-question stages. A person is waiting on a free planning
# step, so this trades latency for precision: ~0.15s per call, and the shared
# ratio pool means a stage that runs out of budget is still measured rather
# than dropped to characters/4. Sized for a five-question dataset paying four
# calls for the first stage and two for each of the other fourteen.
COUNT_CALL_BUDGET = 36

# The batch size the run will really use — `_build_stages` passes 60 to
# scripts.label, so the estimate must plan the same batching or its batch
# count (and therefore its call count) is wrong. Batch 80 produced
# deterministic malformed-JSON failures on the 30k run; 60 is the setting.
LABEL_BATCH_SIZE = 60

# What `scripts.subthemes` defaults to, and therefore what the estimate must
# plan against for its batch count to match the run.
SUBTHEME_BATCH_SIZE = subthemes.DEFAULT_BATCH_SIZE

# Superseded by `calibration`, which learns per-response token rates from this
# install's own completed runs and prices them at the configured rate. Kept
# because a caller may still want the old flat figure, and because it is the
# number every older estimate in the job history was built on.
#
# It was measured once, on the test dataset and the 30k production run: ~$0.10
# per 500 responses. Pooled over every production labeling run since, the real
# figure is ~$0.00012 per response — the old constant runs about 1.7x high,
# which is most of the "~25% high" the UI used to warn about.
LABEL_USD_PER_RESPONSE = 0.10 / 500
# Likewise superseded by `calibration.flat_stage_usd`, which interpolates the
# real cost of these two stages against corpus size. One grouping call each,
# bounded by MAX_CANDIDATES / MAX_SPANS rather than by corpus size — but "does
# not scale with the file" turned out to be true only for the lexicon: the
# locations stage tracks the number of distinct place spans, and on the 30k
# corpus it cost $0.0141, fourteen times this constant.
FLAT_STAGE_USD = 0.001


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


def _new_rows(dataset_id: str, question_id: str, rows: list) -> list:
    """The rows this question's latest labels run has never seen.

    The same filter `_incremental_plan` counts, but returning the rows
    themselves so their real prompts can be built and counted rather than
    multiplied by a rate.
    """
    prior = _prior_label_keys(dataset_id, question_id)
    if prior is None:
        return list(rows)
    return [r for r in rows if r.response_key not in prior]


def _latest_taxonomy(dataset_id: str, question_id: str) -> dict | None:
    """The question's newest taxonomy, for building real labeling prompts.

    Returns None when there is none — which is the normal case for a fresh
    dataset, where induction has not run yet and the labeling estimate must
    fall back to a measured rate.
    """
    from .summary import latest_run_dir

    run = latest_run_dir(
        induction.TAXONOMY_DIR / str(dataset_id) / str(question_id),
        "candidate_taxonomy.json")
    if run is None:
        return None
    try:
        parsed = json.loads(
            (run / "candidate_taxonomy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _label_plan_item(rows, meta, qid: str, dataset_id: str, description: str,
                     batch_size: int, counter, cal, prices) -> dict:
    """One labeling row of the estimate, with the best basis available.

    When the question already has a taxonomy the real batch prompts get built
    and counted; otherwise the token counts come from the calibrated
    per-unique-response rates. Either way the row reports which it was, and
    either way the dollars come from the configured price for the configured
    model.
    """
    taxonomy = _latest_taxonomy(dataset_id, qid)
    plan = labeling.plan_labeling(
        [(r.response_key, r.text) for r in rows], taxonomy,
        batch_size=batch_size, dataset_description=description,
        counter=counter, cal=cal)
    dup = plan["n_responses"] - plan["n_unique"]
    detail = (f"{plan['n_responses']} responses"
              + (f" ({plan['n_unique']} distinct texts, {dup} duplicates "
                 f"labelled once)" if dup else "")
              + f", {plan['n_batches']} batches")
    if plan["input_basis"] == "projected":
        detail += " — no taxonomy yet, tokens from the measured rate"
    return _priced_item(
        stage="label", question_id=qid, question_text=meta["question_text"],
        detail=detail, responses=plan["n_responses"], plan=plan, prices=prices)


def _latest_label_facts(dataset_id: str, question_id: str
                        ) -> tuple[dict[str, int], list[dict]]:
    """`(label_counts, assignments)` from the question's latest labels run.

    Both empty when there is no run — the fresh-dataset case, where sub-theme
    eligibility cannot be known because labeling has not happened yet.
    """
    from .summary import LABELS_DIR, latest_run_dir

    run = latest_run_dir(LABELS_DIR / str(dataset_id) / str(question_id),
                         "assignments.json")
    if run is None:
        return {}, []
    try:
        assignments = json.loads(
            (run / "assignments.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    if not isinstance(assignments, list):
        return {}, []
    counts: dict[str, int] = {}
    for a in assignments:
        for label_id in (a.get("label_ids") or []):
            counts[label_id] = counts.get(label_id, 0) + 1
    return counts, assignments


def _subtheme_plan_item(rows, meta, qid: str, dataset_id: str,
                        description: str, counter, cal, prices) -> dict | None:
    """One sub-theme row of the estimate, or None when the pass would do
    nothing at all for this question.

    Returning None rather than a zero row is deliberate: a question too small
    to have any eligible category is not "$0.00 of sub-theming", it is a
    stage that does not apply, and a table full of zero rows buries the ones
    that cost money.
    """
    taxonomy = _latest_taxonomy(dataset_id, qid)
    counts, assignments = _latest_label_facts(dataset_id, qid)
    plan = subthemes.plan_subthemes(
        rows, taxonomy if counts else None, counts or None, assignments,
        question_text=meta["question_text"], dataset_description=description,
        batch_size=SUBTHEME_BATCH_SIZE, counter=counter, cal=cal)
    if plan["total_calls"] <= 0:
        return None
    return _priced_item(
        stage="subthemes", question_id=qid,
        question_text=meta["question_text"], detail=plan["detail"],
        responses=plan["n_members"], plan=plan, prices=prices)


def _priced_item(*, stage: str, question_id: str, question_text: str,
                 detail: str, responses: int, plan: dict, prices) -> dict:
    """Turn a token plan into a priced estimate row."""
    tokens_in = int(plan["est_input_tokens"])
    tokens_out = int(plan["est_output_tokens"])
    cost = prices.usd(tokens_in, tokens_out)
    return {
        "stage": stage,
        "question_id": question_id,
        "question_text": question_text,
        "detail": detail,
        "responses": responses,
        "est_input_tokens": tokens_in,
        "est_output_tokens": tokens_out,
        "est_cost_usd": round(cost, 4),
        # "planned" when the prompts that drive it were actually built and
        # counted; "projected" when a measured rate stood in for them
        "basis": "planned" if plan["input_basis"] in {"counted", "sampled"}
                 else "projected",
        "input_basis": plan["input_basis"],
    }


class _Prices:
    """The configured rate for the configured model, or none at all."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        rate = llm.prices_per_mtok().get(model_id)
        self.known = rate is not None
        self.price_in, self.price_out = rate if rate else (0.0, 0.0)

    def usd(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1e6 * self.price_in + tokens_out / 1e6 * self.price_out


def estimate_dataset(dataset_id: str, chunk_size: int = 120, seed: int = 7,
                     price_in: float | None = None,
                     price_out: float | None = None,
                     description: str = "", mode: str = "full") -> dict:
    """The plan the analyst approves. Runs only free code.

    Nothing here is billed: prompt tokens are measured with Vertex's
    countTokens, which does not run the model, and everything else comes from
    completed runs' manifests via `calibration`.

    **`basis` still means what it always meant** — "planned" when the figure
    came from prompts this run will really send, "projected" when a measured
    rate stood in for them — but more rows can now earn "planned". Labeling
    against an existing taxonomy is counted rather than extrapolated, because
    the prompts are fully determined. Labeling on a fresh dataset still cannot
    be: its taxonomy is produced by the stage before it.

    In incremental mode every labeling row stays "projected" regardless. The
    new rows' prompts are countable, but rows the existing taxonomy cannot
    place add a proposal call and a relabel of that pool, and how many there
    are is a run-time fact. A number that omits a step it knows may happen is
    not a plan.

    `price_in` / `price_out` override the configured rate for the resolved
    model; leaving them None — which is what the API does — reads
    config.json, so re-pricing is a settings edit.
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

    cal = calibration.active()
    model_id = llm.resolve_model()
    prices = _Prices(model_id)
    if price_in is not None and price_out is not None:
        prices.known = True
        prices.price_in, prices.price_out = price_in, price_out
    # One counter for the whole estimate so its call budget is spent across
    # every question rather than exhausted on the first.
    counter = tokens.TokenCounter(model_id, max_calls=COUNT_CALL_BUDGET)

    items: list[dict] = []
    n_new_responses = 0
    if mode == "incremental":
        inc_plan = _incremental_plan(dataset_id, rows_by_q)
        for q in questions:
            qid = q["question_id"]
            rows, meta, _filtered = rows_by_q[qid]
            facts = inc_plan[qid]
            if not facts["has_taxonomy"]:
                # a newly selected column: full induce + full label, as ever
                plan = induction.plan_dry_run(
                    rows, meta, chunk_size, seed, prices.price_in,
                    prices.price_out, description, counter=counter, cal=cal)
                items.append(_priced_item(
                    stage="induce", question_id=qid,
                    question_text=meta["question_text"],
                    detail=f"no taxonomy yet — full induction, "
                           f"{plan['n_chunks']} chunk(s)",
                    responses=plan["responses_usable"], plan=plan,
                    prices=prices))
            if not facts["has_labels"]:
                item = _label_plan_item(
                    rows, meta, qid, dataset_id, description,
                    LABEL_BATCH_SIZE, counter, cal, prices)
                item["detail"] = f"no labels run yet — {item['detail']}"
                # the pool step below makes any incremental labeling figure a
                # floor, not a plan
                item["basis"] = "projected"
                items.append(item)
                n_new_responses += facts["n_rows"]
            elif facts["n_new"] > 0:
                new_rows = _new_rows(dataset_id, qid, rows)
                item = _label_plan_item(
                    new_rows, meta, qid, dataset_id, description,
                    LABEL_BATCH_SIZE, counter, cal, prices)
                item["stage"] = "label_incr"
                item["detail"] = (
                    f"{item['detail']}; rows the taxonomy can't place (count "
                    f"known only at run time, at most {facts['n_new']}) add one "
                    f"proposal call and a relabel of that pool")
                item["basis"] = "projected"
                items.append(item)
                n_new_responses += facts["n_new"]
            sub = _subtheme_plan_item(rows, meta, qid, dataset_id,
                                      description, counter, cal, prices)
            if sub is not None:
                sub["basis"] = "projected"
                sub["detail"] += "; already-covered categories are carried, not redone"
                items.append(sub)
    else:
        for q in questions:
            qid = q["question_id"]
            rows, meta, _filtered = rows_by_q[qid]
            plan = induction.plan_dry_run(
                rows, meta, chunk_size, seed, prices.price_in,
                prices.price_out, description, counter=counter, cal=cal)
            items.append(_priced_item(
                stage="induce", question_id=qid,
                question_text=meta["question_text"],
                detail=f"{plan['n_chunks']} chunk(s), ~{plan['total_calls']} calls",
                responses=plan["responses_usable"], plan=plan, prices=prices))
            items.append(_label_plan_item(
                rows, meta, qid, dataset_id, description, LABEL_BATCH_SIZE,
                counter, cal, prices))
            sub = _subtheme_plan_item(rows, meta, qid, dataset_id,
                                      description, counter, cal, prices)
            if sub is not None:
                # always projected in full mode: the taxonomy this pass groups
                # inside is produced by the induce stage above it, so today's
                # categories are not the ones it will see
                sub["basis"] = "projected"
                items.append(sub)

    n_responses = sum(len(rows) for rows, _m, _f in rows_by_q.values())
    for key, label in (("lexicon", "keyword lexicon"),
                       ("locations", "location concepts")):
        cost = round(cal.flat_stage_usd(key, n_responses), 4)
        items.append({
            "stage": key, "question_id": "", "question_text": "",
            "detail": f"one grouping call over the whole dataset, "
                      f"{cal.source} rate for a corpus this size",
            "responses": 0,
            "est_input_tokens": 0, "est_output_tokens": 0,
            "est_cost_usd": cost,
            "basis": "projected", "input_basis": "projected",
        })

    total = sum(i["est_cost_usd"] for i in items)
    low, high = cal.spread
    already = [q["question_id"] for q in questions
               if _taxonomy_exists(dataset_id, q["question_id"])]
    counted = sum(1 for i in items if i["input_basis"] in {"counted", "sampled"})
    return {
        "dataset_id": str(dataset_id),
        "dataset_description": description,
        "parquet": str(parquet),
        "mode": mode,
        "n_questions": len(questions),
        "n_responses": n_responses,
        "n_new_responses": n_new_responses,
        "items": items,
        "est_total_usd": round(total, 4),
        # The honest width of "about this much": the 10th-90th percentile
        # spread of per-run rates around the pooled one, from the same
        # manifests the rates came from. Not a guarantee — a band.
        "est_low_usd": round(total * low, 4),
        "est_high_usd": round(total * high, 4),
        "model_id": model_id,
        # tokens are known even when the model has no configured price; say so
        # rather than showing $0.00 as if it were free
        "priced": prices.known,
        "est_input_tokens": sum(i["est_input_tokens"] for i in items),
        "est_output_tokens": sum(i["est_output_tokens"] for i in items),
        "calibration_source": cal.source,
        "calibration_measured_utc": cal.measured_utc,
        "count_calls": counter.calls_made,
        "stages_counted": counted,
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
    _append_subtheme_stage(stages, pq)
    _append_cheap_stages(stages, pq)
    return stages


def _append_subtheme_stage(stages: list[Stage], pq: str) -> None:
    """The sub-theme pass — the second level inside each large category.

    Must come after labeling: it reads the latest labels run to find which
    categories cleared the member floor, and refuses to run without one. It
    is a no-op on a corpus where nothing clears the floor, which is why a
    small dataset can carry this stage and still spend nothing on it.

    `ask_service` has loaded this layer since the sub-theme answer work
    landed; until now nothing in the browser produced it, so a dataset
    processed by the button answered without the depth a CLI-processed one
    had. Estimating it without running it would have been the same gap with
    a bigger number attached.
    """
    stages.append(Stage(
        key="subthemes", label="Build sub-themes inside large categories",
        command=["-m", "scripts.subthemes", "--parquet", pq, "--all"]))


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
    # scripts.subthemes carries by default: categories whose sub-themes are
    # still valid are reused rather than re-induced, so this is cheap after an
    # append and expensive only where new rows pushed a category over the
    # floor for the first time. That is exactly the work an append creates.
    _append_subtheme_stage(stages, pq)
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
