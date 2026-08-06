"""Pipeline endpoints: estimate a run, start it, poll it.

Three calls, matching the two-step-with-a-gate shape the rest of the app uses:

  POST /datasets/{id}/pipeline/estimate   free; the plan the analyst approves
  POST /datasets/{id}/pipeline/run        starts the background job
  GET  /datasets/{id}/pipeline/status     poll while it runs

`estimate` makes no model calls and writes nothing, so the analyst always sees
a figure before any spend. `run` is a 409 if a job is already in flight —
queueing a second billed run nobody watched start would be worse than refusing.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import pipeline
from .schemas import (
    PipelineEstimate,
    PipelineJobOut,
    PipelineRunRequest,
    PipelineStageOut,
)

router = APIRouter(prefix="/datasets/{dataset_id}/pipeline", tags=["pipeline"])


def _job_out(job: pipeline.Job) -> PipelineJobOut:
    return PipelineJobOut(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        status=job.status,
        stages=[PipelineStageOut(**{k: v for k, v in vars(s).items()
                                    if k != "command"})
                for s in job.stages],
        created_utc=job.created_utc,
        finished_utc=job.finished_utc,
        error=job.error,
        cost_usd=job.cost_usd,
        is_processed=pipeline.is_processed(job.dataset_id),
    )


@router.post("/estimate", response_model=PipelineEstimate)
def pipeline_estimate(dataset_id: str, mode: str = "full") -> PipelineEstimate:
    """The plan, for free. No API calls, nothing written. `mode=incremental`
    (after an append) plans only the never-labeled rows."""
    try:
        return PipelineEstimate(**pipeline.estimate_dataset(dataset_id, mode=mode))
    except FileNotFoundError as exc:
        # no export yet — the dataset hasn't finished ingest
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/run", response_model=PipelineJobOut)
def pipeline_run(dataset_id: str, req: PipelineRunRequest) -> PipelineJobOut:
    try:
        job = pipeline.start_job(dataset_id, batch_size=req.batch_size,
                                 mode=req.mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:            # one job at a time, process-wide
        raise HTTPException(status_code=409, detail=str(exc))
    return _job_out(job)


@router.get("/status", response_model=PipelineJobOut | None)
def pipeline_status(dataset_id: str) -> PipelineJobOut | None:
    """The most recent job for this dataset in this process, or null if none
    has run since the server started. A null with `is_processed` true means the
    pipeline was run earlier (or from the CLI) — the UI reads
    /datasets/{id}/pipeline/processed for that."""
    job = pipeline.latest_job(dataset_id)
    return _job_out(job) if job else None


@router.get("/processed")
def pipeline_processed(dataset_id: str) -> dict:
    """Whether the Ask tab will work for this dataset, independent of whether
    any job ran in this process — labels created from the CLI count."""
    return {"dataset_id": str(dataset_id),
            "is_processed": pipeline.is_processed(dataset_id)}


@router.post("/cancel", response_model=PipelineJobOut)
def pipeline_cancel(dataset_id: str) -> PipelineJobOut:
    """Stop after the current stage. The running stage is allowed to finish —
    killing it mid-write is how a half-written assignments.json ends up looking
    like real data."""
    job = pipeline.latest_job(dataset_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No pipeline job for this dataset")
    if not pipeline.cancel_job(job.job_id):
        raise HTTPException(status_code=409,
                            detail=f"Job {job.job_id} is not running ({job.status})")
    return _job_out(job)
