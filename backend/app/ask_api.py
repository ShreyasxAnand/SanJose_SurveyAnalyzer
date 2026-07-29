"""Phase 5 API: the two-step ask flow over HTTP.

Stateless by design (Architecture 1): the server keeps nothing between the
two calls. Step 1 returns a routing proposal the browser holds while the
analyst edits checkboxes; step 2 receives back only *choices* — label ids
and concept names — and recomputes every count and quote server-side from
the artifacts on disk. A client cannot inject a number, a quote, or an id
the taxonomy doesn't contain (unknown ids are a 422, the same server-side
gate philosophy as question wording at ingest).

Both endpoints are thin wrappers over app.ask_service, which scripts.ask
also uses — browser and terminal run the same pipeline and leave the same
audit trail under data/answers/{dataset_id}/{run_id}/.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from . import ask_service, induction
from .llm import DEFAULT_SYNTH_MODEL, GeminiClient
from .schemas import (
    AskAnswerRequest,
    AskAnswerResponse,
    AskAnswerStats,
    AskCandidateOut,
    AskGroupCountOut,
    AskLexiconCountOut,
    AskLocationOut,
    AskRouteRequest,
    AskRouteResponse,
    AskSourceOut,
)

router = APIRouter(prefix="/datasets/{dataset_id}/ask", tags=["ask"])


def _dataset_description(dataset_id: str) -> str:
    """Dataset row first (source of truth), export manifest second, the
    deprecated legacy constant last."""
    try:
        from .db import SessionLocal
        from .models import Dataset

        with SessionLocal() as db:
            dataset = db.get(Dataset, int(dataset_id))
            if dataset is not None and (dataset.description or "").strip():
                return dataset.description.strip()
    except Exception:
        pass
    try:
        from .db import EXPORTS_DIR

        return induction.resolve_description(
            None, EXPORTS_DIR / str(dataset_id) / "responses.parquet")
    except Exception:
        return induction.DEFAULT_DATASET_DESCRIPTION


def _load_context(dataset_id: str, description: str) -> ask_service.AskContext:
    try:
        return ask_service.load_context(dataset_id, description=description)
    except FileNotFoundError as exc:
        # no labels/taxonomy/parquet yet — the pipeline hasn't run for this
        # dataset, which is a client-visible state, not a server fault
        raise HTTPException(status_code=409, detail=str(exc))


def _client() -> GeminiClient:
    try:
        return GeminiClient()
    except RuntimeError as exc:      # missing API key
        raise HTTPException(status_code=503, detail=str(exc))


def _synth_client() -> GeminiClient:
    """The answer-writing model — one call per question, so it runs on the
    stronger model while routing and the corpus-scale stages stay on the
    workhorse. Override with GEMINI_SYNTH_MODEL."""
    try:
        return GeminiClient(model=os.environ.get("GEMINI_SYNTH_MODEL",
                                                 DEFAULT_SYNTH_MODEL))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _candidate_out(c: dict, ctx: ask_service.AskContext) -> AskCandidateOut:
    e = ctx.index[c["label_id"]]
    q_text = next((q["question_text"] for q in ctx.summary["questions"]
                   if q["question_id"] == e["question_id"]), "")
    return AskCandidateOut(
        label_id=c["label_id"],
        name=e["name"],
        parent_name=e.get("parent_name"),
        question_id=e["question_id"],
        question_text=q_text,
        count=len(ctx.members.get(c["label_id"], [])),
        relevance=c["relevance"],
        rationale=c["rationale"],
    )


@router.post("/route", response_model=AskRouteResponse)
def ask_route(dataset_id: str, req: AskRouteRequest) -> AskRouteResponse:
    """Step 1: propose categories. One model call. An unanswerable question
    is a valid 200 with zero candidates — empty results are correct results."""
    description = _dataset_description(dataset_id)
    ctx = _load_context(dataset_id, description)
    client = _client()
    try:
        route, stats = ask_service.propose(client, req.question, ctx, description)
    except RuntimeError as exc:      # Gemini failure after retries
        raise HTTPException(status_code=502, detail=str(exc))

    warnings = list(stats["warnings"])
    if stats["invalid_label_ids"]:
        warnings.append(f"{stats['invalid_label_ids']} invented label id(s) "
                        "dropped from the proposal")
    return AskRouteResponse(
        answerable=route["answerable"],
        route=route["route"],
        reason=route["reason"],
        candidates=[_candidate_out(c, ctx) for c in route["candidates"]],
        lexicon_concepts=route["lexicon_concepts"],
        available_lexicon_concepts=sorted(ctx.valid_concepts),
        group_by=route["group_by"],
        location_filter=route["location_filter"],
        available_locations=sorted(
            (AskLocationOut(name=name,
                            kind=ctx.location_kinds.get(name, "type"),
                            count=len(keys))
             for name, keys in ctx.location_members.items()),
            key=lambda l: (-l.count, l.name)),
        actionability_filter=route["actionability_filter"],
        available_actionability=ctx.actionability_counts,
        event_filter=route["event_filter"],
        available_events=({"reported": len(ctx.events),
                           "coded": len(ctx.event_coded)}
                          if ctx.events else {}),
        warnings=warnings,
    )


@router.post("/answer", response_model=AskAnswerResponse)
def ask_answer(dataset_id: str, req: AskAnswerRequest) -> AskAnswerResponse:
    """Step 2: synthesize from the analyst-approved selection. One model
    call. Counts and quotes are recomputed here — the request carries
    choices, never evidence."""
    description = _dataset_description(dataset_id)
    ctx = _load_context(dataset_id, description)
    try:
        route = ask_service.route_from_selection(
            [c.model_dump() for c in req.selected],
            req.route, req.reason, req.lexicon_concepts, ctx,
            group_by=req.group_by, location_filter=req.location_filter,
            actionability_filter=req.actionability_filter,
            event_filter=req.event_filter)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    client = _client()
    try:
        result = ask_service.answer(
            client, req.question, route, ctx,
            description=description,
            proposed_label_ids=req.proposed_label_ids or None,
            synth_client=_synth_client())
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    selection = result["selection"] or {"deselected": [], "added": []}
    return AskAnswerResponse(
        run_id=result["run_id"],
        answer_markdown=result["answer_body"],
        process_note=result["process_note"],
        stats=AskAnswerStats(**result["stats"]),
        sources=[AskSourceOut(**q) for q in result["evidence"]["quotes"]
                 if q["n"] in {c["n"] for c in result["cited"]}],
        counts={s["label_id"]: s["count"] for s in result["evidence"]["selection"]},
        counts_unfiltered={s["label_id"]: s["count_unfiltered"]
                           for s in result["evidence"]["selection"]
                           if "count_unfiltered" in s},
        sampling_notes=result["evidence"]["sampling_notes"],
        lexicon_counts=[AskLexiconCountOut(**lc)
                        for lc in result["lexicon_counts"]],
        group_counts=[AskGroupCountOut(name=g["name"],
                                       count_unique_responses=g["count_unique_responses"])
                      for g in result["evidence"]["group_counts"]],
        location_filter=result["evidence"].get("location_filter") or [],
        location_counts=[AskLocationOut(**lc)
                         for lc in result["evidence"].get("location_counts") or []],
        location_denominator=result["evidence"].get("location_denominator"),
        actionability_filter=result["evidence"].get("actionability_filter") or "",
        actionability_denominator=result["evidence"].get("actionability_denominator"),
        event_filter=result["evidence"].get("event_filter") or "",
        event_denominator=result["evidence"].get("event_denominator"),
        invalid_citations=result["invalid_citations"],
        deselected=selection["deselected"],
        added=selection["added"],
    )
