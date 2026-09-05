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
from .llm import DEFAULT_SYNTH_MODEL, GeminiClient, resolve_synth_model  # noqa: F401
from . import ask_cache
from .schemas import (
    AskAnswerRequest,
    AskAnswerResponse,
    AskAnswerStats,
    AskCandidateOut,
    AskChildOut,
    AskDemographicOut,
    AskDemographicsRequest,
    AskDemographicsResponse,
    AskDemographicValueOut,
    AskGroupCountOut,
    AskLexiconCountOut,
    AskLocationOut,
    AskParentGroupOut,
    AskQuestionOut,
    AskRouteRequest,
    AskRouteResponse,
    AskSubBreakdownOut,
    AskSubCountOut,
    AskUncoveredOut,
    AskSourceOut,
)

# Matches summary.MAX_DESC_CHARS — the review screen shows a category's
# description so an analyst can judge an unproposed one, and a 350-category
# tree does not need full descriptions on the wire.
MAX_DESC_CHARS = 200

router = APIRouter(prefix="/datasets/{dataset_id}/ask", tags=["ask"])


def _dataset_description(dataset_id: str) -> str:
    """Dataset row first (source of truth), export manifest second, the
    deprecated legacy constant last. The survey's fielding window, when
    recorded at ingest, is appended — it flows into every ask prompt via
    {dataset_context}, so answers can anchor claims to WHEN residents said
    this instead of a timeless present."""
    try:
        from .db import SessionLocal
        from .models import Dataset

        with SessionLocal() as db:
            dataset = db.get(Dataset, int(dataset_id))
            if dataset is not None and (dataset.description or "").strip():
                desc = dataset.description.strip()
                start = (dataset.survey_start_date or "").strip() \
                    if isinstance(dataset.survey_start_date, str) \
                    else dataset.survey_start_date
                end = (dataset.survey_end_date or "").strip() \
                    if isinstance(dataset.survey_end_date, str) \
                    else dataset.survey_end_date
                if start or end:
                    window = (f"{start} to {end}" if start and end
                              else f"{start or end}")
                    desc += f" Survey fielded {window}."
                return desc
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


# The ask path uses llm.py's defaults (240s, 6 attempts) — deliberately, after
# trying twice to be clever and making it worse both times.
#
# Gemini's failure mode here is a response that is SLOW, not one that never
# arrives: a ROUTE call measured 169.9s and then SUCCEEDED, against a ~1.5s
# median. Under the 240s default that call simply completed, and the analyst
# waited. Shortening the timeout turned those successes into failures:
#   * flat 30s x 3 -> three aborted attempts and a 502 the analyst actually hit;
#   * escalating 35/90/120 -> a "TimeoutError ... attempt 1/3" on every slow
#     call, because a 35s first attempt cuts off the common slow case.
# A short timeout only helps when the connection is DEAD, which is not what
# happens. Waiting is the correct behaviour; the logging below is what makes
# waiting tolerable, because now it says so instead of looking frozen.
def _client() -> GeminiClient:
    try:
        return GeminiClient()
    except RuntimeError as exc:      # missing/expired Google credentials (ADC)
        raise HTTPException(status_code=503, detail=str(exc))


def _synth_client() -> GeminiClient:
    """The answer-writing model — one call per question, so it runs on the
    stronger model while routing and the corpus-scale stages stay on the
    workhorse. Override with GEMINI_SYNTH_MODEL or config.json's
    `models.synth`."""
    try:
        return GeminiClient(model=resolve_synth_model())
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


NO_PARENT = "Ungrouped"


def _available_categories(route: dict,
                          ctx: ask_service.AskContext) -> list[AskParentGroupOut]:
    """The full taxonomy as question > parent > child, with the router's
    proposal marked in place.

    Ordering is what makes a 350-category tree usable: parents holding a
    proposed category come first (the proposal stays the thing the analyst
    reads), then the largest parents. Children sort proposed-first then by
    count. Nothing is hidden — the analyst can reach every category.
    """
    proposed = {c["label_id"]: c for c in route["candidates"]}
    groups: list[AskParentGroupOut] = []
    for q in ctx.summary["questions"]:
        by_parent: dict[str, list[dict]] = {}
        for e in q["entries"]:
            by_parent.setdefault(e.get("parent_name") or NO_PARENT, []).append(e)
        for parent_name, entries in by_parent.items():
            children = []
            keys: set[str] = set()
            for e in sorted(entries, key=lambda e: (e["label_id"] not in proposed,
                                                    -e["count"],
                                                    e["name"].lower())):
                p = proposed.get(e["label_id"])
                desc = e.get("description") or ""
                children.append(AskChildOut(
                    label_id=e["label_id"],
                    name=e["name"],
                    count=len(ctx.members.get(e["label_id"], [])),
                    description=(desc[:MAX_DESC_CHARS] + "…"
                                 if len(desc) > MAX_DESC_CHARS else desc),
                    proposed=p is not None,
                    relevance=p["relevance"] if p else "",
                    rationale=p["rationale"] if p else "",
                ))
                # union, not a sum: one response carrying two children of the
                # same parent is one response
                keys.update(ctx.members.get(e["label_id"], []))
            groups.append(AskParentGroupOut(
                question_id=q["question_id"],
                question_text=q["question_text"],
                parent_name=parent_name,
                count_unique_responses=len(keys),
                n_proposed=sum(1 for c in children if c.proposed),
                children=children,
            ))
    groups.sort(key=lambda g: (g.n_proposed == 0, -g.n_proposed,
                               -g.count_unique_responses, g.parent_name.lower()))
    return groups


@router.get("/questions", response_model=list[AskQuestionOut])
def ask_questions(dataset_id: str) -> list[AskQuestionOut]:
    """The dataset's survey questions, for the scope selector on the ask
    form — asked before any routing happens."""
    ctx = _load_context(dataset_id, _dataset_description(dataset_id))
    return [AskQuestionOut(question_id=q["question_id"],
                           question_text=q.get("question_text", ""),
                           n_responses=q["n_responses"])
            for q in ctx.summary["questions"]]


@router.post("/demographics", response_model=AskDemographicsResponse)
def ask_demographics(dataset_id: str,
                     req: AskDemographicsRequest) -> AskDemographicsResponse:
    """The dataset's demographic fields and values for the respondent filter
    on the ask form — same role /questions plays for the scope selector, but
    POST because the counts are faceted: send the current selection and each
    field comes back recounted under the OTHER fields' ticked values, so the
    dropdowns always show what a tick would actually leave. Empty fields list
    = no demographics, which is how the UI knows to keep the control
    disabled."""
    ctx = _load_context(dataset_id, _dataset_description(dataset_id))
    try:
        demo = ask_service.validate_demographic_filter(
            req.demographic_filter, ctx)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    fields, n_matching = ask_service.facet_demographics(ctx, demo)
    return AskDemographicsResponse(
        fields=[AskDemographicOut(
                    field=f,
                    values=[AskDemographicValueOut(value=v, n_respondents=n)
                            for v, n in vals])
                for f, vals in fields],
        n_matching_respondents=n_matching)


@router.post("/route", response_model=AskRouteResponse)
def ask_route(dataset_id: str, req: AskRouteRequest) -> AskRouteResponse:
    """Step 1: propose categories. One routing call, plus an add-only
    completeness call when code flags possibly-missed categories. An
    unanswerable question is a valid 200 with zero candidates — empty results
    are correct results."""
    description = _dataset_description(dataset_id)
    ctx = _load_context(dataset_id, description)
    client = _client()

    # Identical question against identical data serves the stored proposal —
    # temperature 0 is not a determinism guarantee, and a report re-run must
    # not route differently for no reason. Any data/prompt/model change makes
    # a different key, so staleness is structurally impossible.
    cache_key = ask_cache.route_key(
        ask_service.context_cache_key(dataset_id, description),
        req.question, req.question_scope, client.model_id,
        extra={"demographic_filter": {
            f: sorted(v) for f, v in sorted(req.demographic_filter.items())
            if v}} if req.demographic_filter else None)
    hit = ask_cache.load(dataset_id, cache_key)
    if hit is not None:
        return AskRouteResponse(**{**hit, "cached": True})

    try:
        route, stats = ask_service.propose(
            client, req.question, ctx, description,
            question_scope=req.question_scope,
            demographic_filter=req.demographic_filter)
    except ValueError as exc:        # unknown scope ids / demographic values
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:      # Gemini failure after retries
        raise HTTPException(status_code=502, detail=str(exc))

    warnings = list(stats["warnings"])
    if stats["invalid_label_ids"]:
        warnings.append(f"{stats['invalid_label_ids']} invented label id(s) "
                        "dropped from the proposal")
    resp = AskRouteResponse(
        answerable=route["answerable"],
        route=route["route"],
        reason=route["reason"],
        aggregate_target=route.get("aggregate_target", ""),
        candidates=[_candidate_out(c, ctx) for c in route["candidates"]],
        available_categories=_available_categories(route, ctx),
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
        time_filter=route.get("time_filter", ""),
        available_time=({"day": len(ctx.time_day),
                         "night": len(ctx.time_night),
                         "mentioned": len(ctx.time_mentioned)}
                        if (ctx.time_day or ctx.time_night) else {}),
        question_scope=route.get("question_scope", []),
        demographic_filter=route.get("demographic_filter", {}),
        warnings=warnings,
    )
    ask_cache.store(dataset_id, cache_key, resp.model_dump())
    return resp


@router.post("/answer", response_model=AskAnswerResponse)
def ask_answer(dataset_id: str, req: AskAnswerRequest) -> AskAnswerResponse:
    """Step 2: synthesize from the analyst-approved selection. Exactly one
    synthesis call (zero on an aggregate_direct tally); the verifier is pure
    code and never rewrites what it checks. Counts and quotes are recomputed
    here — the request carries choices, never evidence."""
    description = _dataset_description(dataset_id)
    ctx = _load_context(dataset_id, description)
    try:
        route = ask_service.route_from_selection(
            [c.model_dump() for c in req.selected],
            req.route, req.reason, req.lexicon_concepts, ctx,
            group_by=req.group_by, location_filter=req.location_filter,
            actionability_filter=req.actionability_filter,
            event_filter=req.event_filter,
            time_filter=req.time_filter,
            question_scope=req.question_scope,
            aggregate_target=req.aggregate_target,
            demographic_filter=req.demographic_filter)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    client = _client()
    synth_client = _synth_client()

    # The identical approved request against identical data pulls up the
    # ORIGINAL stored answer (same run_id) instead of re-synthesizing — the
    # key covers the full selection and filters, so an analyst edit is a
    # different request and computes fresh.
    cache_key = ask_cache.answer_key(
        ask_service.context_cache_key(dataset_id, description),
        req.model_dump(),
        f"{client.model_id}|{(synth_client or client).model_id}")
    hit = ask_cache.load(dataset_id, cache_key)
    if hit is not None:
        return AskAnswerResponse(**{**hit, "cached": True})

    try:
        result = ask_service.answer(
            client, req.question, route, ctx,
            description=description,
            proposed_label_ids=req.proposed_label_ids or None,
            synth_client=synth_client,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    selection = result["selection"] or {"deselected": [], "added": []}
    resp = AskAnswerResponse(
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
        sub_breakdowns={
            s["label_id"]: AskSubBreakdownOut(
                sub_counts=[AskSubCountOut(**sc) for sc in s["sub_counts"]],
                generic=s.get("sub_generic", 0),
                coded=s.get("sub_coded", 0))
            for s in result["evidence"]["selection"] if s.get("sub_counts")},
        uncovered_categories=[
            AskUncoveredOut(**u)
            for u in result["evidence"].get("uncovered_categories") or []],
        aggregate=result.get("aggregate"),
        verification=result.get("verification"),
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
        time_filter=result["evidence"].get("time_filter") or "",
        time_denominator=result["evidence"].get("time_denominator"),
        demographic_filter=result["evidence"].get("demographic_filter") or {},
        demographic_denominator=result["evidence"].get(
            "demographic_denominator"),
        demographic_thin=bool(result["evidence"].get("demographic_thin")),
        location_filter_implicit_questions=result["evidence"].get(
            "location_filter_implicit_questions") or [],
        invalid_citations=result["invalid_citations"],
        deselected=selection["deselected"],
        added=selection["added"],
    )
    ask_cache.store(dataset_id, cache_key, resp.model_dump())
    return resp

