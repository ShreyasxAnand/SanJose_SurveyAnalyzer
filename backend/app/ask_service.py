"""Shared service layer for Phase 5 ask — one code path for CLI and API.

`scripts.ask` and the `/datasets/{id}/ask/*` endpoints are both thin callers
of this module. Keeping them on the same functions means the browser and the
terminal produce identical evidence, identical artifacts, and identical
audit manifests — there is no "UI version" of the pipeline to drift.

The stateless two-step contract: `load_context` + `propose` serve step 1
(the routing proposal the analyst reviews); `answer` serves step 2, taking
the analyst's approved selection. The caller only ever supplies *choices*
(label ids, concept names) — every count and quote is recomputed here from
the artifacts on disk, so a client can never inject a number.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import induction, labeling, llm, locations as locations_mod, router, summary
from .llm import ModelClient

ANSWERS_DIR = induction.DATA_DIR / "answers"


@dataclass
class AskContext:
    """Everything both steps need, loaded fresh from the latest artifacts so
    the summary can never be stale relative to the labels."""
    dataset_id: str
    summary: dict
    summary_text: str
    index: dict[str, dict]              # label_id -> entry (+question_id)
    valid_ids: set[str]
    question_ids: list[str]
    question_totals: dict[str, int]
    lexicon: dict
    valid_concepts: set[str]
    texts: dict[str, str]               # response_key -> verbatim text
    keys_by_question: dict[str, list[str]]
    members: dict[str, list[str]]       # label_id -> response_keys
    locations: dict = field(default_factory=dict)
    valid_locations: set[str] = field(default_factory=set)
    location_members: dict[str, list[str]] = field(default_factory=dict)
    location_kinds: dict[str, str] = field(default_factory=dict)
    # response_key -> "specific" | "general", straight from the labeling pass.
    # Absent keys were never coded (older labels runs, failed batches) and are
    # excluded from a filtered answer rather than assumed either way.
    actionability: dict[str, str] = field(default_factory=dict)
    actionability_counts: dict[str, int] = field(default_factory=dict)
    # Responses recounting a first-hand incident, and the rows the labeling
    # pass actually checked. Kept as two sets because event_occurred is a
    # bool: False means "did not describe an incident", and for an unreturned
    # or failed-batch row it means nothing at all. `event_coded` is the same
    # predicate labeling's own `responses_event_coded` uses.
    events: set[str] = field(default_factory=set)
    event_coded: set[str] = field(default_factory=set)


def discover_dataset_id(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    ds = sorted(d.name for d in summary.LABELS_DIR.iterdir() if d.is_dir()) \
        if summary.LABELS_DIR.is_dir() else []
    if len(ds) != 1:
        raise FileNotFoundError(f"Found {len(ds)} labeled datasets {ds}; specify one.")
    return ds[0]


def load_context(dataset_id: str, parquet: str | None = None,
                 description: str = "") -> AskContext:
    s = summary.build_summary(dataset_id, description)
    summary.write_summary(s)     # keep the Phase 4 artifact on disk current

    lexicon: dict = {}
    lex_path = summary.LEXICON_DIR / dataset_id / "lexicon.json"
    if lex_path.exists():
        lexicon = json.loads(lex_path.read_text(encoding="utf-8"))

    parquet_path = induction.discover_parquet(parquet)
    question_ids = [q["question_id"] for q in s["questions"]]
    texts: dict[str, str] = {}
    keys_by_question: dict[str, list[str]] = {}
    bulk = induction.load_questions_bulk(parquet_path, question_ids)
    for q in question_ids:
        rows, _meta, _filtered = bulk[q]
        keys_by_question[q] = [r.response_key for r in rows]
        for r in rows:
            texts[r.response_key] = r.text

    members: dict[str, list[str]] = {}
    actionability: dict[str, str] = {}
    events: set[str] = set()
    event_coded: set[str] = set()
    for q in question_ids:
        run = summary.latest_run_dir(summary.LABELS_DIR / dataset_id / q,
                                     "assignments.json")
        assignments = json.loads((run / "assignments.json").read_text(encoding="utf-8"))
        members.update(router.members_by_label(assignments))
        for a in assignments:
            v = a.get("actionability")
            if v in labeling.VALID_ACTIONABILITY:
                actionability[a["response_key"]] = v
            if not a.get("not_returned") and not a.get("batch_failed"):
                event_coded.add(a["response_key"])
                if a.get("event_occurred"):
                    events.add(a["response_key"])
    actionability_counts: dict[str, int] = {}
    for v in actionability.values():
        actionability_counts[v] = actionability_counts.get(v, 0) + 1

    locations: dict = {}
    location_members: dict[str, list[str]] = {}
    loc_path = summary.LOCATIONS_DIR / dataset_id / "locations.json"
    if loc_path.exists():
        locations = json.loads(loc_path.read_text(encoding="utf-8"))
        all_keys = list(texts)
        location_members = locations_mod.match_locations(
            locations, all_keys, [texts[k] for k in all_keys])

    return AskContext(
        dataset_id=dataset_id,
        summary=s,
        summary_text=summary.render_summary(s),
        index=summary.label_index(s),
        valid_ids=summary.valid_label_ids(s),
        question_ids=question_ids,
        question_totals={q["question_id"]: q["n_responses"] for q in s["questions"]},
        lexicon=lexicon,
        valid_concepts={c["name"] for c in lexicon.get("concepts", [])},
        texts=texts,
        keys_by_question=keys_by_question,
        members=members,
        locations=locations,
        valid_locations=set(location_members),
        location_members=location_members,
        location_kinds=locations_mod.concept_kinds(locations),
        actionability=actionability,
        actionability_counts=actionability_counts,
        events=events,
        event_coded=event_coded,
    )


def propose(client: ModelClient, question: str, ctx: AskContext,
            description: str = "") -> tuple[dict, dict]:
    """Step 1: the routing proposal. Returns (route, stats) exactly as
    router.run_route does — the caller renders it for review."""
    return router.run_route(client, question, ctx.summary_text,
                            ctx.valid_ids, ctx.valid_concepts, description,
                            valid_locations=ctx.valid_locations,
                            actionability_counts=ctx.actionability_counts,
                            event_counts=(len(ctx.events), len(ctx.event_coded)))


def usage_block(clients: list[ModelClient], elapsed: float) -> dict:
    """Combined usage plus a per-model split. The split is what makes the
    number checkable once routing and synthesis run on different models at
    different prices — a single total would silently mix them."""
    unique: list[ModelClient] = []
    for c in clients:
        if not any(c is seen for seen in unique):
            unique.append(c)
    by_model: dict[str, dict[str, int]] = {}
    for c in unique:
        # a client that made no calls is not evidence a model was used — the
        # /answer endpoint builds a routing client it never calls
        if not c.usage.calls:
            continue
        e = by_model.setdefault(
            c.model_id, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "thinking_tokens": 0})
        e["calls"] += c.usage.calls
        e["input_tokens"] += c.usage.input_tokens
        e["output_tokens"] += c.usage.output_tokens
        # a share of output_tokens, not an addition to it — billed, invisible
        e["thinking_tokens"] += c.usage.thinking_tokens

    unpriced = []
    total_cost = 0.0
    for model_id, e in by_model.items():
        cost = llm.price_usd(model_id, e["input_tokens"], e["output_tokens"])
        e["est_cost_usd"] = round(cost, 6) if cost is not None else None
        if cost is None:
            unpriced.append(model_id)
        else:
            total_cost += cost

    block = {
        "calls": sum(e["calls"] for e in by_model.values()),
        "input_tokens": sum(e["input_tokens"] for e in by_model.values()),
        "output_tokens": sum(e["output_tokens"] for e in by_model.values()),
        "est_cost_usd": round(total_cost, 6),
        "elapsed_seconds": round(elapsed, 1),
        "by_model": by_model,
    }
    if unpriced:
        # the total covers only the models we have rates for — say which are
        # missing rather than letting the number read as complete
        block["unpriced_models"] = unpriced
    return block


def make_run_id(question: str) -> str:
    return (f"{induction.utc_now()}_"
            f"{hashlib.sha256(question.encode('utf-8')).hexdigest()[:8]}")


ROUTE_LABEL = {
    "retrieval": "retrieval (read what respondents say)",
    "aggregate": "aggregate (answer from computed counts)",
    "comparative": "comparative (contrast groups of responses)",
    "hybrid": "hybrid (counts first, then explain from responses)",
}


def process_note(route: dict, evidence: dict, ctx: AskContext,
                 n_cited: int, lex_counts: list[dict]) -> str:
    """One computed paragraph narrating what the pipeline actually did for
    this question. Deliberately NOT model-written: every figure in it is
    counted from the run's own data, so the process description can never
    disagree with the process."""
    sel = sorted(evidence["selection"], key=lambda s: -s["count"])
    if not sel:
        return (f"Routed as {ROUTE_LABEL.get(route['route'], route['route'])}. "
                "No relevant categories were found, so nothing was retrieved — "
                "an empty result rather than the nearest plausible match.")
    names = [s["name"] for s in sel]
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more"
                                    if len(names) > 3 else "")
    # filter-aware: with a location_filter this is the filtered union, so the
    # note can never quote a broader coverage than the evidence actually had
    unique = evidence.get("n_unique_responses",
                          len({k for s in sel
                               for k in ctx.members.get(s["label_id"], [])}))
    # "_"-prefixed keys are budget disclosures, not per-category notes
    n_sampled = sum(1 for k in evidence["sampling_notes"] if not k.startswith("_"))
    budget_notes = [v for k, v in evidence["sampling_notes"].items()
                    if k.startswith("_quote_budget")]
    parts = [
        f"Routed as {ROUTE_LABEL.get(route['route'], route['route'])}"
        + (f" — {route['reason'].rstrip('.')}" if route["reason"] else "") + ".",
        f"Searched {len(sel)} of {len(ctx.valid_ids)} categories ({shown}), "
        f"covering {unique} unique responses.",
        f"Showed {len(evidence['quotes'])} verbatims to the answering model"
        + (f" (sampled for {n_sampled} categor"
           + ("y" if n_sampled == 1 else "ies")
           + "; counts always cover the full data)" if n_sampled else "")
        + f"; {n_cited} of them are cited in the answer.",
    ]
    for note in budget_notes:
        parts.append(f"Quote budget: {note}.")
    if evidence.get("location_filter"):
        names = ", ".join(evidence["location_filter"])
        parts.append(f"Evidence restricted to responses mentioning {names} "
                     "(deterministic keyword match).")
    denom_act = evidence.get("actionability_denominator")
    if denom_act:
        act = evidence.get("actionability_filter", "")
        uncoded = denom_act["in_scope"] - denom_act["coded"]
        parts.append(
            f"Evidence restricted to responses the labeling pass marked "
            f"\"{act}\": {denom_act['matching']} of {denom_act['in_scope']} "
            f"in-scope responses"
            + (f" ({uncoded} {'was' if uncoded == 1 else 'were'} never marked "
               f"either way)" if uncoded else "")
            + ".")
    denom_evt = evidence.get("event_denominator")
    if denom_evt:
        parts.append(
            f"Evidence restricted to responses recounting a first-hand "
            f"incident: {denom_evt['matching']} of {denom_evt['in_scope']} "
            f"in-scope responses. The rest did not describe one, which is not "
            f"the same as nothing having happened to them.")
    denom = evidence.get("location_denominator")
    if denom:
        parts.append(f"Grouped by place: {denom['naming_any']} of "
                     f"{denom['in_scope']} in-scope responses named one — "
                     "only those are localizable.")
    for lc in lex_counts:
        parts.append(f"Keyword concept \"{lc['concept']}\" matched "
                     f"{lc['mentions_total']} responses by exact match.")
    if evidence["group_counts"]:
        gs = "; ".join(f"\"{g['name']}\" = {g['count_unique_responses']} responses"
                       for g in evidence["group_counts"])
        parts.append(f"Compared groups: {gs}.")
    return " ".join(parts)


def answer(
    client: ModelClient,
    question: str,
    route: dict,
    ctx: AskContext,
    description: str = "",
    max_quotes_per_label: int = router.DEFAULT_MAX_QUOTES_PER_LABEL,
    max_total_quotes: int = router.DEFAULT_MAX_TOTAL_QUOTES,
    seed: int = 7,
    proposed_label_ids: list[str] | None = None,
    route_stats: dict | None = None,
    synth_client: ModelClient | None = None,
) -> dict:
    """Step 2: evidence + synthesis + artifacts, from an approved route.

    `route` must be the shape parse_route_output returns; when the selection
    was edited by an analyst, `proposed_label_ids` carries what the model
    originally proposed so the difference is recorded — deselections are the
    cheapest quality signal the platform has.

    `synth_client` writes the answer; it defaults to `client` but is normally
    a stronger model, since synthesis is one call per question while every
    other stage scales with the corpus."""
    t0 = time.time()
    synth = synth_client or client
    evidence = router.gather_evidence(
        route, ctx.index, ctx.members, ctx.texts,
        max_quotes_per_label=max_quotes_per_label,
        max_total_quotes=max_total_quotes, seed=seed,
        location_members=ctx.location_members or None,
        location_kinds=ctx.location_kinds or None,
        actionability_of=ctx.actionability or None,
        event_keys=ctx.events or None,
        event_coded_keys=ctx.event_coded or None)
    lex_counts = router.lexicon_counts(
        ctx.lexicon, route["lexicon_concepts"], ctx.keys_by_question, ctx.texts
    ) if route["lexicon_concepts"] else []

    raw_answer = router.run_synth(synth, question, route, evidence, ctx.index,
                                  lex_counts, ctx.question_totals, description)
    answer_body, cited, n_invalid = router.resolve_citations(raw_answer, evidence)
    # the single-document form (body + Sources) goes to answer.md and the
    # CLI; the API returns the body and the sources as separate fields
    answer_md = answer_body + router.render_sources_section(cited)
    note = process_note(route, evidence, ctx, len(cited), lex_counts)
    stats = {
        "categories_searched": len(evidence["selection"]),
        "categories_total": len(ctx.valid_ids),
        "unique_responses": evidence["n_unique_responses"],
        "quotes_shown": len(evidence["quotes"]),
        "quotes_cited": len(cited),
    }

    selected_ids = [c["label_id"] for c in route["candidates"]]
    selection = None
    if proposed_label_ids is not None:
        selection = {
            "proposed": proposed_label_ids,
            "selected": selected_ids,
            "deselected": [i for i in proposed_label_ids if i not in selected_ids],
            "added": [i for i in selected_ids if i not in proposed_label_ids],
        }

    run_id = make_run_id(question)
    out_dir = write_artifacts(
        run_id=run_id, question=question, ctx=ctx, route=route,
        route_stats=route_stats or {}, evidence=evidence, lex_counts=lex_counts,
        answer_md=answer_md, n_invalid_citations=n_invalid,
        selection=selection, client=client, description=description,
        elapsed=time.time() - t0, process_note=note, synth_client=synth)

    if selection and (selection["deselected"] or selection["added"]):
        log_path = ANSWERS_DIR / ctx.dataset_id / "selection_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "utc": induction.utc_now(), "run_id": run_id,
                "question": question, **selection,
            }, ensure_ascii=False) + "\n")

    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "answer_markdown": answer_md,
        "answer_body": answer_body,
        "stats": stats,
        "process_note": note,
        "evidence": evidence,
        "lexicon_counts": lex_counts,
        "cited": cited,
        "invalid_citations": n_invalid,
        "selection": selection,
    }


def write_artifacts(*, run_id: str, question: str, ctx: AskContext, route: dict,
                    route_stats: dict, evidence: dict, lex_counts: list[dict],
                    answer_md: str | None, n_invalid_citations: int,
                    selection: dict | None, client: ModelClient,
                    description: str, elapsed: float,
                    process_note: str | None = None,
                    synth_client: ModelClient | None = None) -> Path:
    """answer.md + manifest.json under data/answers/{ds}/{run_id}/ — same
    artifact whether the question came from the CLI or the browser."""
    out_dir = ANSWERS_DIR / ctx.dataset_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = [f"# {question}", "",
           f"route: **{route['route']}** — {route['reason']}", ""]
    if process_note:
        doc += [f"> {process_note}", ""]
    for c in route["candidates"]:
        e = ctx.index[c["label_id"]]
        n = len(ctx.members.get(c["label_id"], []))
        doc.append(f"- `{c['label_id']}` {e['name']} (n={n}, {c['relevance']}) "
                   f"— {c['rationale']}")
    doc.append("")
    if answer_md:
        doc += ["## Answer", "", answer_md, ""]
    elif not route["answerable"]:
        doc += ["## Answer", "", "**Not answerable from this data.** "
                + (route["reason"] or ""), ""]
    (out_dir / "answer.md").write_text("\n".join(doc), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": induction.utc_now(),
        "tool": "ask",
        "schema_version": router.SCHEMA_VERSION,
        # the routing model; synthesis normally runs on a stronger one, and a
        # reproducible run needs both recorded
        "model_id": client.model_id,
        "synth_model_id": (synth_client or client).model_id,
        "question": question,
        "dataset_id": ctx.dataset_id,
        "dataset_description": description,
        "summary_generated_utc": ctx.summary["generated_utc"],
        "labels_runs": {q["question_id"]: q["labels_run"]
                        for q in ctx.summary["questions"]},
        "route": route,
        "route_stats": route_stats,
        "process_note": process_note,
        "selection": selection,
        "evidence": {
            "counts": {s["label_id"]: s["count"] for s in evidence["selection"]},
            "sampling_notes": evidence["sampling_notes"],
            "quotes_shown": [q["response_key"] for q in evidence["quotes"]],
            "lexicon_counts": lex_counts,
            "location_counts": evidence.get("location_counts") or [],
            "location_denominator": evidence.get("location_denominator"),
            "location_filter": evidence.get("location_filter") or [],
            "actionability_filter": evidence.get("actionability_filter") or "",
            "actionability_denominator": evidence.get("actionability_denominator"),
            "event_filter": evidence.get("event_filter") or "",
            "event_denominator": evidence.get("event_denominator"),
        },
        "invalid_citations": n_invalid_citations,
        "usage": usage_block([client] + ([synth_client] if synth_client else []),
                             elapsed),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def route_from_selection(selected: list[dict], route_name: str, reason: str,
                         lexicon_concepts: list[str], ctx: AskContext,
                         group_by: str = "category",
                         location_filter: list[str] | None = None,
                         actionability_filter: str = "",
                         event_filter: str = "") -> dict:
    """Rebuild a route dict from an analyst-approved selection (step 2 of the
    stateless flow). Unknown label ids raise ValueError — the server-side
    gate; the caller turns that into a 422. Unknown concepts are dropped."""
    candidates = []
    seen: set[str] = set()
    unknown: list[str] = []
    for c in selected:
        lid = str(c.get("label_id", "")).strip()
        if lid not in ctx.valid_ids:
            unknown.append(lid)
            continue
        if lid in seen:
            continue
        seen.add(lid)
        relevance = str(c.get("relevance", "")).strip().lower()
        candidates.append({
            "label_id": lid,
            "relevance": relevance if relevance in router.VALID_RELEVANCE else "medium",
            "rationale": str(c.get("rationale", "")).strip(),
        })
    if unknown:
        raise ValueError(f"Unknown label ids: {unknown}")
    if not candidates:
        raise ValueError("No categories selected")
    route_name = route_name.strip().lower()
    if route_name not in router.VALID_ROUTES:
        route_name = "retrieval"
    group_by = (group_by or "category").strip().lower()
    if group_by not in {"category", "location"} or not ctx.valid_locations:
        group_by = "category"
    # an unknown filter value is a client bug, not something to guess at —
    # 422 rather than silently answering over a different evidence set
    act = router.normalize_actionability(actionability_filter)
    if act == "invalid":
        raise ValueError(
            f"Unknown actionability_filter {actionability_filter!r}; "
            f"expected one of: specific, general, or empty")
    if act and not ctx.actionability:
        act = ""
    evt = router.normalize_event(event_filter)
    if evt == "invalid":
        raise ValueError(
            f"Unknown event_filter {event_filter!r}; expected \"reported\" or "
            f"empty (there is no filter for responses without an incident)")
    if evt and not ctx.events:
        evt = ""
    return {
        "answerable": True,
        "route": route_name,
        "reason": reason.strip(),
        "candidates": candidates,
        "groups": [],
        "lexicon_concepts": [c for c in lexicon_concepts if c in ctx.valid_concepts],
        "group_by": group_by,
        "location_filter": [l for l in (location_filter or [])
                            if l in ctx.valid_locations],
        "actionability_filter": act,
        "event_filter": evt,
    }
