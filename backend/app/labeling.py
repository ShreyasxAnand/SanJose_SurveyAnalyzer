"""Phase 3: labeling pass — assign frozen taxonomy labels to every response.

This is where scale lives. Induction reads a sample of the corpus; labeling
reads all of it, which is the only way to get real counts. Counts come from
counting rows, never from a model estimate.

Per response the model returns: child label_ids (multi-label), sentiment, an
explicit `uncategorized` when nothing fits, and a fit score. Guards mirror
induction's evidence resolution — responses are numbered in the prompt and
resolved back to response_key in code, and any label_id the model invents is
dropped and counted rather than trusted.

`uncategorized` and low `fit` are not failures; they are the completeness
signal. A cluster of them is how a missing category announces itself without
anyone having to guess what to look for.
"""
from __future__ import annotations

import json
from pathlib import Path

from .llm import ModelClient

SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 40
MAX_RESPONSE_CHARS = 600

LABEL_SYSTEM = """\
{dataset_context}You are coding open-ended survey responses against a FROZEN taxonomy.

Survey question shown to respondents:
"{question_text}"

TAXONOMY (id | name — description):
{taxonomy}

You will receive numbered responses. Code each one.

Rules:
- Use ONLY the ids listed above, copied exactly. Never invent an id.
- Most responses get 1-3 labels. Assign every label that genuinely applies —
  a response raising three separate issues gets three labels.
- If NOTHING in the taxonomy fits, set "uncategorized": true and leave
  "label_ids" empty. This is a correct and expected answer. Do NOT force a
  response into the nearest category just to avoid an empty list.
- "fit": how well the assigned labels cover what the response actually says.
  3 = fully covered. 2 = partly, something is missing. 1 = poor, the labels
  are the closest available but not really right. Use 1 honestly; it is how
  gaps in the taxonomy get found.
- "sentiment": "negative", "positive", "neutral", or "mixed" — the
  respondent's stance in the response, not your opinion of the topic.
- "locations": every place the response mentions, copied VERBATIM as written.
  Both formal names ("St. James Park", "Story Road") and informal places
  ("bus stop", "downtown", "the park by my house", "freeway"). Empty list if
  none. Never normalize, expand, or add a place the text does not contain —
  copy the exact words.
- "time_context": phrases saying WHEN, copied VERBATIM as written — time of
  day ("at night", "after dark"), frequency ("every weekend"), or period
  ("since covid", "the last few years"). Empty list if none. Same rule:
  copy the exact words, never paraphrase.
- "actionability": "specific" if the response proposes a concrete,
  implementable action (names a place, mechanism, or particular change —
  "add lighting on Story Road"); "general" if it is a broad wish, complaint,
  or condition ("fix crime", "too much trash"). Judge the response, not the
  topic.
- Code what the response says, not what you assume the respondent meant.
- Never estimate counts, frequencies or percentages.

Return ONLY valid JSON, exactly this shape:
{{"responses": [{{"n": 1, "label_ids": ["q_001"], "uncategorized": false,
"fit": 3, "sentiment": "negative", "locations": ["bus stop"],
"time_context": ["at night"], "actionability": "general"}}]}}
"""

LABEL_USER = """Responses to code ({n} total):
{numbered_responses}
"""

VALID_SENTIMENT = {"negative", "positive", "neutral", "mixed"}
VALID_ACTIONABILITY = {"specific", "general"}


def _verbatim_spans(raw_list, source_text: str) -> tuple[list[str], int]:
    """Keep only spans that literally appear in the source (case-insensitive),
    deduped. Returns (kept, n_dropped). The guard that makes hallucinated
    extractions structurally impossible."""
    source_lower = source_text.lower()
    kept, seen, dropped = [], set(), 0
    for raw in raw_list or []:
        span = str(raw).strip()
        if not span:
            continue
        if span.lower() not in source_lower:
            dropped += 1
        elif span.lower() not in seen:
            seen.add(span.lower())
            kept.append(span)
    return kept, dropped


def respondent_key(response_key: str) -> str | None:
    """"{dataset_id}:{question_id}:{source_row_index}" -> "{dataset_id}:{row}".
    The same source row across questions is the same person — wide-format
    ingest guarantees it — so this key joins one respondent's answers."""
    parts = response_key.split(":")
    return f"{parts[0]}:{parts[2]}" if len(parts) == 3 else None


def render_taxonomy(taxonomy: dict) -> str:
    """Children only. Parents are never assigned — labeling keys off child
    label_ids so the parent layer stays editable without a relabel."""
    lines = []
    for lab in taxonomy["labels"]:
        desc = (lab.get("description") or "").replace("\n", " ").strip()
        lines.append(f"{lab['label_id']} | {lab['name']} — {desc}")
    return "\n".join(lines)


def build_label_prompts(
    taxonomy: dict, batch: list[tuple[str, str]], dataset_description: str = ""
) -> tuple[str, str]:
    from .induction import context_block

    lines = []
    for i, (_key, text) in enumerate(batch, start=1):
        t = text.replace("\n", " ").strip()
        if len(t) > MAX_RESPONSE_CHARS:
            t = t[:MAX_RESPONSE_CHARS] + "…"
        lines.append(f"{i}. {t}")
    system = LABEL_SYSTEM.format(
        dataset_context=context_block(dataset_description),
        question_text=taxonomy["question_text"],
        taxonomy=render_taxonomy(taxonomy),
    )
    user = LABEL_USER.format(n=len(batch), numbered_responses="\n".join(lines))
    return system, user


def parse_label_output(
    raw: str, batch: list[tuple[str, str]], valid_ids: set[str]
) -> tuple[list[dict], dict]:
    """Resolve one batch. Returns (assignments, stats). A response the model
    omits is recorded as unlabelled rather than dropped — silence is not the
    same as 'nothing fits', and conflating them would understate the gap."""
    from .induction import extract_json

    obj = extract_json(raw)
    # a model sometimes returns the responses array bare, without the wrapper
    # object — valid JSON, wrong shape. Normalize; anything else non-dict goes
    # to ValueError so the caller's retry-once path handles it.
    if isinstance(obj, list):
        obj = {"responses": obj}
    if not isinstance(obj, dict):
        raise ValueError(f"model returned {type(obj).__name__}, not an object")
    got = {}
    stats = {"invalid_ids": 0, "out_of_range": 0, "missing": 0,
             "invalid_locations": 0, "invalid_time_context": 0}
    for r in obj.get("responses") or []:
        if not isinstance(r, dict):
            continue
        try:
            n = int(r.get("n"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(batch):
            stats["out_of_range"] += 1
            continue
        ids, seen = [], set()
        for raw_id in r.get("label_ids") or []:
            rid = str(raw_id).strip()
            if rid not in valid_ids:
                stats["invalid_ids"] += 1
            elif rid not in seen:
                seen.add(rid)
                ids.append(rid)
        sentiment = str(r.get("sentiment", "")).strip().lower()
        try:
            fit = int(r.get("fit", 0))
        except (TypeError, ValueError):
            fit = 0
        source_text = batch[n - 1][1]
        locations, n_bad_loc = _verbatim_spans(r.get("locations"), source_text)
        stats["invalid_locations"] += n_bad_loc
        times, n_bad_time = _verbatim_spans(r.get("time_context"), source_text)
        stats["invalid_time_context"] += n_bad_time
        actionability = str(r.get("actionability", "")).strip().lower()
        got[n] = {
            "label_ids": ids,
            # a model can say uncategorized OR simply return no valid ids;
            # both mean the taxonomy did not cover this response
            "uncategorized": bool(r.get("uncategorized")) or not ids,
            "fit": fit if 1 <= fit <= 3 else None,
            "sentiment": sentiment if sentiment in VALID_SENTIMENT else None,
            "locations": locations,
            "time_context": times,
            "actionability": (actionability
                              if actionability in VALID_ACTIONABILITY else None),
        }

    assignments = []
    for i, (key, text) in enumerate(batch, start=1):
        a = got.get(i)
        if a is None:
            stats["missing"] += 1
            a = {"label_ids": [], "uncategorized": True, "fit": None,
                 "sentiment": None, "locations": [], "time_context": [],
                 "actionability": None, "not_returned": True}
        assignments.append({"response_key": key,
                            "respondent_key": respondent_key(key), **a})
    return assignments, stats


def run_labeling(
    rows: list[tuple[str, str]],
    taxonomy: dict,
    client: ModelClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dataset_description: str = "",
    progress: bool = True,
) -> tuple[list[dict], dict]:
    """Label every response. Returns (assignments, report)."""
    valid_ids = {lab["label_id"] for lab in taxonomy["labels"]}
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    assignments: list[dict] = []
    totals = {"invalid_ids": 0, "out_of_range": 0, "missing": 0,
              "invalid_locations": 0, "invalid_time_context": 0}
    failed_batches: list[dict] = []

    for bi, batch in enumerate(batches):
        system, user = build_label_prompts(taxonomy, batch, dataset_description)
        try:
            raw = client.complete(system, user)
            try:
                got, stats = parse_label_output(raw, batch, valid_ids)
            except (ValueError, json.JSONDecodeError):
                raw = client.complete(
                    system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
                    user)
                got, stats = parse_label_output(raw, batch, valid_ids)
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            # a failed batch leaves its responses unlabelled and says so, rather
            # than aborting a run that is 90% done
            failed_batches.append({"batch": bi, "n_responses": len(batch),
                                   "error": f"{type(exc).__name__}: {exc}"[:200]})
            got = [{"response_key": k, "respondent_key": respondent_key(k),
                    "label_ids": [], "uncategorized": True,
                    "fit": None, "sentiment": None, "locations": [],
                    "time_context": [], "actionability": None,
                    "batch_failed": True}
                   for k, _ in batch]
            stats = {"invalid_ids": 0, "out_of_range": 0, "missing": len(batch),
                     "invalid_locations": 0, "invalid_time_context": 0}
            if progress:
                print(f"  batch {bi + 1}/{len(batches)}: FAILED ({type(exc).__name__})")
        else:
            if progress:
                n_unc = sum(1 for a in got if a["uncategorized"])
                print(f"  batch {bi + 1}/{len(batches)}: {len(got)} coded, "
                      f"{n_unc} uncategorized, {stats['invalid_ids']} invalid ids")
        for k in totals:
            totals[k] += stats[k]
        assignments.extend(got)

    labelled = [a for a in assignments if a["label_ids"]]
    counts: dict[str, int] = {lab["label_id"]: 0 for lab in taxonomy["labels"]}
    for a in labelled:
        for lid in a["label_ids"]:
            counts[lid] += 1
    per_response = [len(a["label_ids"]) for a in labelled]

    report = {
        "schema_version": SCHEMA_VERSION,
        "batch_size": batch_size,
        "n_batches": len(batches),
        "failed_batches": failed_batches,
        "responses_total": len(rows),
        "responses_labelled": len(labelled),
        "responses_uncategorized": sum(1 for a in assignments if a["uncategorized"]),
        "invalid_ids_dropped": totals["invalid_ids"],
        "responses_not_returned": totals["missing"],
        "invalid_locations_dropped": totals["invalid_locations"],
        "responses_with_locations": sum(
            1 for a in assignments if a.get("locations")),
        "invalid_time_context_dropped": totals["invalid_time_context"],
        "responses_with_time_context": sum(
            1 for a in assignments if a.get("time_context")),
        "labels_per_response_mean": (
            round(sum(per_response) / len(per_response), 2) if per_response else 0
        ),
        "label_counts": counts,
        "labels_with_zero_responses": [
            lab["label_id"] for lab in taxonomy["labels"] if counts[lab["label_id"]] == 0
        ],
        "fit_distribution": dict(
            sorted(((str(a["fit"]), 0) for a in assignments), key=lambda x: x[0])
        ),
        "sentiment_distribution": {},
    }
    fit_counter: dict[str, int] = {}
    sent_counter: dict[str, int] = {}
    act_counter: dict[str, int] = {}
    for a in assignments:
        fit_counter[str(a["fit"])] = fit_counter.get(str(a["fit"]), 0) + 1
        sent_counter[str(a["sentiment"])] = sent_counter.get(str(a["sentiment"]), 0) + 1
        act_counter[str(a["actionability"])] = act_counter.get(str(a["actionability"]), 0) + 1
    report["fit_distribution"] = dict(sorted(fit_counter.items()))
    report["sentiment_distribution"] = dict(sorted(sent_counter.items()))
    report["actionability_distribution"] = dict(sorted(act_counter.items()))
    return assignments, report
