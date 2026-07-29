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

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import ModelClient

SCHEMA_VERSION = 2          # v2 = assignments carry event_occurred, no sentiment
DEFAULT_BATCH_SIZE = 40
# Batches are independent and temperature is 0, so concurrency changes latency
# and nothing else. Bounded because the ceiling here is the provider's rate
# limit, not local CPU; the client already backs off on 429.
DEFAULT_WORKERS = 8
MAX_RESPONSE_CHARS = 600

LABEL_SYSTEM = """\
{dataset_context}You are coding open-ended survey responses against a FROZEN taxonomy.

Survey question shown to respondents:
"{question_text}"

TAXONOMY (id | name — description):
{taxonomy}

You will receive numbered responses. Code each one.

Output keys are ABBREVIATED. Use exactly these keys and no others:
  n = the response's number, copied from the input
  l = list of taxonomy ids that apply
  f = fit, 1-3
  p = places mentioned, verbatim
  t = time phrases, verbatim
  a = actionability, "s" or "g"
  e = event occurred, 1 or 0

Rules:
- "l": use ONLY the ids listed above, copied exactly. Never invent an id.
  Most responses get 1-3. Assign every label that genuinely applies — a
  response raising three separate issues gets three labels.
- If NOTHING in the taxonomy fits, return an EMPTY list for "l". This is a
  correct and expected answer. Do NOT force a response into the nearest
  category just to avoid an empty list.
- "f": how well the assigned labels cover what the response actually says.
  3 = fully covered. 2 = partly, something is missing. 1 = poor, the labels
  are the closest available but not really right. Use 1 honestly; it is how
  gaps in the taxonomy get found. Always include "f".
- "p": every place the response mentions, copied VERBATIM as written.
  Both formal names ("St. James Park", "Story Road") and informal places
  ("bus stop", "downtown", "the park by my house", "freeway"). Never
  normalize, expand, or add a place the text does not contain — copy the
  exact words. OMIT the "p" key entirely if the response names no place.
- "t": phrases saying WHEN, copied VERBATIM as written — time of day ("at
  night", "after dark"), frequency ("every weekend"), or period ("since
  covid", "the last few years"). Same rule: copy the exact words, never
  paraphrase. OMIT the "t" key entirely if there are none.
- "a": "s" if the response proposes a concrete, implementable action (names a
  place, mechanism, or particular change — "add lighting on Story Road");
  "g" if it is a broad wish, complaint, or condition ("fix crime", "too much
  trash"). Judge the response, not the topic.
- "e": 1 if the response reports a particular thing that actually happened to
  a particular person — the respondent, their household, or someone they
  refer to ("I was robbed at the light rail station", "my car window got
  smashed", "my neighbor's house was broken into", "I saw someone get jumped
  outside the arena"). 0 otherwise.
  The line is a specific incident vs. an ongoing state of affairs. A recurring
  or general condition is NOT an event, even when the respondent clearly
  witnesses it: "people shoot up on that corner every day" is 0, but
  "someone tried to break into my car last month" is 1. Opinions, complaints,
  and proposals are always 0 ("crime is out of control", "we need more
  lighting"). Do not infer an incident the response does not actually
  describe. Always include "e" — never omit it.
- Code what the response says, not what you assume the respondent meant.
- Never estimate counts, frequencies or percentages.

Return ONLY valid JSON, compact, no spaces after colons or commas. Exactly
this shape — the second object shows a response with no place, no time
phrase, and nothing in the taxonomy that fits:
{{"responses":[{{"n":1,"l":["q_001"],"f":3,"p":["bus stop"],"t":["at night"],"a":"g","e":0}},{{"n":2,"l":[],"f":1,"a":"g","e":0}}]}}
"""

LABEL_USER = """Responses to code ({n} total):
{numbered_responses}
"""

def prompt_hash(dataset_description: str = "") -> str:
    """Version stamp for the LABELING prompt, mirroring induction.prompt_hash.

    Deliberately separate from the induction hash: labels runs used to be
    stamped with it, which meant an edit to LABEL_SYSTEM left no trace at all.
    Measured on q2, a labeling-prompt change reshuffles roughly a third of
    per-response label sets while leaving aggregate counts stable — so two
    runs against the same taxonomy can legitimately disagree row by row, and
    without this there is nothing in the artifact saying why.

    Kept adjacent to the templates it hashes so the two do not drift apart."""
    blob = (LABEL_SYSTEM + LABEL_USER + (dataset_description or "")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


VALID_ACTIONABILITY = {"specific", "general"}
ACTIONABILITY_ALIASES = {"s": "specific", "g": "general"}

# The prompt asks for abbreviated keys to cut output tokens, but the wire
# format is an optimization, not something we can enforce — a model that
# reverts to the verbose names must still parse rather than silently produce
# a batch of empty labels. Every field is read through _field(), which accepts
# either spelling.
WIRE_KEYS = {
    "label_ids": "l", "fit": "f", "locations": "p",
    "time_context": "t", "actionability": "a", "event_occurred": "e",
}


def _field(record: dict, long: str, default=None):
    """Read one field by its compact key, falling back to the verbose key."""
    short = WIRE_KEYS[long]
    if short in record:
        return record[short]
    if long in record:
        return record[long]
    return default


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


def _coerce_bool(raw) -> bool:
    """A real JSON bool is the normal case, but a model sometimes emits the
    string "false" — which is truthy in Python and would silently invert the
    flag. Anything unrecognized falls back to False, the negative case."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "yes", "1"}
    return bool(raw) if isinstance(raw, (int, float)) else False


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
        for raw_id in _field(r, "label_ids") or []:
            rid = str(raw_id).strip()
            if rid not in valid_ids:
                stats["invalid_ids"] += 1
            elif rid not in seen:
                seen.add(rid)
                ids.append(rid)
        try:
            fit = int(_field(r, "fit", 0))
        except (TypeError, ValueError):
            fit = 0
        source_text = batch[n - 1][1]
        locations, n_bad_loc = _verbatim_spans(_field(r, "locations"), source_text)
        stats["invalid_locations"] += n_bad_loc
        times, n_bad_time = _verbatim_spans(_field(r, "time_context"), source_text)
        stats["invalid_time_context"] += n_bad_time
        actionability = str(_field(r, "actionability", "") or "").strip().lower()
        actionability = ACTIONABILITY_ALIASES.get(actionability, actionability)
        got[n] = {
            "label_ids": ids,
            # The prompt no longer asks for an explicit uncategorized flag —
            # an empty label list already means it. Still honoured when a model
            # volunteers it, so this stays behaviour-identical to the verbose
            # schema rather than changing what the contradiction case resolves to.
            "uncategorized": bool(r.get("uncategorized")) or not ids,
            "fit": fit if 1 <= fit <= 3 else None,
            "locations": locations,
            "time_context": times,
            "actionability": (actionability
                              if actionability in VALID_ACTIONABILITY else None),
            "event_occurred": _coerce_bool(_field(r, "event_occurred")),
        }

    assignments = []
    for i, (key, text) in enumerate(batch, start=1):
        a = got.get(i)
        if a is None:
            stats["missing"] += 1
            a = {"label_ids": [], "uncategorized": True, "fit": None,
                 "locations": [], "time_context": [],
                 "actionability": None, "event_occurred": False,
                 "not_returned": True}
        assignments.append({"response_key": key,
                            "respondent_key": respondent_key(key), **a})
    return assignments, stats


def _label_one_batch(
    bi: int,
    batch: list[tuple[str, str]],
    taxonomy: dict,
    client: ModelClient,
    valid_ids: set[str],
    dataset_description: str,
) -> tuple[int, list[dict], dict, dict | None]:
    """One batch, start to finish. Returns (batch_index, assignments, stats,
    failure or None). Raises nothing — a failed batch leaves its responses
    unlabelled and says so, rather than aborting a run that is 90% done."""
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
        failure = {"batch": bi, "n_responses": len(batch),
                   "error": f"{type(exc).__name__}: {exc}"[:200]}
        got = [{"response_key": k, "respondent_key": respondent_key(k),
                "label_ids": [], "uncategorized": True,
                "fit": None, "locations": [],
                "time_context": [], "actionability": None,
                "event_occurred": False, "batch_failed": True}
               for k, _ in batch]
        stats = {"invalid_ids": 0, "out_of_range": 0, "missing": len(batch),
                 "invalid_locations": 0, "invalid_time_context": 0}
        return bi, got, stats, failure
    return bi, got, stats, None


def run_labeling(
    rows: list[tuple[str, str]],
    taxonomy: dict,
    client: ModelClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dataset_description: str = "",
    progress: bool = True,
    workers: int = DEFAULT_WORKERS,
    retry_failed: bool = True,
) -> tuple[list[dict], dict]:
    """Label every response. Returns (assignments, report).

    Exact-duplicate texts are labelled once and the record fanned out to
    every row carrying that text (with each row's own keys). Verbatim short
    answers repeat heavily in survey corpora, temperature is 0, and the
    coding depends only on the text — so this cuts spend proportionally
    while making duplicate rows coded identically by construction. The
    collapse count is disclosed in the report."""
    valid_ids = {lab["label_id"] for lab in taxonomy["labels"]}
    rep_of_text: dict[str, str] = {}
    unique_rows: list[tuple[str, str]] = []
    for k, t in rows:
        if t not in rep_of_text:
            rep_of_text[t] = k
            unique_rows.append((k, t))
    batches = [unique_rows[i : i + batch_size]
               for i in range(0, len(unique_rows), batch_size)]
    totals = {"invalid_ids": 0, "out_of_range": 0, "missing": 0,
              "invalid_locations": 0, "invalid_time_context": 0}

    # Results are collected as they complete but reassembled in batch order:
    # concurrency must not make the output depend on which call returned first.
    done: dict[int, tuple[list[dict], dict, dict | None]] = {}
    n_workers = max(1, min(workers, len(batches))) if batches else 1
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_label_one_batch, bi, batch, taxonomy, client,
                        valid_ids, dataset_description)
            for bi, batch in enumerate(batches)
        ]
        for n_finished, fut in enumerate(as_completed(futures), start=1):
            bi, got, stats, failure = fut.result()
            done[bi] = (got, stats, failure)
            if progress:
                if failure:
                    print(f"  [{n_finished}/{len(batches)}] batch {bi + 1}: "
                          f"FAILED ({failure['error'].split(':')[0]})")
                else:
                    n_unc = sum(1 for a in got if a["uncategorized"])
                    print(f"  [{n_finished}/{len(batches)}] batch {bi + 1}: "
                          f"{len(got)} coded, {n_unc} uncategorized, "
                          f"{stats['invalid_ids']} invalid ids")

    # One serial retry sweep for failed batches, after the concurrent storm
    # has passed. At production scale (hundreds of batch calls) at least one
    # transient failure is the expected case, and without this each one
    # silently turns a whole batch into uncategorized rows.
    failed_idxs = [bi for bi in range(len(batches)) if done[bi][2] is not None]
    n_recovered = 0
    if retry_failed and failed_idxs:
        if progress:
            print(f"  retrying {len(failed_idxs)} failed batch(es) serially…")
        for bi in failed_idxs:
            bi2, got, stats, failure = _label_one_batch(
                bi, batches[bi], taxonomy, client, valid_ids, dataset_description)
            if failure is None:
                done[bi] = (got, stats, None)
                n_recovered += 1
                if progress:
                    print(f"  batch {bi + 1}: recovered on retry")

    assignments: list[dict] = []
    failed_batches: list[dict] = []
    for bi in range(len(batches)):
        got, stats, failure = done[bi]
        if failure:
            failed_batches.append(failure)
        for k in totals:
            totals[k] += stats[k]
        assignments.extend(got)

    # fan each representative's record back out to its duplicate rows, in
    # the original row order, so the output is indistinguishable from having
    # labelled every row individually (minus the model drift)
    if len(unique_rows) != len(rows):
        by_key = {a["response_key"]: a for a in assignments}
        expanded: list[dict] = []
        for k, t in rows:
            rep = by_key[rep_of_text[t]]
            if rep["response_key"] == k:
                expanded.append(rep)
            else:
                dup = json.loads(json.dumps(rep))
                dup["response_key"] = k
                dup["respondent_key"] = respondent_key(k)
                expanded.append(dup)
        assignments = expanded

    labelled = [a for a in assignments if a["label_ids"]]
    counts: dict[str, int] = {lab["label_id"]: 0 for lab in taxonomy["labels"]}
    for a in labelled:
        for lid in a["label_ids"]:
            counts[lid] += 1
    per_response = [len(a["label_ids"]) for a in labelled]

    report = {
        "schema_version": SCHEMA_VERSION,
        "batch_size": batch_size,
        "workers": n_workers,
        "n_batches": len(batches),
        "duplicate_responses_collapsed": len(rows) - len(unique_rows),
        "failed_batches": failed_batches,
        "failed_batches_retried": len(failed_idxs) if retry_failed else 0,
        "failed_batches_recovered": n_recovered,
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
        "responses_with_event_occurred": sum(
            1 for a in assignments if a.get("event_occurred")),
        # the honest denominator for an event rate: rows the model actually
        # coded. Unreturned and failed-batch rows default to False, so
        # dividing by responses_total would understate the rate.
        "responses_event_coded": sum(
            1 for a in assignments
            if not a.get("not_returned") and not a.get("batch_failed")),
        "labels_per_response_mean": (
            round(sum(per_response) / len(per_response), 2) if per_response else 0
        ),
        "label_counts": counts,
        "labels_with_zero_responses": [
            lab["label_id"] for lab in taxonomy["labels"] if counts[lab["label_id"]] == 0
        ],
        "fit_distribution": {},
    }
    fit_counter: dict[str, int] = {}
    act_counter: dict[str, int] = {}
    for a in assignments:
        fit_counter[str(a["fit"])] = fit_counter.get(str(a["fit"]), 0) + 1
        act_counter[str(a["actionability"])] = act_counter.get(str(a["actionability"]), 0) + 1
    report["fit_distribution"] = dict(sorted(fit_counter.items()))
    report["actionability_distribution"] = dict(sorted(act_counter.items()))
    return assignments, report
