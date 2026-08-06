"""Phase 5: query router — an analyst question in, a grounded answer out.

Two model calls, everything between them deterministic:

  1. ROUTE: the model sees the analyst's question and the Phase 4 taxonomy
     summary, and returns candidate child categories (each with a rationale
     and relevance), a route (retrieval / aggregate / comparative / hybrid),
     and optionally lexicon concepts worth a keyword count. An empty
     selection is explicitly permitted — a question the data can't answer
     returns nothing rather than the nearest plausible match. Invented
     label ids are dropped and counted, never trusted.

  2. SYNTH: the model writes the answer from evidence assembled in code:
     computed counts (counting assignment rows — a model never produces a
     number) and verbatim quotes, numbered in the prompt and resolved back
     to response_key in code, so every claim traces to source rows. When a
     category has more members than the quote budget, a seeded sample is
     shown and the prompt says so.

The routing structure mirrors labeling's guards: numbered references
resolved in code, invalid output dropped and counted, empty results treated
as correct results.
"""
from __future__ import annotations

import json
import random
import re

from .induction import context_block, extract_json
from .labeling import ACTIONABILITY_ALIASES, VALID_ACTIONABILITY
from .llm import ModelClient

SCHEMA_VERSION = 1
VALID_ROUTES = {"retrieval", "aggregate", "comparative", "hybrid"}
VALID_RELEVANCE = {"high", "medium", "low"}
# how each actionability value reads in a prompt and in disclosure text —
# "specific"/"general" are the stored codes, these are the English
ACTIONABILITY_PHRASE = {
    "specific": "proposing a specific, concrete action",
    "general": "raising a general concern rather than a concrete action",
}
# These caps protect the synth model's ATTENTION, not the bill (~22 tokens per
# response, so even 500 quotes is only ~15k prompt tokens).
#
# They were raised to 50/500 on 2026-08-03 to stop a uniform draw missing
# minority signals, on the math that a 10%-prevalence sub-signal is missed
# 0.5% of the time at 50 quotes vs 35% at 10. That math is right and it
# optimizes the wrong quantity: what matters is P(signal reaches the ANSWER)
# = P(in prompt) x P(the model actually uses it), and the second term
# collapses as the prompt grows. Measured over 40 answers on dataset 2
# (2026-08-05), median share of shown quotes cited: 55% at 60, 35% at 120,
# 21% at 150-250, 7.8% at 500 — so the raise cut end-to-end signal delivery
# from ~0.18 to ~0.05 while looking like an improvement on paper.
#
# Worse, at 500 the answer silently drops whole categories the analyst
# selected: median category coverage 100% at 120 vs 90% at 500, and on a
# 16-candidate question 75% -> 37.5%.
#
# 10/120 won: same median citations as 20/120 (26) with longer answers
# (9,514 vs 8,557 chars), higher utilisation (30% vs 23%) and a smaller,
# cheaper prompt. Do not raise these without re-running that sweep — this
# failure is invisible in latency and cost, which both *improve* as the
# budget grows, and invisible in the sampling math, which improves too.
DEFAULT_MAX_QUOTES_PER_LABEL = 10
DEFAULT_MAX_TOTAL_QUOTES = 120
MIN_QUOTES_PER_LABEL = 3
MAX_QUOTE_CHARS = 400

ROUTE_SYSTEM = """\
{dataset_context}You are routing an analyst's question about a coded open-ended survey.

Below is the complete category summary. Each line is one child category:
`label_id | parent theme > name — description (n=count)`. The counts are
real, computed from the coded data — never re-estimate or adjust them.

{summary}

Decide how to answer the analyst's question:

- "answerable": false if this data cannot answer the question — wrong
  domain, information the survey never collected, or judgment the responses
  do not contain. Returning nothing is CORRECT and expected in that case;
  never select the nearest plausible category just to return something.
- "route": one of
  - "retrieval" — the question asks WHAT people say; answer by reading responses
  - "aggregate" — the question asks how often / what is most common; answer from counts
  - "comparative" — the question asks how groups of responses differ; contrast them
  - "hybrid" — aggregate first, then explain the top categories from responses
- "candidates": EVERY child category relevant to the question, each with a
  relevance rating and a rationale of AT MOST 10 WORDS — a fragment, not a
  sentence. For example a rationale might read: direct match, trash on
  sidewalks. There is no cap on how MANY candidates — include all genuinely
  relevant categories, and nothing else. Copy label_id exactly.
- "reason": at most 15 words.
- Never write a double quote inside any rationale or reason: it breaks the
  JSON. Use plain words or a comma instead.
- "groups": ONLY for route "comparative": two or more named groups of
  label_ids to contrast (e.g. downtown categories vs neighborhood ones).
- "lexicon_concepts": names from the lexicon concept list (if shown) whose
  exact-keyword counts would strengthen the answer. Empty list if none.
- "group_by": "category" normally. Set "location" ONLY when the question asks
  WHERE something happens or which places are affected — the answer is then
  organized by place instead of by category. Candidates are still required:
  they define which responses are in scope. Only valid when a Locations list
  is shown below.
- "location_filter": names copied from the Locations list (if shown) ONLY
  when the question names specific places ("what do people say about
  downtown?") — evidence is then restricted to responses mentioning them.
  Leave it EMPTY for a general where-question: group_by "location" already
  organizes by place, and adding a broad filter would hide how many
  responses named no place at all.
{actionability_block}{event_block}{time_block}- Never estimate counts, frequencies, or percentages.

Return ONLY valid JSON, exactly this shape:
{{"answerable": true, "route": "retrieval", "reason": "one line",
"candidates": [{{"label_id": "2_001", "relevance": "high", "rationale": "..."}}],
"groups": [], "lexicon_concepts": [], "group_by": "category",
"location_filter": [], "actionability_filter": "", "event_filter": "",
"time_filter": ""}}
"""

# Only shown when the labels actually carry an actionability code — a filter
# the data can't honour must never be advertised to the router.
ACTIONABILITY_BLOCK = """\
- "actionability_filter": every coded response is marked either "specific"
  (it proposes a concrete, implementable action — names a place, mechanism,
  or particular change) or "general" (a broad wish, complaint, or condition).
  Set this to "specific" ONLY when the question asks what should be DONE —
  what to fix, build, change, prioritise, or which quick wins to pursue.
  Set it to "general" only when the question is explicitly about broad
  sentiment or vague concerns. Leave it "" for everything else, including
  ordinary "what do people say about X" questions: the filter drops most
  responses, so use it only when the analyst wants proposals rather than
  opinions. In this dataset {coded} coded responses carry the mark
  ({counts}).
"""

# Deliberately one-directional. `event_occurred=false` means "this response
# does not describe a specific incident" — NOT "this respondent has not
# experienced one"; people simply don't recount incidents in a one-line answer.
# Offering the negative as a filter would invite a confident wrong contrast
# ("what people who haven't been victimised think"), so it is not offered.
EVENT_BLOCK = """\
- "event_filter": {n_events} of the {n_coded} coded responses recount a
  specific thing that actually happened to a particular person ("my car
  window got smashed", "I was robbed at the light rail station"), as opposed
  to an opinion, a proposal, or an ongoing condition. When the analyst asks
  what people have personally experienced, witnessed, or had happen to them,
  set this to "reported" and select the relevant categories as usual — those
  {n_events} responses are real evidence, so such a question IS answerable
  and must not be refused. Leave it "" for questions about opinion,
  prevalence, or what people want, where the filter would wrongly discard
  most of the data. There is no option to select responses WITHOUT an
  incident: not recounting one does not mean nothing happened.
"""

# Like EVENT_BLOCK: capability first, severity after, and no negative
# direction — a response naming no time of day is not evidence about when
# anything happened.
TIME_BLOCK = """\
- "time_filter": {n_night} coded responses explicitly mention nighttime
  ("at night", "after dark") and {n_day} mention daytime, out of
  {n_mentioned} naming any time at all. When the analyst asks specifically
  about experiences at night or during the day, set this to "night" or
  "day" — those responses are real evidence, so such a question IS
  answerable and must not be refused. Leave it "" otherwise. There is no
  filter for responses naming no time: not naming one does not mean it
  happened at any particular time of day.
"""

ROUTE_USER = """Analyst question:
{question}
"""

SYNTH_SYSTEM = """\
{dataset_context}You are answering an analyst's question about an open-ended survey, using
ONLY the evidence provided: computed counts and verbatim responses.

Rules:
- NEVER produce a number of your own. Every count, percentage, or "most
  common" claim must come from the COMPUTED COUNTS section, copied exactly.
  If a number is not there, do not state one — write "several" or name the
  categories instead.
- Ground every claim in the verbatim responses and cite them by number in
  square brackets, e.g. [12] or [3][17]. Cite only numbers that appear in
  the VERBATIM RESPONSES section.
- Quote only text that appears in the responses shown. Never invent or
  embellish a quote.
- Some categories show a sample of their responses (marked "showing k of
  n"); the counts always cover the full data.
- If the evidence does not actually answer the question, say so plainly —
  a clear "the data does not answer this" is a correct answer.
- Format the answer for fast scanning, in markdown:
  - Start with a 1-2 sentence takeaway that directly answers the question,
    with its key counts in **bold**.
  - Then organize the detail under short "### " subheadings (3-6 words
    each), one per distinct theme or cost/issue type.
  - Inside each section, **bold** the key finding and use "- " bullets for
    lists of specifics. Keep paragraphs to 2-3 sentences.
  - Ground each section in AT LEAST 3 distinct cited verbatims (more is
    fine) whenever that many relevant responses were shown — a section
    resting on one or two citations under-uses the evidence. Never pad with
    irrelevant citations if fewer than 3 apply.
  - If the whole answer fits in one or two short paragraphs, skip the
    subheadings — do not pad a short answer with structure.
- {route_guidance}

Return ONLY valid JSON, exactly this shape:
{{"answer_markdown": "..."}}
"""

ROUTE_GUIDANCE = {
    "retrieval": (
        "Organize the answer around the SPECIFICS respondents raise, not the "
        "category names — read the quotes and report what people actually say."),
    "aggregate": (
        "Lead with the computed counts, narrating them exactly as given; use "
        "quotes only to illustrate what a category means."),
    "comparative": (
        "Summarize each group separately from its own responses, then contrast "
        "the groups directly. Only use numbers from COMPUTED COUNTS."),
    "hybrid": (
        "First establish which categories dominate using the computed counts, "
        "then explain WHY using the quoted responses from those categories."),
}

SYNTH_USER = """Analyst question:
{question}

COMPUTED COUNTS (real, computed from the coded data):
{counts_block}

VERBATIM RESPONSES:
{quotes_block}
"""


# ---------------------------------------------------------------------------
# ROUTE call
# ---------------------------------------------------------------------------


def build_route_prompts(question: str, summary_text: str,
                        dataset_description: str = "",
                        actionability_counts: dict[str, int] | None = None,
                        event_counts: tuple[int, int] | None = None,
                        time_counts: dict[str, int] | None = None,
                        ) -> tuple[str, str]:
    """`event_counts` is (n_reporting_an_incident, n_coded); `time_counts`
    is {"day": n, "night": n, "mentioned": n}. All optional blocks are
    omitted entirely when the artifacts don't carry the field, so the
    router is never offered a filter the data cannot honour."""
    act_block = ""
    if actionability_counts:
        act_block = ACTIONABILITY_BLOCK.format(
            coded=sum(actionability_counts.values()),
            counts=", ".join(f"{n} {v}" for v, n in
                             sorted(actionability_counts.items())),
        )
    evt_block = ""
    if event_counts and event_counts[0]:
        evt_block = EVENT_BLOCK.format(n_events=event_counts[0],
                                       n_coded=event_counts[1])
    time_block = ""
    if time_counts and (time_counts.get("day") or time_counts.get("night")):
        time_block = TIME_BLOCK.format(n_night=time_counts.get("night", 0),
                                       n_day=time_counts.get("day", 0),
                                       n_mentioned=time_counts.get("mentioned", 0))
    system = ROUTE_SYSTEM.format(
        dataset_context=context_block(dataset_description),
        summary=summary_text.rstrip(),
        actionability_block=act_block,
        event_block=evt_block,
        time_block=time_block,
    )
    return system, ROUTE_USER.format(question=question.strip())


def normalize_actionability(raw) -> str:
    """Map a requested actionability filter onto a stored code, or "" for no
    filter. Accepts the wire abbreviations labeling emits ("s"/"g") and the
    several ways a model spells "no filter"."""
    v = str(raw or "").strip().lower()
    v = ACTIONABILITY_ALIASES.get(v, v)
    if v in {"", "any", "all", "none", "null", "both"}:
        return ""
    return v if v in VALID_ACTIONABILITY else "invalid"


def normalize_event(raw) -> str:
    """Map a requested event filter onto "" (no filter) or "reported". The
    negative direction is intentionally unsupported — see EVENT_BLOCK."""
    v = str(raw or "").strip().lower()
    if v in {"", "any", "all", "none", "null", "false", "0"}:
        return ""
    if v in {"reported", "true", "1", "yes", "event", "events", "occurred"}:
        return "reported"
    return "invalid"


def normalize_time(raw) -> str:
    """Map a requested time filter onto "" | "day" | "night". No filter for
    responses naming no time — see TIME_BLOCK."""
    v = str(raw or "").strip().lower()
    if v in {"", "any", "all", "none", "null"}:
        return ""
    if v in {"night", "nighttime", "evening", "dark", "after dark"}:
        return "night"
    if v in {"day", "daytime", "daylight", "morning", "afternoon"}:
        return "day"
    return "invalid"


def parse_route_output(raw: str, valid_ids: set[str],
                       valid_concepts: set[str],
                       valid_locations: set[str] = frozenset(),
                       actionability_available: bool = False,
                       events_available: bool = False,
                       time_available: bool = False) -> tuple[dict, dict]:
    """Validate the routing decision. Returns (route, stats). Invented label
    ids, unknown concepts and unknown locations are dropped and counted; a
    malformed route raises so the caller's retry path can handle it."""
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"router returned {type(obj).__name__}, not an object")

    stats = {"invalid_label_ids": 0, "invalid_concepts": 0,
             "invalid_locations": 0, "warnings": []}

    route = str(obj.get("route", "")).strip().lower()
    if route not in VALID_ROUTES:
        stats["warnings"].append(f"unknown route {route!r}, defaulting to retrieval")
        route = "retrieval"

    candidates, seen = [], set()
    for c in obj.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        lid = str(c.get("label_id", "")).strip()
        if lid not in valid_ids:
            stats["invalid_label_ids"] += 1
            continue
        if lid in seen:
            continue
        seen.add(lid)
        relevance = str(c.get("relevance", "")).strip().lower()
        candidates.append({
            "label_id": lid,
            "relevance": relevance if relevance in VALID_RELEVANCE else "medium",
            "rationale": str(c.get("rationale", "")).strip(),
        })

    groups = []
    for g in obj.get("groups") or []:
        if not isinstance(g, dict):
            continue
        name = str(g.get("name", "")).strip()
        ids = [str(i).strip() for i in g.get("label_ids") or []]
        kept = [i for i in ids if i in seen]
        dropped = len(ids) - len(kept)
        if dropped:
            stats["warnings"].append(
                f"group {name!r}: {dropped} id(s) not in the selected candidates, dropped")
        if name and kept:
            groups.append({"name": name, "label_ids": kept})
    if route == "comparative" and len(groups) < 2:
        stats["warnings"].append(
            "comparative route without 2+ valid groups; downgraded to retrieval")
        route, groups = "retrieval", []

    concepts = []
    for raw_c in obj.get("lexicon_concepts") or []:
        c = str(raw_c).strip().lower()
        if c in valid_concepts:
            if c not in concepts:
                concepts.append(c)
        else:
            stats["invalid_concepts"] += 1

    group_by = str(obj.get("group_by", "category")).strip().lower() or "category"
    if group_by not in {"category", "location"}:
        stats["warnings"].append(f"unknown group_by {group_by!r}, defaulting to category")
        group_by = "category"
    location_filter = []
    for raw_l in obj.get("location_filter") or []:
        name = str(raw_l).strip().lower()
        if name in valid_locations:
            if name not in location_filter:
                location_filter.append(name)
        else:
            stats["invalid_locations"] += 1
    if group_by == "location" and not valid_locations:
        stats["warnings"].append(
            "group_by=location but no location layer exists; grouping by category")
        group_by = "category"

    actionability_filter = normalize_actionability(obj.get("actionability_filter"))
    if actionability_filter == "invalid":
        stats["warnings"].append(
            f"unknown actionability_filter "
            f"{str(obj.get('actionability_filter'))!r}, ignored")
        actionability_filter = ""
    if actionability_filter and not actionability_available:
        stats["warnings"].append(
            "actionability_filter requested but no response carries an "
            "actionability code; ignoring the filter")
        actionability_filter = ""

    event_filter = normalize_event(obj.get("event_filter"))
    if event_filter == "invalid":
        stats["warnings"].append(
            f"unknown event_filter {str(obj.get('event_filter'))!r}, ignored")
        event_filter = ""
    if event_filter and not events_available:
        stats["warnings"].append(
            "event_filter requested but no response reports a first-hand "
            "incident; ignoring the filter")
        event_filter = ""

    time_filter = normalize_time(obj.get("time_filter"))
    if time_filter == "invalid":
        stats["warnings"].append(
            f"unknown time_filter {str(obj.get('time_filter'))!r}, ignored")
        time_filter = ""
    if time_filter and not time_available:
        stats["warnings"].append(
            "time_filter requested but no response carries a classified "
            "time-of-day mention; ignoring the filter")
        time_filter = ""

    answerable = bool(obj.get("answerable", True)) and bool(candidates)
    if obj.get("answerable", True) and not candidates:
        stats["warnings"].append(
            "router said answerable but selected no valid categories; treating as unanswerable")

    return {
        "answerable": answerable,
        "route": route,
        "reason": str(obj.get("reason", "")).strip(),
        "candidates": candidates,
        "groups": groups,
        "lexicon_concepts": concepts,
        "group_by": group_by,
        "location_filter": location_filter,
        "actionability_filter": actionability_filter,
        "event_filter": event_filter,
        "time_filter": time_filter,
    }, stats


# ---------------------------------------------------------------------------
# Evidence assembly — deterministic, counts computed here and only here
# ---------------------------------------------------------------------------


def members_by_label(assignments: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in assignments:
        for lid in a.get("label_ids") or []:
            out.setdefault(lid, []).append(a["response_key"])
    return out


def composite_sample(keys: list[str], budget: int, rng: random.Random,
                     tags_of) -> tuple[list[str], dict | None]:
    """Pick `budget` keys as coverage picks + a uniform draw.

    A pure uniform sample loses minority signals: a sub-signal present in 10%
    of a category is entirely absent from a 10-quote draw 35% of the time.
    So half the budget greedily covers "signal tags" — metadata the pipeline
    already computed per response (co-assigned labels, places, actionability,
    event flag, length band) — guaranteeing every distinct signal the data
    carries at least one quote while the budget lasts. The other half stays a
    uniform draw, because coverage picks over-represent unusual responses and
    the answer still needs to reflect what is typical.

    Deterministic: keys are visited in sorted order, ties break on that order,
    and the fill uses the caller's seeded rng. Returns (sorted selection,
    detail-for-disclosure); detail is None when no sampling happened.
    """
    keys = sorted(keys)
    if len(keys) <= budget:
        return keys, None
    tag_sets = {k: tags_of(k) for k in keys}
    all_tags: set[str] = set().union(*tag_sets.values())
    cover_budget = budget // 2
    covered: set[str] = set()
    chosen: list[str] = []
    chosen_set: set[str] = set()
    while len(chosen) < cover_budget and len(covered) < len(all_tags):
        best, best_gain = None, 0
        for k in keys:
            if k in chosen_set:
                continue
            gain = len(tag_sets[k] - covered)
            if gain > best_gain:
                best, best_gain = k, gain
        if best is None:
            break
        chosen.append(best)
        chosen_set.add(best)
        covered |= tag_sets[best]
    remaining = [k for k in keys if k not in chosen_set]
    n_fill = budget - len(chosen)
    fill = rng.sample(remaining, n_fill) if len(remaining) > n_fill else remaining
    detail = {"coverage_picks": len(chosen), "random_picks": len(fill),
              "tags_covered": len(covered), "tags_total": len(all_tags)}
    return sorted(chosen + fill), detail


def _sampling_note(shown: int, total: int, detail: dict | None = None,
                   n_withheld: int = 0, withheld_reason: str = "") -> str:
    """The disclosure string for one category/place, or "" when the quotes
    shown really are all of them.

    Keeps the "showing k of n" prefix the quote headers and manifest have
    always used, then says how the sample was composed — an undisclosed
    stratification would read as a uniform draw, which it no longer is.

    `n_withheld` covers responses counted in `total` that were deliberately
    not quotable here. In group_by=location a response naming two places is
    quoted once, under the higher-count place; without this the second place's
    header rendered as "all n" while showing fewer, telling the synth model it
    had that place's complete evidence when it did not.
    """
    if shown >= total and not detail and not n_withheld:
        return ""
    note = f"showing {shown} of {total}"
    if detail and detail["coverage_picks"]:
        note += (f" ({detail['coverage_picks']} covering "
                 f"{detail['tags_covered']} signal tags — co-labels, places, "
                 f"actionability, events, length; {detail['random_picks']} random")
        uncovered = detail["tags_total"] - detail["tags_covered"]
        if uncovered:
            note += f"; {uncovered} tags uncovered"
        note += ")"
    if n_withheld and withheld_reason:
        note += f"; {n_withheld} {withheld_reason}"
    return note


def gather_evidence(
    route: dict,
    index: dict[str, dict],                    # label_id -> summary entry (+question_id)
    members: dict[str, list[str]],             # label_id -> response_keys (full corpus)
    texts: dict[str, str],                     # response_key -> verbatim text
    max_quotes_per_label: int = DEFAULT_MAX_QUOTES_PER_LABEL,
    max_total_quotes: int = DEFAULT_MAX_TOTAL_QUOTES,
    seed: int = 7,
    location_members: dict[str, list[str]] | None = None,  # concept -> response_keys
    location_kinds: dict[str, str] | None = None,          # concept -> named|type
    actionability_of: dict[str, str] | None = None,        # response_key -> specific|general
    event_keys: set[str] | None = None,        # responses reporting an incident
    event_coded_keys: set[str] | None = None,  # responses the pass actually coded
    location_implicit_questions: dict[str, set[str]] | None = None,
    time_day_keys: set[str] | None = None,     # day/night from time_context spans
    time_night_keys: set[str] | None = None,
    time_mentioned_keys: set[str] | None = None,
) -> dict:
    """Counts + numbered quotes for the synth prompt. Sampling is seeded,
    composite (coverage picks over signal tags + a uniform draw — see
    composite_sample) and disclosed; counts always cover the full (possibly
    filtered) membership.

    Three orthogonal filters compose here, each restricting the same evidence
    set and each leaving the unfiltered per-category count behind for
    disclosure: location_filter (responses mentioning given places),
    actionability_filter (responses marked as proposing a concrete action vs
    raising a general concern), and event_filter (responses describing an
    incident that actually happened to someone). They apply in that fixed
    order, and each one's denominator is measured against the scope the
    previous filters left — so "460 of 933" always reads "of the responses
    that survived everything before me". With group_by=location, quotes and
    counts are additionally organized per place. Every denominator is
    computed here — never estimated."""
    loc_filter = route.get("location_filter") or []
    act_filter = route.get("actionability_filter") or ""
    evt_filter = route.get("event_filter") or ""
    time_filter = route.get("time_filter") or ""
    group_by = route.get("group_by", "category")

    allowed: set[str] | None = None

    def scope_under(allow: set[str] | None) -> set[str]:
        """The union of selected-category members surviving `allow`."""
        return {k for c in route["candidates"]
                for k in members.get(c["label_id"], [])
                if allow is None or k in allow}

    def restrict(keys: set[str]) -> set[str]:
        return keys if allowed is None else (allowed & keys)

    implicit_qs: list[str] = []
    if loc_filter and location_members:
        hits: set[str] = set()
        for name in loc_filter:
            hits.update(location_members.get(name, []))
        # A question whose own wording names the place ("changes to improve
        # downtown") makes every answer to it about that place by
        # construction — those responses match implicitly instead of being
        # dropped for not re-typing the place name. Disclosed downstream.
        if location_implicit_questions:
            iq = {q for name in loc_filter
                  for q in location_implicit_questions.get(name, set())}
            if iq:
                for c in route["candidates"]:
                    if index[c["label_id"]]["question_id"] in iq:
                        hits.update(members.get(c["label_id"], []))
                implicit_qs = sorted(
                    iq & {index[c["label_id"]]["question_id"]
                          for c in route["candidates"]})
        allowed = restrict(hits)

    # `coded` is tracked separately from `matching` because a response the
    # labeling pass never coded is not evidence of absence — it is missing
    # data, and every disclosure says so rather than folding it into the
    # negative case.
    actionability_denominator: dict[str, int] | None = None
    if act_filter and actionability_of:
        pre = scope_under(allowed)
        hits = {k for k, v in actionability_of.items() if v == act_filter}
        actionability_denominator = {
            "in_scope": len(pre),
            "coded": sum(1 for k in pre if actionability_of.get(k)),
            "matching": len(pre & hits),
        }
        allowed = restrict(hits)

    event_denominator: dict[str, int] | None = None
    if evt_filter and event_keys is not None:
        pre = scope_under(allowed)
        event_denominator = {
            "in_scope": len(pre),
            "coded": len(pre if event_coded_keys is None
                         else pre & event_coded_keys),
            "matching": len(pre & event_keys),
        }
        allowed = restrict(set(event_keys))

    # `mentioning` is tracked because a response naming no time of day is not
    # evidence about when anything happened — only the responses that named
    # one are classifiable, and the disclosure must say so.
    time_denominator: dict[str, int] | None = None
    if time_filter:
        t_keys = time_night_keys if time_filter == "night" else time_day_keys
        if t_keys is not None:
            pre = scope_under(allowed)
            time_denominator = {
                "in_scope": len(pre),
                "mentioning": len(pre & (time_mentioned_keys or set())),
                "matching": len(pre & t_keys),
            }
            allowed = restrict(set(t_keys))

    def member_keys(lid: str) -> list[str]:
        keys = members.get(lid, [])
        return [k for k in keys if k in allowed] if allowed is not None else keys

    selection = []
    for c in route["candidates"]:
        e = index[c["label_id"]]
        keys = member_keys(c["label_id"])
        entry = {
            "label_id": c["label_id"],
            "name": e["name"],
            "parent_name": e.get("parent_name"),
            "question_id": e["question_id"],
            "count": len(keys),
            "relevance": c["relevance"],
            "rationale": c["rationale"],
        }
        if allowed is not None:
            entry["count_unfiltered"] = len(members.get(c["label_id"], []))
        selection.append(entry)

    # the (possibly filtered) union of selected-category members — the one
    # "responses in scope" figure every disclosure downstream should use
    unique_keys = {k for c in route["candidates"] for k in member_keys(c["label_id"])}

    rng = random.Random(seed)
    quotes, n = [], 0
    sampling_notes: dict[str, str] = {}

    # Signal tags for composite sampling — the inversions (key -> labels,
    # key -> places) cover the whole corpus, so they are built once and only
    # if some category actually overflows its quote budget.
    _tag_ctx: dict | None = None

    def tag_context() -> dict:
        nonlocal _tag_ctx
        if _tag_ctx is None:
            labels_of: dict[str, list[str]] = {}
            for l, ks in members.items():
                for k in ks:
                    labels_of.setdefault(k, []).append(l)
            places_of: dict[str, list[str]] = {}
            for name, ks in (location_members or {}).items():
                for k in ks:
                    places_of.setdefault(k, []).append(name)
            _tag_ctx = {"labels_of": labels_of, "places_of": places_of}
        return _tag_ctx

    def tags_for(sample_keys: list[str], exclude_lid: str = ""):
        """tags_of callable for one category/place: length bands are terciles
        WITHIN this group, so "long for this question" stays meaningful."""
        ctx = tag_context()
        lens = sorted(len(texts[k]) for k in sample_keys)
        t1, t2 = lens[len(lens) // 3], lens[(2 * len(lens)) // 3]

        def tags_of(k: str) -> set[str]:
            ln = len(texts[k])
            tags = {"len:" + ("short" if ln <= t1 else
                              "long" if ln > t2 else "mid")}
            for l in ctx["labels_of"].get(k, ()):
                if l != exclude_lid:
                    tags.add("colabel:" + l)
            for p in ctx["places_of"].get(k, ()):
                tags.add("place:" + p)
            v = (actionability_of or {}).get(k)
            if v:
                tags.add("act:" + v)
            if event_keys and k in event_keys:
                tags.add("evt:reported")
            return tags

        return tags_of

    def add_quote(key: str, lid: str, location: str | None = None) -> None:
        nonlocal n
        n += 1
        t = texts[key].replace("\n", " ").strip()
        if len(t) > MAX_QUOTE_CHARS:
            t = t[:MAX_QUOTE_CHARS] + "…"
        q = {"n": n, "response_key": key, "label_id": lid, "text": t}
        if location is not None:
            q["location"] = location
        quotes.append(q)

    location_counts: list[dict] = []
    location_denominator: dict | None = None

    if group_by == "location" and location_members:
        # scope = responses in the selected categories; per-place counts are
        # intersections computed here, and the denominator says how many
        # in-scope responses are localizable at all
        scope = unique_keys
        owner: dict[str, str] = {}          # response_key -> a selected label_id
        for c in route["candidates"]:
            for k in member_keys(c["label_id"]):
                owner.setdefault(k, c["label_id"])
        loc_sets = {name: scope & set(keys)
                    for name, keys in location_members.items()}
        loc_sets = {name: s for name, s in loc_sets.items() if s}
        location_counts = sorted(
            ({"name": name, "kind": (location_kinds or {}).get(name, "type"),
              "count": len(s)} for name, s in loc_sets.items()),
            key=lambda lc: (-lc["count"], lc["name"]))
        naming_any = len(set().union(*loc_sets.values())) if loc_sets else 0
        location_denominator = {"in_scope": len(scope), "naming_any": naming_any}

        # Budget = floor division so the total cap actually binds — the old
        # decrement-to-a-floor loop silently blew past max_total_quotes once
        # places outnumbered budget/MIN_QUOTES_PER_LABEL.
        per_loc = max_quotes_per_label
        quotable_locs = {lc["name"] for lc in location_counts}
        if loc_sets:
            per_loc = max(1, min(max_quotes_per_label,
                                 max_total_quotes // len(loc_sets)))
            if per_loc < MIN_QUOTES_PER_LABEL:
                note = (f"{len(loc_sets)} places share the {max_total_quotes}-"
                        f"quote budget; showing up to {per_loc} per place "
                        "(counts always cover the full data)")
                if len(loc_sets) > max_total_quotes:
                    # more places than quotes: quote the largest ones only
                    # (location_counts is already sorted by count desc)
                    quotable_locs = {lc["name"]
                                     for lc in location_counts[:max_total_quotes]}
                    note = (f"{len(loc_sets)} places selected; quotes shown for "
                            f"the {max_total_quotes} most-mentioned, counts "
                            "cover all")
                sampling_notes["_quote_budget:location"] = note
        quoted: set[str] = set()            # a response quoted once, under its
        for lc in location_counts:          # highest-count place
            if lc["name"] not in quotable_locs:
                continue
            available = sorted(k for k in loc_sets[lc["name"]]
                               if k in texts and k not in quoted)
            # counted in lc["count"] but not quotable here — already quoted
            # under a higher-count place, or absent from the text map. Must be
            # disclosed: otherwise this place's header claims "all n".
            n_withheld = lc["count"] - len(available)
            keys, detail = available, None
            if len(available) > per_loc:
                keys, detail = composite_sample(available, per_loc, rng,
                                                tags_for(available))
            note = _sampling_note(len(keys), lc["count"], detail, n_withheld,
                                  "already quoted under another place")
            if note:
                sampling_notes[f"loc:{lc['name']}"] = note
            for k in keys:
                quoted.add(k)
                add_quote(k, owner.get(k, ""), location=lc["name"])
    else:
        # Same budget arithmetic as the location branch: identical to the old
        # loop while floor(budget/n) >= MIN_QUOTES_PER_LABEL, but past that
        # the cap now binds (2, then 1 quote per category) instead of the
        # floor silently defeating it.
        candidates = route["candidates"]
        per_label = max_quotes_per_label
        quotable_lids = {c["label_id"] for c in candidates}
        keys_by_lid = {c["label_id"]: member_keys(c["label_id"])
                       for c in candidates}
        if candidates:
            per_label = max(1, min(max_quotes_per_label,
                                   max_total_quotes // len(candidates)))
            if per_label < MIN_QUOTES_PER_LABEL:
                note = (f"{len(candidates)} categories share the "
                        f"{max_total_quotes}-quote budget; showing up to "
                        f"{per_label} per category (counts always cover the "
                        "full data)")
                if len(candidates) > max_total_quotes:
                    ranked = sorted(candidates,
                                    key=lambda c: (-len(keys_by_lid[c["label_id"]]),
                                                   c["label_id"]))
                    quotable_lids = {c["label_id"]
                                     for c in ranked[:max_total_quotes]}
                    note = (f"{len(candidates)} categories selected; quotes "
                            f"shown for the {max_total_quotes} largest, "
                            "counts cover all")
                sampling_notes["_quote_budget"] = note
        for c in candidates:
            lid = c["label_id"]
            if lid not in quotable_lids:
                continue
            all_keys = keys_by_lid[lid]
            available = [k for k in all_keys if k in texts]
            # a member the corpus has no text for (a labels run referencing
            # rows a re-ingest dropped) is counted but unquotable — say so
            # rather than letting the header read "all n"
            n_withheld = len(all_keys) - len(available)
            keys, detail = available, None
            if len(available) > per_label:
                keys, detail = composite_sample(available, per_label, rng,
                                                tags_for(available, lid))
            note = _sampling_note(len(keys), len(all_keys), detail, n_withheld,
                                  "have no stored response text")
            if note:
                sampling_notes[lid] = note
            for k in keys:
                add_quote(k, lid)

    group_counts = []
    for g in route["groups"]:
        unique = {k for lid in g["label_ids"] for k in member_keys(lid)}
        group_counts.append({"name": g["name"], "label_ids": g["label_ids"],
                             "count_unique_responses": len(unique)})

    return {"selection": selection, "quotes": quotes,
            "sampling_notes": sampling_notes, "group_counts": group_counts,
            "n_unique_responses": len(unique_keys),
            "group_by": group_by, "location_filter": loc_filter,
            "location_counts": location_counts,
            "location_denominator": location_denominator,
            "location_filter_implicit_questions": implicit_qs,
            "actionability_filter": act_filter,
            "actionability_denominator": actionability_denominator,
            "event_filter": evt_filter,
            "event_denominator": event_denominator,
            "time_filter": time_filter,
            "time_denominator": time_denominator}


def lexicon_counts(lexicon: dict, concept_names: list[str],
                   keys_by_question: dict[str, list[str]],
                   texts: dict[str, str]) -> list[dict]:
    """Exact keyword-match counts per question for the selected concepts.
    Deterministic regex matching — the lexicon's whole point."""
    from .lexicon import compile_concept

    by_name = {c["name"]: c for c in lexicon.get("concepts", [])}
    out = []
    for name in concept_names:
        concept = by_name.get(name)
        if not concept:
            continue
        pat = compile_concept(concept["terms"])
        per_q = {q: sum(1 for k in keys if k in texts and pat.search(texts[k]))
                 for q, keys in keys_by_question.items()}
        out.append({"concept": name, "mentions_by_question": per_q,
                    "mentions_total": sum(per_q.values())})
    return out


# ---------------------------------------------------------------------------
# SYNTH call
# ---------------------------------------------------------------------------


def filter_phrase(evidence: dict) -> str:
    """How the active evidence filters read in English — one phrase shared by
    the counts block, the synth guidance and the process note, so the three
    can never describe the same restriction differently."""
    parts = []
    if evidence.get("location_filter"):
        parts.append("mentioning " + ", ".join(evidence["location_filter"]))
    act = evidence.get("actionability_filter")
    if act:
        parts.append(ACTIONABILITY_PHRASE.get(act, act))
    if evidence.get("event_filter"):
        parts.append("recounting a first-hand incident")
    t = evidence.get("time_filter")
    if t:
        parts.append(f"mentioning {t}time" if t in {"day", "night"} else t)
    return " and ".join(parts)


def render_counts_block(evidence: dict, lex_counts: list[dict],
                        question_totals: dict[str, int],
                        question_texts: dict[str, str] | None = None) -> str:
    def q_phrase(qid: str) -> str:
        """How a survey question reads in a count line — its own wording when
        known, so counts from different questions never read as one ranking."""
        text = (question_texts or {}).get(qid, "")
        return f'question {qid} ("{text}")' if text else f"question {qid}"

    lines = []
    for qid in evidence.get("location_filter_implicit_questions") or []:
        lines.append(f'- every response to {q_phrase(qid)} is about '
                     f'{", ".join(evidence.get("location_filter") or [])} by '
                     f'construction (the question itself asks about it), so '
                     f'all of them count as mentioning it')
    denom_act = evidence.get("actionability_denominator")
    if denom_act:
        act = evidence.get("actionability_filter", "")
        lines.append(f'- responses in scope before the {act}-only filter: '
                     f'{denom_act["in_scope"]}')
        lines.append(f'- of those, responses marked "{act}": '
                     f'{denom_act["matching"]} — the answer covers ONLY these, '
                     f'and must state this denominator')
        uncoded = denom_act["in_scope"] - denom_act["coded"]
        if uncoded:
            lines.append(f'- {uncoded} in-scope response'
                         f'{"" if uncoded == 1 else "s"} '
                         f'{"was" if uncoded == 1 else "were"} never marked '
                         f'either way (missing data, not evidence of absence)')
    denom_evt = evidence.get("event_denominator")
    if denom_evt:
        lines.append(f'- responses in scope before the first-hand-incident '
                     f'filter: {denom_evt["in_scope"]}')
        lines.append(f'- of those, responses recounting an incident that '
                     f'actually happened: {denom_evt["matching"]} — the answer '
                     f'covers ONLY these, and must state this denominator')
        uncoded_evt = denom_evt["in_scope"] - denom_evt["coded"]
        if uncoded_evt:
            lines.append(f'- {uncoded_evt} in-scope response'
                         f'{"" if uncoded_evt == 1 else "s"} '
                         f'{"was" if uncoded_evt == 1 else "were"} never '
                         f'checked for an incident (missing data)')
        lines.append('- the remainder did not DESCRIBE an incident, which is '
                     'not the same as nothing having happened to them — never '
                     'report it as such')
    denom_time = evidence.get("time_denominator")
    if denom_time:
        t = evidence.get("time_filter", "")
        lines.append(f'- responses in scope before the {t}time filter: '
                     f'{denom_time["in_scope"]}; of those, '
                     f'{denom_time["mentioning"]} named any time of day at all')
        lines.append(f'- responses explicitly mentioning {t}time: '
                     f'{denom_time["matching"]} — the answer covers ONLY '
                     f'these, and must state this denominator')
        lines.append('- the remainder named no time of day, which says '
                     'nothing about when their experience happened — never '
                     'report it as the other time of day')
    denom_loc = evidence.get("location_denominator")
    if denom_loc:
        lines.append(f'- responses in scope (union of selected categories): '
                     f'{denom_loc["in_scope"]}')
        lines.append(f'- of those, responses naming any place: '
                     f'{denom_loc["naming_any"]} — only these are localizable; '
                     f'state this denominator in the answer')
    for lc in evidence.get("location_counts") or []:
        lines.append(f'- place "{lc["name"]}" ({lc["kind"]}): {lc["count"]} '
                     f'in-scope responses mention it')
    phrase = filter_phrase(evidence)
    for s in sorted(evidence["selection"], key=lambda s: -s["count"]):
        total = question_totals.get(s["question_id"])
        denom = (f" of {total} coded responses to {q_phrase(s['question_id'])}"
                 if total else "")
        if "count_unfiltered" in s:
            lines.append(f'- {s["name"]} ({s["label_id"]}): {s["count"]} responses '
                         f'{phrase} '
                         f'(of {s["count_unfiltered"]} total in this category)')
            continue
        lines.append(f'- {s["name"]} ({s["label_id"]}): {s["count"]} responses{denom}')
    for g in evidence["group_counts"]:
        lines.append(f'- group "{g["name"]}": {g["count_unique_responses"]} unique responses')
    for lc in lex_counts:
        per_q = ", ".join(f"q{q}: {n}" for q, n in sorted(lc["mentions_by_question"].items()))
        lines.append(f'- keyword concept "{lc["concept"]}": '
                     f'{lc["mentions_total"]} responses mention it ({per_q})')
    return "\n".join(lines) if lines else "(none)"


def render_quotes_block(evidence: dict, index: dict[str, dict],
                        group_of: dict[str, str] | None = None) -> str:
    """Quotes grouped under category headers, numbered globally. In
    group_by=location mode the headers are places instead."""
    if evidence.get("group_by") == "location" and evidence.get("location_counts"):
        lines = []
        by_loc: dict[str, list[dict]] = {}
        for q in evidence["quotes"]:
            by_loc.setdefault(q.get("location", ""), []).append(q)
        for lc in evidence["location_counts"]:
            qs = by_loc.get(lc["name"])
            if not qs:
                continue
            note = evidence["sampling_notes"].get(
                f'loc:{lc["name"]}', f'all {lc["count"]}')
            lines.append(f'{lc["name"]} ({note}):')
            for q in qs:
                lines.append(f'  {q["n"]}. {q["text"]}')
            lines.append("")
        return "\n".join(lines).rstrip() or "(none)"
    lines = []
    by_label: dict[str, list[dict]] = {}
    for q in evidence["quotes"]:
        by_label.setdefault(q["label_id"], []).append(q)
    for s in evidence["selection"]:
        lid = s["label_id"]
        qs = by_label.get(lid)
        if not qs:
            continue
        note = evidence["sampling_notes"].get(lid, f"all {s['count']}")
        header = f'{s["name"]} ({note}):'
        if group_of and lid in group_of:
            header = f'[group: {group_of[lid]}] ' + header
        lines.append(header)
        for q in qs:
            lines.append(f'  {q["n"]}. {q["text"]}')
        lines.append("")
    return "\n".join(lines).rstrip() or "(none)"


def build_synth_prompts(question: str, route: dict, evidence: dict,
                        index: dict[str, dict], lex_counts: list[dict],
                        question_totals: dict[str, int],
                        dataset_description: str = "",
                        question_texts: dict[str, str] | None = None,
                        ) -> tuple[str, str]:
    group_of = {lid: g["name"] for g in route["groups"] for lid in g["label_ids"]}
    guidance = ROUTE_GUIDANCE[route["route"]]
    if len({s["question_id"] for s in evidence["selection"]}) > 1:
        guidance += (
            " The evidence spans MORE THAN ONE survey question, each with its"
            " own denominator — when narrating a count, say which survey"
            " question it comes from (the counts block names each), and never"
            " present counts from different questions as one ranking.")
    if route.get("group_by") == "location":
        guidance += (
            " Organize the answer by PLACE, using the per-place counts from "
            "COMPUTED COUNTS. Lead with the localizable-responses denominator "
            "— only responses that named a place can be placed, and the "
            "answer must say so, copying those numbers exactly.")
    if route.get("location_filter"):
        guidance += (
            " The evidence is restricted to responses mentioning: "
            + ", ".join(route["location_filter"])
            + " — make that restriction explicit in the answer.")
    if route.get("actionability_filter") == "specific":
        guidance += (
            " The evidence is restricted to responses that propose a concrete,"
            " implementable action — report the actions respondents actually"
            " propose, in their own terms, and say plainly that the answer"
            " covers only this subset, copying its denominator from COMPUTED"
            " COUNTS. Never present it as what all respondents said.")
    elif route.get("actionability_filter") == "general":
        guidance += (
            " The evidence is restricted to responses raising a general"
            " concern rather than proposing a concrete action — say plainly"
            " that the answer covers only this subset, copying its"
            " denominator from COMPUTED COUNTS.")
    if route.get("event_filter"):
        guidance += (
            " The evidence is restricted to responses recounting something"
            " that actually happened to a specific person — report what"
            " respondents say happened, keeping their own framing, and lead"
            " with the denominator from COMPUTED COUNTS. These are"
            " self-reported first-hand accounts, not verified incidents or"
            " crime statistics: describe them as what respondents reported."
            " Never write or imply that the other respondents had nothing"
            " happen to them — they simply did not describe an incident.")
    if route.get("time_filter") in {"day", "night"}:
        t = route["time_filter"]
        guidance += (
            f" The evidence is restricted to responses explicitly mentioning"
            f" {t}time — say so, copying the denominator from COMPUTED"
            f" COUNTS. Never write or imply anything about when the OTHER"
            f" responses' experiences happened: naming no time of day is not"
            f" evidence either way.")
    system = SYNTH_SYSTEM.format(
        dataset_context=context_block(dataset_description),
        route_guidance=guidance,
    )
    user = SYNTH_USER.format(
        question=question.strip(),
        counts_block=render_counts_block(evidence, lex_counts, question_totals,
                                         question_texts),
        quotes_block=render_quotes_block(evidence, index, group_of or None),
    )
    return system, user


def parse_synth_output(raw: str) -> str:
    obj = extract_json(raw)
    if not isinstance(obj, dict) or not str(obj.get("answer_markdown", "")).strip():
        raise ValueError("synth output missing answer_markdown")
    return str(obj["answer_markdown"]).strip()


# One bracket may carry several citations — flash-lite sometimes writes
# [1, 3, 4] instead of [1][3][4]. Those used to parse as NOTHING: absent from
# Sources and not counted invalid (found by the 2026-08-03 audit).
CITATION_RE = re.compile(r"\[(\d{1,4}(?:\s*,\s*\d{1,4})*)\]")


def resolve_citations(answer_md: str, evidence: dict) -> tuple[str, list[dict], int]:
    """Map [n] citations back to response_keys. Returns (answer body, cited
    quotes, n_invalid). Citations of numbers not in the prompt are counted
    and flagged inline rather than silently kept. The body deliberately does
    NOT include a Sources section — the API sends sources as structured data
    the UI renders itself; render_sources_section produces the markdown
    version for answer.md and the CLI."""
    by_n = {q["n"]: q for q in evidence["quotes"]}
    cited, invalid = [], 0
    seen: set[int] = set()
    for m in CITATION_RE.finditer(answer_md):
        for part in m.group(1).split(","):
            n = int(part)
            if n in by_n:
                if n not in seen:
                    seen.add(n)
                    cited.append(by_n[n])
            else:
                invalid += 1
    answer = answer_md
    if invalid:
        answer += f"\n\n> NOTE: {invalid} citation(s) referenced numbers not in the evidence and could not be resolved."
    return answer, cited, invalid


def render_sources_section(cited: list[dict]) -> str:
    """The markdown Sources block, appended to the answer body wherever the
    answer lives as a single document (answer.md, CLI output)."""
    if not cited:
        return ""
    lines = ["", "---", "**Sources** (response_key — verbatim):", ""]
    for q in sorted(cited, key=lambda q: q["n"]):
        lines.append(f'- [{q["n"]}] `{q["response_key"]}`: "{q["text"]}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _complete_json(client: ModelClient, system: str, user: str) -> str:
    raw = client.complete(system, user)
    try:
        extract_json(raw)
        return raw
    except (ValueError, json.JSONDecodeError):
        return client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )


def run_route(client: ModelClient, question: str, summary_text: str,
              valid_ids: set[str], valid_concepts: set[str],
              dataset_description: str = "",
              valid_locations: set[str] = frozenset(),
              actionability_counts: dict[str, int] | None = None,
              event_counts: tuple[int, int] | None = None,
              time_counts: dict[str, int] | None = None,
              ) -> tuple[dict, dict]:
    system, user = build_route_prompts(question, summary_text,
                                       dataset_description, actionability_counts,
                                       event_counts, time_counts)
    act_available = bool(actionability_counts)
    evt_available = bool(event_counts and event_counts[0])
    time_available = bool(time_counts and
                          (time_counts.get("day") or time_counts.get("night")))
    raw = _complete_json(client, system, user)
    try:
        return parse_route_output(raw, valid_ids, valid_concepts,
                                  valid_locations, act_available, evt_available,
                                  time_available)
    except (ValueError, json.JSONDecodeError):
        raw = client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )
        return parse_route_output(raw, valid_ids, valid_concepts,
                                  valid_locations, act_available, evt_available,
                                  time_available)


def run_synth(client: ModelClient, question: str, route: dict, evidence: dict,
              index: dict[str, dict], lex_counts: list[dict],
              question_totals: dict[str, int],
              dataset_description: str = "",
              question_texts: dict[str, str] | None = None) -> str:
    system, user = build_synth_prompts(
        question, route, evidence, index, lex_counts, question_totals,
        dataset_description, question_texts)
    raw = _complete_json(client, system, user)
    try:
        return parse_synth_output(raw)
    except (ValueError, json.JSONDecodeError):
        raw = client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )
        return parse_synth_output(raw)
