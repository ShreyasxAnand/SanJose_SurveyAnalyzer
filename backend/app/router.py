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
  one-line rationale and a relevance rating. There is no cap — include all
  genuinely relevant categories, and nothing else. Copy label_id exactly.
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
{actionability_block}{event_block}- Never estimate counts, frequencies, or percentages.

Return ONLY valid JSON, exactly this shape:
{{"answerable": true, "route": "retrieval", "reason": "one line",
"candidates": [{{"label_id": "2_001", "relevance": "high", "rationale": "..."}}],
"groups": [], "lexicon_concepts": [], "group_by": "category",
"location_filter": [], "actionability_filter": "", "event_filter": ""}}
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
                        ) -> tuple[str, str]:
    """`event_counts` is (n_reporting_an_incident, n_coded); both optional
    blocks are omitted entirely when the artifacts don't carry the field, so
    the router is never offered a filter the data cannot honour."""
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
    system = ROUTE_SYSTEM.format(
        dataset_context=context_block(dataset_description),
        summary=summary_text.rstrip(),
        actionability_block=act_block,
        event_block=evt_block,
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


def parse_route_output(raw: str, valid_ids: set[str],
                       valid_concepts: set[str],
                       valid_locations: set[str] = frozenset(),
                       actionability_available: bool = False,
                       events_available: bool = False) -> tuple[dict, dict]:
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
) -> dict:
    """Counts + numbered quotes for the synth prompt. Sampling is seeded and
    disclosed; counts always cover the full (possibly filtered) membership.

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
    group_by = route.get("group_by", "category")

    allowed: set[str] | None = None

    def scope_under(allow: set[str] | None) -> set[str]:
        """The union of selected-category members surviving `allow`."""
        return {k for c in route["candidates"]
                for k in members.get(c["label_id"], [])
                if allow is None or k in allow}

    def restrict(keys: set[str]) -> set[str]:
        return keys if allowed is None else (allowed & keys)

    if loc_filter and location_members:
        hits: set[str] = set()
        for name in loc_filter:
            hits.update(location_members.get(name, []))
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
            keys = sorted(k for k in loc_sets[lc["name"]]
                          if k in texts and k not in quoted)
            if len(keys) > per_loc:
                keys = sorted(rng.sample(keys, per_loc))
                sampling_notes[f"loc:{lc['name']}"] = \
                    f"showing {per_loc} of {lc['count']}"
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
            keys = [k for k in all_keys if k in texts]
            if len(keys) > per_label:
                keys = sorted(rng.sample(keys, per_label))
                sampling_notes[lid] = f"showing {per_label} of {len(all_keys)}"
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
            "actionability_filter": act_filter,
            "actionability_denominator": actionability_denominator,
            "event_filter": evt_filter,
            "event_denominator": event_denominator}


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
    return " and ".join(parts)


def render_counts_block(evidence: dict, lex_counts: list[dict],
                        question_totals: dict[str, int]) -> str:
    lines = []
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
        denom = f" of {total} coded responses to question {s['question_id']}" if total else ""
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
                        dataset_description: str = "") -> tuple[str, str]:
    group_of = {lid: g["name"] for g in route["groups"] for lid in g["label_ids"]}
    guidance = ROUTE_GUIDANCE[route["route"]]
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
    system = SYNTH_SYSTEM.format(
        dataset_context=context_block(dataset_description),
        route_guidance=guidance,
    )
    user = SYNTH_USER.format(
        question=question.strip(),
        counts_block=render_counts_block(evidence, lex_counts, question_totals),
        quotes_block=render_quotes_block(evidence, index, group_of or None),
    )
    return system, user


def parse_synth_output(raw: str) -> str:
    obj = extract_json(raw)
    if not isinstance(obj, dict) or not str(obj.get("answer_markdown", "")).strip():
        raise ValueError("synth output missing answer_markdown")
    return str(obj["answer_markdown"]).strip()


CITATION_RE = re.compile(r"\[(\d{1,4})\]")


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
        n = int(m.group(1))
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
              ) -> tuple[dict, dict]:
    system, user = build_route_prompts(question, summary_text,
                                       dataset_description, actionability_counts,
                                       event_counts)
    act_available = bool(actionability_counts)
    evt_available = bool(event_counts and event_counts[0])
    raw = _complete_json(client, system, user)
    try:
        return parse_route_output(raw, valid_ids, valid_concepts,
                                  valid_locations, act_available, evt_available)
    except (ValueError, json.JSONDecodeError):
        raw = client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )
        return parse_route_output(raw, valid_ids, valid_concepts,
                                  valid_locations, act_available, evt_available)


def run_synth(client: ModelClient, question: str, route: dict, evidence: dict,
              index: dict[str, dict], lex_counts: list[dict],
              question_totals: dict[str, int],
              dataset_description: str = "") -> str:
    system, user = build_synth_prompts(
        question, route, evidence, index, lex_counts, question_totals,
        dataset_description)
    raw = _complete_json(client, system, user)
    try:
        return parse_synth_output(raw)
    except (ValueError, json.JSONDecodeError):
        raw = client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )
        return parse_synth_output(raw)
