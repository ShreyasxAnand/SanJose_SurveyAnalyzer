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
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import dates as dates_mod
from . import induction, labeling, llm, locations as locations_mod, router, subthemes, summary, verify
from .llm import ModelClient

ANSWERS_DIR = induction.DATA_DIR / "answers"

# Coverage guardrails. SMALL_BASE_N follows the ~200-comment threshold the
# Key Point Analysis literature uses for "a human could have just read
# these": below it the answer must say its base is small, prominently.
# COVERAGE_FLOOR is the share of in-scope coded responses the selected
# categories must cover before the answer stops having to name what it
# leaves out (the q15_04 case: 84 of 5,894 covered, disclosed only in a
# footnote nobody read).
SMALL_BASE_N = 200
COVERAGE_FLOOR = 0.60
MAX_UNCOVERED_SHOWN = 5

# A demographic-filtered answer resting on fewer matching responses than this
# gets a prominent thin-cell caveat (docs/DEMOGRAPHICS_PLAN.md §7.1) — the
# same failure SMALL_BASE_N guards, one order of magnitude down. Disclosed,
# never blocked: quotes still show, the answer is still produced.
DEMOGRAPHIC_NOTICE_N = 10

# Deterministic day/night classification of labeling's verbatim time_context
# spans. Deliberately coarse: a span matching neither set (e.g. "recently",
# "for years") stays unclassified, and one matching both ("day and night")
# counts as both. "every day"/"everyday" are frequency, not time of day.
TIME_NIGHT_RE = re.compile(
    r"night|evening|dark|midnight|overnight|dusk", re.IGNORECASE)
TIME_DAY_RE = re.compile(
    r"(?<!every )(?<!every)\bday(?:time|light)?\b|morning|afternoon|\bnoon"
    r"|daylight", re.IGNORECASE)


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
    # question_id -> the analyst's question wording, for provenance: a count
    # shown next to another question's count must say which question it
    # answers, or two different denominators read as one ranking.
    question_texts: dict[str, str] = field(default_factory=dict)
    # concept name -> question_ids whose own wording matches the concept
    # (e.g. "downtown" matches "changes to improve downtown"). Every response
    # to such a question is about the place by construction, so a location
    # filter must not drop it for not re-typing the place name.
    location_implicit_questions: dict[str, set[str]] = field(default_factory=dict)
    # Day/night classification of labeling's verbatim time_context spans.
    # `time_mentioned` = responses whose spans named any time at all; absent
    # spans mean the respondent named no time, never "it happened at noon".
    time_day: set[str] = field(default_factory=set)
    time_night: set[str] = field(default_factory=set)
    time_mentioned: set[str] = field(default_factory=set)
    # Sub-theme layer (app.subthemes): full-coverage sub-code membership
    # within large categories, so answers get real intra-category
    # denominators instead of inferring structure from sampled quotes.
    # Empty when no sub-themes run exists — everything downstream degrades
    # to exactly the pre-sub-theme behavior.
    sub_members: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    sub_names: dict[str, str] = field(default_factory=dict)
    sub_coded: dict[str, set[str]] = field(default_factory=dict)
    sub_runs: dict[str, str] = field(default_factory=dict)
    # label_id -> its sub-theme names, for the routing summary — the router
    # sees NAMES only (selection stays at category level)
    sub_names_by_label: dict[str, list[str]] = field(default_factory=dict)
    # Demographics (docs/DEMOGRAPHICS_PLAN.md), joined from the respondents
    # sidecar. This is an ANALYST-side filter like question_scope, not a
    # routing dimension: the router LLM never sees these fields — the UI
    # offers them from demographic_values, the request carries the selection,
    # and gather_evidence applies it deterministically.
    #   field -> value -> response_keys (mirrors location_members)
    demographic_members: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    # field -> response_keys whose respondent HAS a value — the `coded` set;
    # a respondent with no value is missing data, never a filterable bucket
    demographic_coded: dict[str, set[str]] = field(default_factory=dict)
    # field -> [(value, n_respondents)] sorted by count, for the UI dropdown.
    # Respondent counts, not response counts — a demographic belongs to the
    # person, who contributes one response per question.
    demographic_values: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    # field -> value -> respondent_keys, straight from the sidecar (no corpus
    # join): the faceted dropdown counts recount RESPONDENTS under the other
    # fields' selections, including respondents none of whose responses made
    # it into the coded corpus.
    demographic_respondents: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    # How long this context took to assemble, and whether the location sweep
    # was read from its cache or recomputed. Both land in the answer manifest:
    # the load is the bulk of an ask's wall clock, and a run that reports only
    # its model time understates what the analyst waited for.
    load_seconds: float = 0.0
    location_members_source: str = ""
    # "computed" or "cache" — which path produced this context. Disclosed in
    # the answer manifest for the same reason location_members_source is: a
    # surprising number should be traceable to a stale cache, not guessed at.
    context_source: str = ""


def discover_dataset_id(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    ds = sorted(d.name for d in summary.LABELS_DIR.iterdir() if d.is_dir()) \
        if summary.LABELS_DIR.is_dir() else []
    if len(ds) != 1:
        raise FileNotFoundError(f"Found {len(ds)} labeled datasets {ds}; specify one.")
    return ds[0]


def dataset_parquet(dataset_id: str, explicit: str | None = None) -> Path:
    """The corpus belonging to THIS dataset.

    `induction.discover_parquet` picks the most recent export across every
    dataset. That is right for the single-dataset CLI, and wrong here: it makes
    an older dataset's asks depend on which dataset was exported last. Uploading
    a 30-row file as dataset 3 repointed dataset 2's asks at the wrong corpus
    and turned them into a 500 — the bug this function exists to prevent.

    A missing export raises rather than falling back to another dataset's
    corpus: answering over the wrong responses is far worse than not answering.
    ask_api turns it into a 409, the same client-visible "pipeline hasn't run"
    state as missing labels.
    """
    if explicit:
        return induction.discover_parquet(explicit)
    ds_dir = induction.DATA_DIR / "exports" / str(dataset_id)
    for name in ("responses.parquet", "reshaped.parquet"):
        p = ds_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No export parquet for dataset {dataset_id} under {ds_dir}. "
        f"Re-export it (POST /datasets/{dataset_id}/export) or pass one "
        f"explicitly with --parquet.")


# --- in-process context cache ---------------------------------------------
#
# The stateless two-step flow loads the same context TWICE per analyst question
# (/ask/route then /ask/answer), and on dataset 2 that is ~1.3s a time — 0.96s
# of it re-reading the same 29k-row parquet. Nothing between the two calls can
# change it, so the second load is pure waste, as is every load for the next
# question about the same dataset.
#
# This is a memo, NOT the on-disk correctness cache `locations.members.json` is.
# That one keys on content fingerprints because a stale hit would persist wrong
# counts across processes forever; this one lives in one process, expires, and
# is validated on every hit against:
#   * which labels run is latest per question — a readdir, and every pipeline /
#     label / review / incremental run creates a NEW run dir, so a CLI run in
#     another process invalidates it;
#   * size + mtime of the export, locations and lexicon files — mtime alone is
#     not trusted anywhere in this codebase, which is why `_write_exports` also
#     drops the entry explicitly for the in-process writers (append, re-export,
#     metadata edit);
#   * a TTL, so anything both of those miss self-corrects in seconds rather
#     than lasting until a restart.
# Any mismatch is a miss, so the cost of being wrong is one second, not a wrong
# answer.
_CTX_CACHE: dict[str, tuple[str, float, AskContext]] = {}
_CTX_TTL_SECONDS = 120.0
_CTX_LOCK = threading.Lock()


def _context_key(dataset_id: str, description: str) -> str:
    """Cheap staleness key — stat and readdir only. Never reads the corpus,
    which is the second this cache exists to avoid."""
    parts = [hashlib.sha256(description.encode("utf-8")).hexdigest()[:8]]
    ds_dir = summary.LABELS_DIR / str(dataset_id)
    if ds_dir.is_dir():
        for qdir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            runs = sorted(d.name for d in qdir.iterdir()
                          if d.is_dir() and (d / "assignments.json").exists())
            parts.append(f"{qdir.name}={runs[-1] if runs else '-'}")
    sub_dir = subthemes.SUBTHEMES_DIR / str(dataset_id)
    if sub_dir.is_dir():
        for qdir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            runs = sorted(d.name for d in qdir.iterdir()
                          if d.is_dir() and (d / "sub_taxonomy.json").exists())
            parts.append(f"sub:{qdir.name}={runs[-1] if runs else '-'}")
    # Run-dir NAMES are not enough: taxonomies and sub-taxonomies invite
    # in-place hand edits (their review_instructions say so), and those
    # change answers without creating a run dir. Stat every artifact file so
    # an edit — any run, any question — changes the key. Stats only, no
    # reads; a few dozen files, well under a millisecond.
    for root, pattern in (
        (summary.TAXONOMY_DIR / str(dataset_id), "*/*/candidate_taxonomy.json"),
        (summary.LABELS_DIR / str(dataset_id), "*/*/assignments.json"),
        (subthemes.SUBTHEMES_DIR / str(dataset_id), "*/*/sub_taxonomy.json"),
        (subthemes.SUBTHEMES_DIR / str(dataset_id), "*/*/sub_assignments.json"),
    ):
        n, size, mtime = 0, 0, 0
        for f in root.glob(pattern) if root.is_dir() else ():
            try:
                st = f.stat()
            except OSError:
                continue
            n += 1
            size += st.st_size
            mtime = max(mtime, st.st_mtime_ns)
        parts.append(f"{pattern}:{n}:{size}:{mtime}")
    # both export forms: dataset_parquet() serves whichever exists, so both
    # must be watched — a reshaped-only dataset re-exported must invalidate
    for p in (induction.DATA_DIR / "exports" / str(dataset_id) / "responses.parquet",
              induction.DATA_DIR / "exports" / str(dataset_id) / "reshaped.parquet",
              # the demographics sidecar: re-selecting metadata columns
              # rewrites it without touching responses.parquet, and a stale
              # hit would answer a filtered ask from the previous selection
              induction.DATA_DIR / "exports" / str(dataset_id) / "respondents.parquet",
              # the manifest carries the date-period config (metadata_columns
              # value_type + date_ranges) that reshapes demographic facets —
              # an edited config must not serve yesterday's periods
              induction.DATA_DIR / "exports" / str(dataset_id) / "manifest.json",
              summary.LOCATIONS_DIR / str(dataset_id) / "locations.json",
              summary.LEXICON_DIR / str(dataset_id) / "lexicon.json"):
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{p.name}:-")
    return "|".join(parts)


def ask_logic_hash() -> str:
    """Fingerprint of this module's answer-shaping behavior, for the
    persistent ask cache: coverage guardrail thresholds and the day/night
    classification regexes all change what an answer says without touching
    any prompt text. Tuning one must invalidate stored answers."""
    blob = repr((SMALL_BASE_N, COVERAGE_FLOOR, MAX_UNCOVERED_SHOWN,
                 DEMOGRAPHIC_NOTICE_N,
                 TIME_NIGHT_RE.pattern, TIME_DAY_RE.pattern)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def context_cache_key(dataset_id: str | int, description: str = "") -> str:
    """The dataset's current data-state fingerprint, for the persistent ask
    cache (app.ask_cache). Same key the in-process context cache validates
    against — a cached answer can never be more stale than the context the
    live pipeline would answer from. Cheap: stat + readdir only."""
    return _context_key(str(dataset_id), description or "")


def invalidate_context_cache(dataset_id: str | int | None = None) -> None:
    """Drop cached contexts. Called by whatever rewrites a dataset's export in
    this process, so an append or a re-export is never answered around."""
    with _CTX_LOCK:
        if dataset_id is None:
            _CTX_CACHE.clear()
        else:
            _CTX_CACHE.pop(str(dataset_id), None)


def load_context(dataset_id: str, parquet: str | None = None,
                 description: str = "", use_cache: bool = True) -> AskContext:
    # An explicit parquet is the CLI steering at a specific file — never serve
    # that from a cache keyed on the dataset's own export.
    cacheable = use_cache and parquet is None
    if cacheable:
        with _CTX_LOCK:
            hit = _CTX_CACHE.get(str(dataset_id))
        if hit is not None:
            key, cached_at, ctx = hit
            if (time.time() - cached_at < _CTX_TTL_SECONDS
                    and key == _context_key(dataset_id, description)):
                # a shallow copy: the big dicts are shared (nothing mutates
                # them), but each run reports its own timing honestly
                return replace(ctx, load_seconds=0.0, context_source="cache")

    t0 = time.time()
    # the sink hands back the assignment rows build_summary already parsed, so
    # the actionability/event pass below reads them instead of re-resolving
    # "latest" and re-parsing every question's file
    assignments_by_q: dict[str, list[dict]] = {}
    s = summary.build_summary(dataset_id, description, assignments_by_q)
    summary.write_summary(s)     # keep the Phase 4 artifact on disk current

    lexicon: dict = {}
    lex_path = summary.LEXICON_DIR / dataset_id / "lexicon.json"
    if lex_path.exists():
        lexicon = json.loads(lex_path.read_text(encoding="utf-8"))

    parquet_path = dataset_parquet(dataset_id, parquet)
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
    time_day: set[str] = set()
    time_night: set[str] = set()
    time_mentioned: set[str] = set()
    for q in question_ids:
        assignments = assignments_by_q.get(q)
        if assignments is None:      # sink miss shouldn't happen; read rather than skip
            run = summary.latest_run_dir(summary.LABELS_DIR / dataset_id / q,
                                         "assignments.json")
            assignments = json.loads(
                (run / "assignments.json").read_text(encoding="utf-8"))
        members.update(router.members_by_label(assignments))
        for a in assignments:
            v = a.get("actionability")
            if v in labeling.VALID_ACTIONABILITY:
                actionability[a["response_key"]] = v
            if not a.get("not_returned") and not a.get("batch_failed"):
                event_coded.add(a["response_key"])
                if a.get("event_occurred"):
                    events.add(a["response_key"])
            spans = a.get("time_context") or []
            if spans:
                k = a["response_key"]
                time_mentioned.add(k)
                joined = " ".join(str(s) for s in spans)
                if TIME_NIGHT_RE.search(joined):
                    time_night.add(k)
                if TIME_DAY_RE.search(joined):
                    time_day.add(k)
    actionability_counts: dict[str, int] = {}
    for v in actionability.values():
        actionability_counts[v] = actionability_counts.get(v, 0) + 1

    question_texts = {q["question_id"]: q.get("question_text", "")
                      for q in s["questions"]}

    sub_members, sub_names, sub_coded, sub_runs = subthemes.load_sub_context(
        dataset_id, question_ids)
    sub_names_by_label = {
        lid: [sub_names[sid] for sid in sorted(subs) if sid in sub_names]
        for lid, subs in sub_members.items()}

    locations: dict = {}
    location_members: dict[str, list[str]] = {}
    location_implicit_questions: dict[str, set[str]] = {}
    members_source = ""
    loc_path = summary.LOCATIONS_DIR / dataset_id / "locations.json"
    if loc_path.exists():
        locations = json.loads(loc_path.read_text(encoding="utf-8"))
        all_keys = list(texts)
        # the sweep is deterministic in (concepts, corpus) and both are on
        # disk, so it is computed once per change rather than once per
        # question — the fingerprints inside decide, not a timestamp
        location_members, members_source = locations_mod.match_locations_cached(
            locations, all_keys, [texts[k] for k in all_keys], dataset_id)
        # A concept matching a QUESTION's own wording means every answer to
        # that question is about the place by construction ("changes to
        # improve downtown") — the location filter treats those responses as
        # implicit matches instead of starving the evidence down to the few
        # that re-typed the place name.
        from .lexicon import compile_concept
        for concept in locations.get("concepts", []):
            pat = compile_concept(concept["spans"])
            qs = {qid for qid, text in question_texts.items()
                  if text and pat.search(text)}
            if qs:
                location_implicit_questions[concept["name"]] = qs

    # Demographics: join the respondents sidecar to the corpus on
    # respondent_key. Both files are written by the same _write_exports call,
    # so a sidecar implies the corpus parquet carries respondent_key; the
    # column check below is the guard for a hand-supplied older parquet.
    demographic_members: dict[str, dict[str, set[str]]] = {}
    demographic_coded: dict[str, set[str]] = {}
    demographic_values: dict[str, list[tuple[str, int]]] = {}
    demographic_respondents: dict[str, dict[str, set[str]]] = {}
    sidecar = induction.DATA_DIR / "exports" / dataset_id / "respondents.parquet"
    if sidecar.exists():
        import pyarrow.parquet as pq
        corpus_cols = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        if {"response_key", "respondent_key"} <= corpus_cols:
            tbl = pq.read_table(parquet_path,
                                columns=["response_key", "respondent_key"])
            keys_of: dict[str, list[str]] = {}
            for rk, pk in zip(tbl.column("response_key").to_pylist(),
                              tbl.column("respondent_key").to_pylist()):
                if rk in texts:   # coded corpus only — empties/sentinels out
                    keys_of.setdefault(pk, []).append(rk)
            side = pq.read_table(sidecar,
                                 columns=["respondent_key", "field", "value"])
            n_respondents: dict[str, dict[str, int]] = {}
            for pk, f, v in zip(side.column("respondent_key").to_pylist(),
                                side.column("field").to_pylist(),
                                side.column("value").to_pylist()):
                # blank is missing data, never an "Unknown" bucket to filter for
                if v is None or not str(v).strip():
                    continue
                v = str(v).strip()
                n_respondents.setdefault(f, {})
                n_respondents[f][v] = n_respondents[f].get(v, 0) + 1
                demographic_respondents.setdefault(f, {}).setdefault(
                    v, set()).add(pk)
                rks = keys_of.get(pk)
                if rks:
                    demographic_members.setdefault(f, {}).setdefault(
                        v, set()).update(rks)
                    demographic_coded.setdefault(f, set()).update(rks)
            demographic_values = {
                f: sorted(vals.items(), key=lambda t: (-t[1], t[0]))
                for f, vals in n_respondents.items()}

            # Date-typed fields hold raw ISO dates — thousands of distinct
            # facet values nobody can filter on. Collapse them into period
            # labels ("2023 Q3", or the analyst's named ranges) derived from
            # the manifest's config at read time, so re-cutting periods is a
            # metadata edit, never a re-ingest. The manifest travels in the
            # same _write_exports as the sidecar, so the two never disagree.
            manifest_path = (induction.DATA_DIR / "exports" / dataset_id
                             / "manifest.json")
            date_fields: set[str] = set()
            date_config = None
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text("utf-8"))
                    date_fields = {
                        m["label"]
                        for m in manifest.get("metadata_columns", [])
                        if m.get("value_type") == "date"
                    }
                    date_config = manifest.get("date_ranges")
                except (OSError, json.JSONDecodeError, KeyError):
                    pass  # older/hand-damaged manifest: raw dates, no crash
            dates_mod.apply_period_labels(
                date_fields, date_config, demographic_values,
                demographic_members, demographic_coded,
                demographic_respondents)

    ctx = AskContext(
        dataset_id=dataset_id,
        summary=s,
        summary_text=summary.render_summary(s, sub_names_by_label),
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
        question_texts=question_texts,
        location_implicit_questions=location_implicit_questions,
        time_day=time_day,
        time_night=time_night,
        time_mentioned=time_mentioned,
        sub_members=sub_members,
        sub_names=sub_names,
        sub_coded=sub_coded,
        sub_runs=sub_runs,
        sub_names_by_label=sub_names_by_label,
        demographic_members=demographic_members,
        demographic_coded=demographic_coded,
        demographic_values=demographic_values,
        demographic_respondents=demographic_respondents,
        load_seconds=time.time() - t0,
        location_members_source=members_source,
        context_source="computed",
    )
    if cacheable:
        # keyed on the state as it is NOW — anything that changed while we were
        # loading shows up as a mismatch on the next hit, not as a stale serve
        with _CTX_LOCK:
            _CTX_CACHE[str(dataset_id)] = (
                _context_key(dataset_id, description), time.time(), ctx)
    return ctx


def validate_demographic_filter(demo: dict | None,
                                ctx: AskContext) -> dict[str, list[str]]:
    """The analyst's demographic filter, validated against the dataset's
    actual fields and values. Unknown field or value raises ValueError (the
    API's 422): the UI builds the filter from the server's own list, so a
    mismatch is a client bug or stale data — never something to guess at.
    Semantics downstream: values within one field are OR, fields are AND."""
    out: dict[str, list[str]] = {}
    known_fields = set(ctx.demographic_members) | set(ctx.demographic_values)
    for f, vals in (demo or {}).items():
        f = str(f).strip()
        vals = sorted({str(v).strip() for v in (vals or []) if str(v).strip()})
        if not f or not vals:
            continue
        if f not in known_fields:
            raise ValueError(
                f"Unknown demographic field {f!r}; this dataset has "
                f"{sorted(known_fields)}")
        # a value can legitimately exist only on respondents whose responses
        # never made the coded corpus — offered by the UI, matching zero
        # evidence; that is an honest empty result, not a client bug
        known = set(ctx.demographic_members.get(f, {})) | {
            v for v, _ in ctx.demographic_values.get(f, [])}
        unknown = [v for v in vals if v not in known]
        if unknown:
            raise ValueError(
                f"Unknown values for demographic field {f!r}: {unknown}")
        out[f] = vals
    return out


def facet_demographics(ctx: AskContext, demo: dict[str, list[str]],
                       ) -> tuple[list[tuple[str, list[tuple[str, int]]]],
                                  int | None]:
    """Dropdown counts under the CURRENT selection, faceted-search style:
    each field's value counts are recomputed against the OTHER fields'
    ticked values (a field never restricts its own list, so an OR selection
    can still be extended), plus the number of respondents matching the
    whole filter (None when no filter is active). Counts are respondents,
    from the sidecar; value order stays the unfiltered one so the dropdown
    doesn't reshuffle as the analyst ticks."""
    R = ctx.demographic_respondents

    def matching(exclude: str | None) -> set[str] | None:
        """Respondents matching every selected field except `exclude`;
        None = no restriction applies."""
        allowed: set[str] | None = None
        for f, vals in demo.items():
            if f == exclude:
                continue
            hits = set().union(*(R.get(f, {}).get(v, set()) for v in vals))
            allowed = hits if allowed is None else (allowed & hits)
        return allowed

    fields: list[tuple[str, list[tuple[str, int]]]] = []
    for f, vals in sorted(ctx.demographic_values.items()):
        allowed = matching(exclude=f)
        fields.append((f, [
            (v, n if allowed is None
             else len(R.get(f, {}).get(v, set()) & allowed))
            for v, n in vals]))
    overall = matching(exclude=None)
    return fields, (len(overall) if overall is not None else None)


def propose(client: ModelClient, question: str, ctx: AskContext,
            description: str = "",
            question_scope: list[str] | None = None,
            demographic_filter: dict[str, list[str]] | None = None,
            ) -> tuple[dict, dict]:
    """Step 1: the routing proposal. Returns (route, stats) exactly as
    router.run_route does — the caller renders it for review.

    `question_scope` restricts the ask to specific survey questions: the
    router only ever SEES the scoped questions' summary and only their label
    ids validate, so an out-of-scope category is structurally impossible in
    the proposal — the analyst's stated scope is enforced in code, not
    requested in prose. Unknown question ids raise ValueError (the API's
    422). An empty scope means all questions, as before.

    `demographic_filter` is the same kind of analyst-side restriction, one
    dimension over: {field: [values]} picked from the dataset's own
    demographics. It is validated and carried on the route here, but the
    router LLM never sees it — categories are proposed over the whole corpus
    and the filter is applied deterministically at evidence time."""
    demo = validate_demographic_filter(demographic_filter, ctx)
    scope = [str(q).strip() for q in (question_scope or []) if str(q).strip()]
    summary_text = ctx.summary_text
    valid_ids = ctx.valid_ids
    if scope:
        unknown = [q for q in scope if q not in ctx.question_ids]
        if unknown:
            raise ValueError(
                f"Unknown question ids in scope: {unknown}; "
                f"this dataset has {ctx.question_ids}")
        scoped = {**ctx.summary,
                  "questions": [q for q in ctx.summary["questions"]
                                if q["question_id"] in scope]}
        summary_text = summary.render_summary(scoped, ctx.sub_names_by_label)
        valid_ids = {e["label_id"] for q in scoped["questions"]
                     for e in q["entries"]}
    route, stats = router.run_route(
        client, question, summary_text,
        valid_ids, ctx.valid_concepts, description,
        valid_locations=ctx.valid_locations,
        actionability_counts=ctx.actionability_counts,
        event_counts=(len(ctx.events), len(ctx.event_coded)),
        time_counts={"day": len(ctx.time_day), "night": len(ctx.time_night),
                     "mentioned": len(ctx.time_mentioned)}
                    if ctx.time_mentioned else None,
        # informational only — the router is told the restriction is already
        # handled in code, so it routes the topic instead of refusing the
        # question for its demographic wording
        demographic_filter=demo or None)
    route["question_scope"] = scope
    route["demographic_filter"] = demo

    # Add-only completeness ratification: code finds unselected categories
    # sharing meaningful terms with the question; one cheap call rules on
    # exactly those. Fires only when the net catches something, and can only
    # ADD low-relevance lines the analyst can untick — a correct selection
    # cannot be damaged (observed miss it exists for: the violent-vs-QoL ask
    # that excluded the two biggest quality-of-life categories).
    if route.get("answerable") and route.get("route") != "aggregate_direct":
        questions_src = (ctx.summary["questions"] if not scope else
                         [q for q in ctx.summary["questions"]
                          if q["question_id"] in scope])
        entries = [e for q in questions_src for e in q["entries"]]
        selected = {c["label_id"] for c in route["candidates"]}
        missed = router.candidate_misses(question, entries, selected,
                                         route_name=route.get("route", ""))
        if missed:
            added = router.run_ratify(
                client, question, missed, description,
                selected=[e for e in entries if e["label_id"] in selected])
            for lid in added:
                route["candidates"].append({
                    "label_id": lid, "relevance": "low",
                    "rationale": "added by completeness check"})
            stats["completeness"] = {
                "scanned": len(missed), "added": added}
    return route, stats


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
    # "_"-prefixed keys are budget disclosures, not per-category notes. A note
    # now means "the quotes shown are not all of them" — from sampling, or from
    # a response already quoted under another place — so the wording below says
    # "disclosed for", not "sampled for".
    n_sampled = sum(1 for k in evidence["sampling_notes"] if not k.startswith("_"))
    budget_notes = [v for k, v in evidence["sampling_notes"].items()
                    if k.startswith("_quote_budget")]
    parts = [
        f"Routed as {ROUTE_LABEL.get(route['route'], route['route'])}"
        + (f" — {route['reason'].rstrip('.')}" if route["reason"] else "") + ".",
        f"Searched {len(sel)} of {len(ctx.valid_ids)} categories ({shown}), "
        f"covering {unique} unique responses"
        + (f" — {round(100 * (evidence.get('scope_coverage') or {}).get('ratio', 0) or 0)}% "
           f"of the {(evidence.get('scope_coverage') or {}).get('scope_total')} "
           f"coded responses in scope"
           if (evidence.get("scope_coverage") or {}).get("ratio") is not None
           else "") + ".",
        f"Showed {len(evidence['quotes'])} verbatims to the answering model"
        + (f" (quote coverage disclosed for {n_sampled} categor"
           + ("y" if n_sampled == 1 else "ies")
           + "; counts always cover the full data)" if n_sampled else "")
        + f"; {n_cited} of them are cited in the answer.",
    ]
    if evidence.get("small_base"):
        parts.append(f"CAUTION: small base — only {unique} responses underlie "
                     f"this answer.")
    if evidence.get("uncovered_categories"):
        top_un = ", ".join(u["name"] for u in evidence["uncovered_categories"][:3])
        parts.append(f"Largest in-scope categories NOT searched: {top_un}.")
    for note in budget_notes:
        parts.append(f"Quote budget: {note}.")
    if evidence.get("location_filter"):
        names = ", ".join(evidence["location_filter"])
        note = (f"Evidence restricted to responses mentioning {names} "
                "(deterministic keyword match")
        implicit = evidence.get("location_filter_implicit_questions") or []
        if implicit:
            worded = ", ".join(
                f'"{ctx.question_texts.get(q, q)}"' for q in implicit)
            note += (f"; every response to {worded} counts as mentioning it — "
                     "the question itself asks about that place")
        parts.append(note + ").")
    scope = route.get("question_scope") or []
    if scope:
        worded = ", ".join(f'"{ctx.question_texts.get(q, q)}"' for q in scope)
        parts.append(f"Scope restricted by the analyst to survey question"
                     f"{'s' if len(scope) > 1 else ''} {worded}.")
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
    denom_time = evidence.get("time_denominator")
    if denom_time:
        t = evidence.get("time_filter", "")
        parts.append(
            f"Evidence restricted to responses explicitly mentioning "
            f"{t}time: {denom_time['matching']} of {denom_time['in_scope']} "
            f"in-scope responses ({denom_time['mentioning']} named any time "
            f"of day at all). The rest named no time, which says nothing "
            f"about when their experience happened.")
    denom_demo = evidence.get("demographic_denominator")
    if denom_demo:
        worded = "; ".join(
            f"{f} = {' or '.join(vals)}"
            for f, vals in (evidence.get("demographic_filter") or {}).items())
        no_value = denom_demo["in_scope"] - denom_demo["coded"]
        parts.append(
            f"Evidence restricted by the analyst to respondents with "
            f"{worded}: {denom_demo['matching']} of {denom_demo['in_scope']} "
            f"in-scope responses"
            + (f" ({no_value} {'has' if no_value == 1 else 'have'} no "
               f"recorded value for the filtered field"
               f"{'s' if len(evidence.get('demographic_filter') or {}) > 1 else ''}"
               f" — missing data, not a group)" if no_value else "")
            + ".")
        if evidence.get("demographic_thin"):
            n = denom_demo["matching"]
            parts.append(
                f"CAUTION: only {n} response{'' if n == 1 else 's'} "
                f"match{'es' if n == 1 else ''} this demographic filter — "
                f"read the answer as those few voices, not as the group.")
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
    if route.get("route") == "aggregate_direct":
        # tally routes never synthesize — every figure is a computed count
        # and the narration is a template, so the whole answer costs zero
        # model calls past the route step
        return answer_aggregate(client, question, route, ctx,
                                description=description,
                                route_stats=route_stats)
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
        event_coded_keys=ctx.event_coded or None,
        location_implicit_questions=ctx.location_implicit_questions or None,
        time_day_keys=ctx.time_day or None,
        time_night_keys=ctx.time_night or None,
        time_mentioned_keys=ctx.time_mentioned or None,
        sub_members=ctx.sub_members or None,
        sub_names=ctx.sub_names or None,
        sub_coded=ctx.sub_coded or None,
        demographic_members=ctx.demographic_members or None,
        demographic_coded=ctx.demographic_coded or None)
    # thin-cell notice (docs/DEMOGRAPHICS_PLAN.md §7.1): a demographic filter
    # narrow enough to rest on a handful of responses is disclosed, never
    # blocked — the constant is folded into ask_logic_hash
    dd = evidence.get("demographic_denominator")
    evidence["demographic_thin"] = bool(
        dd and dd["matching"] < DEMOGRAPHIC_NOTICE_N)
    lex_counts = router.lexicon_counts(
        ctx.lexicon, route["lexicon_concepts"], ctx.keys_by_question, ctx.texts
    ) if route["lexicon_concepts"] else []

    # Coverage guardrails — computed BEFORE synthesis so the counts block can
    # force the answer to disclose what it does and does not cover.
    scope_qids = sorted({s["question_id"] for s in evidence["selection"]})
    scope_total = sum(ctx.question_totals.get(q, 0) for q in scope_qids)
    covered = evidence["n_unique_responses"]
    evidence["scope_coverage"] = {
        "covered": covered, "scope_total": scope_total, "questions": scope_qids,
        "ratio": round(covered / scope_total, 3) if scope_total else None}
    evidence["small_base"] = 0 < covered < SMALL_BASE_N
    if scope_total and covered < COVERAGE_FLOOR * scope_total:
        selected_lids = {s["label_id"] for s in evidence["selection"]}
        uncovered = [
            {"label_id": lid, "name": e["name"],
             "count": len(ctx.members.get(lid, []))}
            for lid, e in ctx.index.items()
            if e["question_id"] in scope_qids and lid not in selected_lids
            and ctx.members.get(lid)]
        uncovered.sort(key=lambda u: (-u["count"], u["label_id"]))
        evidence["uncovered_categories"] = uncovered[:MAX_UNCOVERED_SHOWN]

    raw_answer = router.run_synth(synth, question, route, evidence, ctx.index,
                                  lex_counts, ctx.question_totals, description,
                                  ctx.question_texts or None)
    answer_body, cited, n_invalid = router.resolve_citations(raw_answer, evidence)

    # Verification: deterministic guards on every answer, always — and nothing
    # is ever rewritten. What the guards flag is DISCLOSED, so the text an
    # analyst reads is exactly what the synth model produced from the
    # evidence, and the manifest's provenance means what it says.
    #
    # A model repair call used to sit here. Removed 2026-08-25: across the six
    # stored answers that reached it, it cleared the flags exactly once, and
    # its own instructions licensed it to reword a quotation ("quote what it
    # actually says") — the one edit this pipeline must never make, since a
    # model that has just fabricated a quote cannot be trusted to author its
    # replacement. A wrong answer now ships visibly wrong instead of quietly
    # rewritten.
    # The SAME plan text the answer was written against feeds both guards: its
    # numbers are computed in code, so they are legitimate to state (the
    # union-counted "Everything else" line exists nowhere in `evidence`), and
    # its lines are the structure the answer must have.
    plan_str = router.render_section_plan(evidence)
    violations = verify.find_violations(answer_body, evidence, lex_counts,
                                        ctx.question_totals, plan_str)
    # The structure guard used to run ONLY on a repaired answer, so a draft
    # that stated the wrong counts but quoted cleanly was never checked at
    # all — which is how the 2026-08-25 trash answer reached an analyst with
    # five of its ten sections understating their category by leading with a
    # sub-theme's count. Counts are what an analyst quotes onward; they get
    # the same unconditional check as everything else.
    violations += verify.plan_structure_violations(answer_body, plan_str)
    verification = {"checked": True, "violations": violations}

    # the single-document form (body + Sources) goes to answer.md and the
    # CLI; the API returns the body and the sources as separate fields
    answer_md = answer_body + router.render_sources_section(cited)
    note = process_note(route, evidence, ctx, len(cited), lex_counts)
    if violations:
        note += (f" CAUTION: {len(violations)} statement"
                 f"{'' if len(violations) == 1 else 's'} could "
                 f"not be verified against the computed data — see the "
                 f"verification record.")
    stats = {
        "categories_searched": len(evidence["selection"]),
        "categories_total": len(ctx.valid_ids),
        "unique_responses": evidence["n_unique_responses"],
        "quotes_shown": len(evidence["quotes"]),
        "quotes_cited": len(cited),
        "scope_total": scope_total,
        "scope_coverage": evidence["scope_coverage"]["ratio"],
        "small_base": evidence["small_base"],
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
        "verification": verification,
    }


def answer_aggregate(client: ModelClient, question: str, route: dict,
                     ctx: AskContext, description: str = "",
                     route_stats: dict | None = None) -> dict:
    """Deterministic answer for route "aggregate_direct": the tally the
    question asks for, computed from coded data and narrated by template.
    No synthesis call — nothing here can hallucinate, and identical inputs
    produce byte-identical answers. Candidates, when present, narrow the
    tally to their members; otherwise it covers every coded response in
    the question scope."""
    t0 = time.time()
    target = route.get("aggregate_target", "")
    scope_qids = [q for q in (route.get("question_scope") or []) or ctx.question_ids]
    scope: set[str] = set()
    for q in scope_qids:
        scope.update(ctx.keys_by_question.get(q, []))
    n_question_scope = len(scope)
    narrowed = bool(route.get("candidates"))
    if narrowed:
        selected: set[str] = set()
        for c in route["candidates"]:
            selected.update(ctx.members.get(c["label_id"], []))
        scope &= selected
    # The analyst's demographic filter narrows the tally too — a filtered ask
    # answered over everyone would be silently wrong, the worst failure this
    # route has. Same OR-within-field / AND-across-fields semantics as
    # gather_evidence, same {in_scope, coded, matching} disclosure.
    demo_filter = route.get("demographic_filter") or {}
    demographic_denominator = None
    if demo_filter and ctx.demographic_members:
        pre = set(scope)
        hits = set.intersection(*[
            set().union(*(ctx.demographic_members.get(f, {}).get(v, set())
                          for v in vals))
            for f, vals in demo_filter.items()])
        coded_all = set.intersection(*[
            ctx.demographic_coded.get(f, set()) for f in demo_filter])
        demographic_denominator = {
            "in_scope": len(pre),
            "coded": len(pre & coded_all),
            "matching": len(pre & hits),
        }
        scope &= hits
    q_names = ", ".join(f'"{ctx.question_texts.get(q, q)}"' for q in scope_qids)
    # When categories narrow the tally, "coded responses to <question>" would
    # claim the wrong denominator (a reviewer caught 1,350 category-narrowed
    # responses being presented as the question's total of 3,322) — the
    # attribution must say which base it is.
    if narrowed:
        q_names = (f"{q_names} within the {len(route['candidates'])} selected "
                   f"categor{'y' if len(route['candidates']) == 1 else 'ies'} "
                   f"(the question has {n_question_scope} coded responses in "
                   f"total)")
    if demographic_denominator:
        worded = "; ".join(f"{f} = {' or '.join(vals)}"
                           for f, vals in demo_filter.items())
        q_names = (f"{q_names}, restricted to respondents with {worded} "
                   f"({demographic_denominator['matching']} of "
                   f"{demographic_denominator['in_scope']} in-scope responses)")
    n_scope = len(scope)

    lines: list[str] = []
    aggregate: dict = {"target": target, "in_scope": n_scope,
                       "questions": scope_qids}
    if demographic_denominator:
        aggregate["demographic_filter"] = demo_filter
        aggregate["demographic_denominator"] = demographic_denominator
    if target == "location":
        rows = []
        naming: set[str] = set()
        for name, keys in ctx.location_members.items():
            hit = scope & set(keys)
            if hit:
                naming |= hit
                rows.append({"name": name,
                             "kind": ctx.location_kinds.get(name, "type"),
                             "count": len(hit)})
        rows.sort(key=lambda r: (-r["count"], r["name"]))
        aggregate.update({"naming_any": len(naming), "counts": rows[:50]})
        lines += [
            f"Of the **{n_scope}** coded responses to {q_names}, "
            f"**{len(naming)}** name at least one place. The most-mentioned "
            f"places (a response can name several):", ""]
        lines += [f"| Place | Kind | Responses |", "|---|---|---|"]
        lines += [f"| {r['name']} | {r['kind']} | {r['count']} |"
                  for r in rows[:20]]
        if len(rows) > 20:
            lines.append(f"\n…and {len(rows) - 20} more places with smaller "
                         f"counts.")
        lines.append(f"\nThe remaining {n_scope - len(naming)} responses name "
                     f"no place; that says nothing about where their concern "
                     f"applies.")
    elif target == "time":
        mentioned = scope & ctx.time_mentioned
        day, night = scope & ctx.time_day, scope & ctx.time_night
        both = day & night
        classified = day | night
        # `mentioned` includes ANY time phrase the coding captured —
        # frequencies ("every weekend") and periods ("since covid") as well
        # as times of day. The rendered arithmetic must close: mentioned =
        # classified day/night + other-time-phrases (a reviewer caught the
        # earlier wording implying 439 all named a time of day when only
        # 192 did).
        aggregate.update({"mentioning_any": len(mentioned), "day": len(day),
                          "night": len(night), "both": len(both),
                          "day_or_night": len(classified)})
        lines += [
            f"Of the **{n_scope}** coded responses to {q_names}, "
            f"**{len(mentioned)}** include some time reference, and "
            f"**{len(classified)}** of those name a time of day:", "",
            f"- **{len(night)}** mention nighttime (“at night”, "
            f"“after dark”…)",
            f"- **{len(day)}** mention daytime",
            f"- **{len(both)}** mention both", "",
            f"The other {len(mentioned) - len(classified)} time references "
            f"are frequencies or periods (“every weekend”, “for years”), "
            f"not times of day. The remaining "
            f"{n_scope - len(mentioned)} responses name no time at all — "
            f"which is not evidence about when their experience happened, "
            f"so no day/night split can honestly be claimed for them."]
    elif target == "event":
        coded = scope & ctx.event_coded
        events = scope & ctx.events
        aggregate.update({"coded": len(coded), "reported_incident": len(events)})
        uncoded = n_scope - len(coded)
        lines += [
            f"Of the **{len(coded)}** coded responses to {q_names}, "
            f"**{len(events)}** recount a specific incident that actually "
            f"happened to a particular person (“my car window got "
            f"smashed”) — the rest, **{len(coded) - len(events)}**, "
            f"raise a concern, opinion, or ongoing condition without "
            f"describing an incident.", "",
            f"Not describing an incident is NOT evidence that nothing "
            f"happened to that respondent; survey answers are short and most "
            f"people do not recount events unprompted. These are self-reported "
            f"accounts, not verified incidents."]
        if uncoded:
            lines.append(f"\n{uncoded} in-scope responses were never checked "
                         f"for an incident (missing data).")
    else:
        raise ValueError(f"unknown aggregate_target {target!r}")

    answer_md = "\n".join(lines)
    note = (f"Answered deterministically from the coded {target} data — every "
            f"figure is a computed count over {n_scope} in-scope responses; "
            f"no synthesis model was involved. Scope: {q_names}.")
    scope_total = sum(ctx.question_totals.get(q, 0) for q in scope_qids)
    stats = {
        "categories_searched": len(route.get("candidates") or []),
        "categories_total": len(ctx.valid_ids),
        "unique_responses": n_scope,
        "quotes_shown": 0,
        "quotes_cited": 0,
        "scope_total": scope_total,
        "scope_coverage": round(n_scope / scope_total, 3) if scope_total else None,
        "small_base": 0 < n_scope < SMALL_BASE_N,
    }
    evidence = {
        "selection": [], "quotes": [], "sampling_notes": {},
        "group_counts": [], "n_unique_responses": n_scope,
        "group_by": "category", "location_filter": [],
        "scope_coverage": {"covered": n_scope, "scope_total": scope_total,
                           "questions": scope_qids, "ratio": stats["scope_coverage"]},
        "small_base": stats["small_base"],
        "aggregate": aggregate,
        "demographic_filter": demo_filter,
        "demographic_denominator": demographic_denominator,
        "demographic_thin": bool(
            demographic_denominator
            and demographic_denominator["matching"] < DEMOGRAPHIC_NOTICE_N),
    }
    run_id = make_run_id(question)
    out_dir = write_artifacts(
        run_id=run_id, question=question, ctx=ctx, route=route,
        route_stats=route_stats or {}, evidence=evidence, lex_counts=[],
        answer_md=answer_md, n_invalid_citations=0, selection=None,
        client=client, description=description, elapsed=time.time() - t0,
        process_note=note)
    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "answer_markdown": answer_md,
        "answer_body": answer_md,
        "stats": stats,
        "process_note": note,
        "evidence": evidence,
        "lexicon_counts": [],
        "cited": [],
        "invalid_citations": 0,
        "selection": None,
        "aggregate": aggregate,
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
            "time_filter": evidence.get("time_filter") or "",
            "time_denominator": evidence.get("time_denominator"),
            "demographic_filter": evidence.get("demographic_filter") or {},
            "demographic_denominator": evidence.get("demographic_denominator"),
            "location_filter_implicit_questions":
                evidence.get("location_filter_implicit_questions") or [],
        },
        "invalid_citations": n_invalid_citations,
        "usage": usage_block([client] + ([synth_client] if synth_client else []),
                             elapsed),
        # usage.elapsed_seconds covers the model phase only (it is timed inside
        # answer(), which receives an already-loaded context). Loading that
        # context reads every labels run and, on a cold cache, sweeps the whole
        # corpus for locations — historically the larger half of an ask. Report
        # both so neither number can be mistaken for the whole wait.
        "timing": {
            "context_load_seconds": round(ctx.load_seconds, 2),
            "answer_seconds": round(elapsed, 2),
            "total_seconds": round(ctx.load_seconds + elapsed, 2),
            "location_members_source": ctx.location_members_source,
            "context_source": ctx.context_source,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def route_from_selection(selected: list[dict], route_name: str, reason: str,
                         lexicon_concepts: list[str], ctx: AskContext,
                         group_by: str = "category",
                         location_filter: list[str] | None = None,
                         actionability_filter: str = "",
                         event_filter: str = "",
                         time_filter: str = "",
                         question_scope: list[str] | None = None,
                         aggregate_target: str = "",
                         demographic_filter: dict[str, list[str]] | None = None,
                         ) -> dict:
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
    route_name = route_name.strip().lower()
    if route_name not in router.VALID_ROUTES:
        route_name = "retrieval"
    aggregate_target = (aggregate_target or "").strip().lower()
    if route_name == "aggregate_direct":
        available = {"location": bool(ctx.valid_locations),
                     "time": bool(ctx.time_mentioned),
                     "event": bool(ctx.event_coded)}
        if not available.get(aggregate_target):
            raise ValueError(
                f"aggregate_direct needs an available aggregate_target; got "
                f"{aggregate_target!r} (available: "
                f"{sorted(k for k, v in available.items() if v)})")
    else:
        aggregate_target = ""
        if not candidates:
            raise ValueError("No categories selected")
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
    t = router.normalize_time(time_filter)
    if t == "invalid":
        raise ValueError(
            f"Unknown time_filter {time_filter!r}; expected \"day\", "
            f"\"night\", or empty (there is no filter for responses naming "
            f"no time of day)")
    if t and not (ctx.time_day if t == "day" else ctx.time_night):
        t = ""
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
        "time_filter": t,
        # recorded for the manifest; the proposal step is where scope is
        # ENFORCED — step 2 stays permissive so an analyst's deliberate
        # out-of-scope addition is not rejected
        "question_scope": [str(q).strip() for q in (question_scope or [])
                           if str(q).strip()],
        "aggregate_target": aggregate_target,
        # analyst-side restriction, validated the same way an unknown label
        # id is: the UI offers only real values, so a mismatch is a 422
        "demographic_filter": validate_demographic_filter(
            demographic_filter, ctx),
    }
