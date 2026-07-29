"""Location layer: WHERE responses talk about, built from labeling's verbatim
location spans.

The labeling pass already extracts every place mention verbatim (with a
substring guard, so a span is always literally in the response). This module
turns those raw spans into something the query router can use:

  1. COLLECT the spans from the latest assignments, normalized and counted.
     Deterministic and free.
  2. CANONICALIZE with one model call — the only model call this module ever
     makes, mirroring lexicon.py. Surface forms of the same place merge
     ("downtown", "downtown area", "downtown san jose"); each concept is
     classified as a NAMED place ("santana row") or a place TYPE ("streets").
     The distinction matters at answer time: "on the streets at night" is a
     valid answer to a where-question without being a point on a map.
  3. MATCH deterministically forever after, by regex sweep over the full
     corpus — a superset of the labeled spans (every span is a literal
     substring of its response), so it also catches mentions the labeling
     pass didn't extract. Counts are computed, never estimated.

Coverage honesty: only responses that volunteered a place are localizable.
Every consumer of this layer must carry that denominator — "75 of 222
responses named a place" — rather than implying full coverage.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .llm import ModelClient

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCATIONS_DIR = REPO_ROOT / "data" / "locations"

SCHEMA_VERSION = 1
MIN_SPAN_COUNT = 2      # a span seen once ("the park by my house") is not a
                        # groupable place; singletons are counted and disclosed
# Hard cap on spans in the grouping prompt, lexicon-style (MAX_CANDIDATES).
# Span counts are Zipfian, so the top spans carry nearly all mentions — and
# the regex sweep recovers most mentions a dropped rare variant would have
# matched, via its concept's common surface forms. Without a cap, a 30k-row
# corpus puts thousands of spans in one prompt whose output must echo them
# all back, blowing the output-token ceiling.
MAX_SPANS = 300

VALID_KINDS = {"named", "type"}

GROUP_SYSTEM = """\
{dataset_context}You are canonicalizing place mentions extracted verbatim from survey responses.

You will receive spans respondents actually wrote, with how often each
occurred. The list mixes named places, generic kinds of place, spelling and
phrasing variants, and junk.

Group the spans into location CONCEPTS an analyst might ask "where" about.

Rules:
- Every span in a concept must be copied EXACTLY from the candidate list.
  Never invent a span — matching is literal, so an invented span matches
  nothing.
- Merge surface forms of the same place: abbreviations ("sj" with "san
  jose"), containment ("downtown" with "downtown area", "downtown san jose"),
  singular/plural and synonyms of the same kind of place ("street",
  "streets", "roads", "roadways").
- "kind" is "named" for a specific identifiable place (a city, neighborhood,
  park, road or landmark with a name) or "type" for a kind of place
  ("streets", "parks", "bus stops"). A concept must not mix the two.
- DROP spans that do not localize anything: vague references ("everywhere",
  "here", "my area", "outside"), and spans so generic they match half the
  corpus without saying where ("city", "area").
- Give each concept a short lowercase name an analyst would recognise —
  usually its most common span.
- Never estimate counts or frequencies.

Return ONLY valid JSON, exactly this shape:
{{"concepts": [{{"name": "downtown", "kind": "named",
"spans": ["downtown", "downtown area", "downtown san jose"]}},
{{"name": "streets", "kind": "type", "spans": ["street", "streets", "roads"]}}]}}
"""

GROUP_USER = """Place spans extracted from responses ({n} total):
{spans}
"""


# ---------------------------------------------------------------------------
# 1. Collection — no model, reads the latest assignments
# ---------------------------------------------------------------------------


def normalize_span(span: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges and a leading
    article, so "The Streets." and "streets" aggregate as one span."""
    s = re.sub(r"\s+", " ", str(span).strip().lower())
    s = s.strip(".,;:!?\"'()[]")
    for article in ("the ", "a ", "an "):
        if s.startswith(article) and len(s) > len(article) + 2:
            s = s[len(article):]
            break
    return s.strip()


def collect_spans(assignments_by_question: dict[str, list[dict]]
                  ) -> tuple[Counter, dict[str, dict]]:
    """Aggregate normalized location spans across questions. Returns
    (span_counts, per_question coverage) — the coverage numbers are the
    denominator every downstream answer must disclose."""
    span_counts: Counter[str] = Counter()
    coverage: dict[str, dict] = {}
    for q, assignments in assignments_by_question.items():
        with_loc = 0
        for a in assignments:
            spans = [normalize_span(s) for s in a.get("locations") or []]
            spans = [s for s in spans if s]
            if spans:
                with_loc += 1
                span_counts.update(set(spans))   # once per response, not per repeat
        coverage[q] = {"responses": len(assignments),
                       "responses_with_location": with_loc}
    return span_counts, coverage


def load_assignments_by_question(dataset_id: str) -> dict[str, list[dict]]:
    """Latest assignments per question, same "latest run" convention as the
    summary (so a *_review run supersedes the run it derived from)."""
    from .summary import LABELS_DIR, latest_run_dir

    ds_dir = LABELS_DIR / str(dataset_id)
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"No labels directory for dataset {dataset_id}")
    out: dict[str, list[dict]] = {}
    for qdir in sorted((d for d in ds_dir.iterdir() if d.is_dir()),
                       key=lambda d: (len(d.name), d.name)):
        run = latest_run_dir(qdir, "assignments.json")
        if run is not None:
            out[qdir.name] = json.loads(
                (run / "assignments.json").read_text(encoding="utf-8"))
    return out


def render_spans(
    span_counts: Counter, max_spans: int = MAX_SPANS
) -> tuple[str, list[str], int, list[str]]:
    """(prompt text, candidate spans, n_singletons_dropped,
    spans_over_cap_dropped). Sorted by count so the model sees the important
    spans first; everything past max_spans is dropped and disclosed."""
    kept = [(s, n) for s, n in span_counts.most_common() if n >= MIN_SPAN_COUNT]
    dropped = sum(1 for n in span_counts.values() if n < MIN_SPAN_COUNT)
    over_cap = [s for s, _ in kept[max_spans:]]
    kept = kept[:max_spans]
    text = "\n".join(f"{s} ({n})" for s, n in kept)
    return text, [s for s, _ in kept], dropped, over_cap


# ---------------------------------------------------------------------------
# 2. Canonicalization — the module's only model call
# ---------------------------------------------------------------------------


def build_locations(
    span_counts: Counter, client: ModelClient, dataset_description: str = "",
    max_spans: int = MAX_SPANS,
) -> tuple[dict, list[str]]:
    """Returns (locations, warnings). Spans the model invents are dropped —
    they would match nothing, and keeping them would overstate coverage."""
    from .induction import context_block, extract_json      # local: avoid cycle

    rendered, candidates, n_singletons, over_cap = render_spans(
        span_counts, max_spans)
    allowed = set(candidates)

    system = GROUP_SYSTEM.format(dataset_context=context_block(dataset_description))
    user = GROUP_USER.format(n=len(candidates), spans=rendered)
    raw = client.complete(system, user)
    try:
        obj = extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        raw = client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )
        obj = extract_json(raw)

    concepts, warnings, seen_spans, seen_names = [], [], set(), set()
    for c in obj.get("concepts") or []:
        if not isinstance(c, dict):
            continue
        name = normalize_span(c.get("name", ""))
        if not name or name in seen_names:
            continue
        kind = str(c.get("kind", "")).strip().lower()
        if kind not in VALID_KINDS:
            warnings.append(f"concept {name!r}: kind {kind!r} invalid, treated as 'type'")
            kind = "type"
        spans = []
        for s in c.get("spans") or []:
            s = normalize_span(s)
            if not s:
                continue
            if s not in allowed:
                warnings.append(f"concept {name!r}: span {s!r} not in the corpus, dropped")
            elif s in seen_spans:
                warnings.append(f"concept {name!r}: span {s!r} already used, dropped")
            else:
                seen_spans.add(s)
                spans.append(s)
        if spans:
            seen_names.add(name)
            concepts.append({"name": name, "kind": kind, "spans": sorted(spans)})
        else:
            warnings.append(f"concept {name!r}: no valid spans, dropped")

    locations = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Concepts are matched literally and case-insensitively against "
            "response text; counts from this layer are exact. Only responses "
            "that volunteered a place are localizable — always disclose the "
            "denominator. 'named' = a specific place; 'type' = a kind of place."
        ),
        "n_singleton_spans_dropped": n_singletons,
        "n_spans_over_cap_dropped": len(over_cap),
        "spans_over_cap_sample": over_cap[:10],
        "concepts": sorted(concepts, key=lambda c: (c["kind"], c["name"])),
    }
    return locations, warnings


# ---------------------------------------------------------------------------
# 3. Matching — deterministic, free, repeatable
# ---------------------------------------------------------------------------


def match_locations(locations: dict, keys: list[str],
                    texts: list[str]) -> dict[str, list[str]]:
    """concept name -> response_keys mentioning it, by regex sweep over the
    full corpus. A superset of the labeled spans (each span is a literal
    substring of its response), so it also catches mentions labeling missed."""
    from .lexicon import compile_concept

    out: dict[str, list[str]] = {}
    for concept in locations.get("concepts", []):
        pat = compile_concept(concept["spans"])
        out[concept["name"]] = [k for k, t in zip(keys, texts) if pat.search(t)]
    return out


def concept_kinds(locations: dict) -> dict[str, str]:
    return {c["name"]: c["kind"] for c in locations.get("concepts", [])}


def write_locations(locations: dict, manifest: dict, dataset_id: str) -> Path:
    out_dir = LOCATIONS_DIR / str(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "locations.json").write_text(
        json.dumps(locations, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
