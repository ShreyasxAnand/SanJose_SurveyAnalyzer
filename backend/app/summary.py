"""Phase 4: the taxonomy summary artifact — what the query router reads.

A compact flat list per question — `Parent > Child — description (n=340)` —
plus the dataset description. Small enough to fit in one prompt, and every
count in it is computed by counting assignment rows, never estimated.

Building it is deterministic and free (no model calls), so it is simply
regenerated from the latest pipeline artifacts every time it is needed;
`scripts.summarize` writes it to disk and `scripts.ask` rebuilds it on every
question so it can never be stale relative to the labels.

"Latest" follows the repo-wide convention: the lexicographically last run
directory wins, which makes a `*_review` run supersede the run it derived
from. The taxonomy paired with a labels run is resolved in this order:
  1. a taxonomy run with the SAME run_id (review runs write both sides),
  2. the `taxonomy_path` recorded in the labels manifest (scripts.label runs),
  3. the latest taxonomy run for the question (last resort, disclosed).
"""
from __future__ import annotations

import json
from pathlib import Path

from .induction import DATA_DIR, TAXONOMY_DIR, utc_now

SCHEMA_VERSION = 1
SUMMARY_DIR = DATA_DIR / "summary"
LABELS_DIR = DATA_DIR / "labels"
LEXICON_DIR = DATA_DIR / "lexicon"
LOCATIONS_DIR = DATA_DIR / "locations"

MAX_DESC_CHARS = 200   # per-label description cap inside the prompt rendering


def latest_run_dir(question_dir: Path, required_file: str) -> Path | None:
    """Lexicographically last run dir that actually contains required_file.
    Run ids sort correctly because they start with a UTC timestamp, and a
    `*_review` suffix sorts after the plain hash runs from the same second."""
    if not question_dir.is_dir():
        return None
    hits = sorted(
        d for d in question_dir.iterdir()
        if d.is_dir() and (d / required_file).exists()
    )
    return hits[-1] if hits else None


def resolve_taxonomy_path(labels_run: Path, dataset_id: str, question_id: str) -> tuple[Path, str]:
    """Find the taxonomy the latest labels actually correspond to.
    Returns (path, how) — `how` goes into the artifact so a fallback is
    visible instead of silent."""
    paired = TAXONOMY_DIR / dataset_id / question_id / labels_run.name / "candidate_taxonomy.json"
    if paired.exists():
        return paired, "same_run_id"
    manifest_path = labels_run / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("taxonomy_path",):
            p = manifest.get(key)
            if p and Path(p).exists():
                return Path(p), "labels_manifest"
        run = manifest.get("taxonomy_run")
        if run:
            p = TAXONOMY_DIR / dataset_id / question_id / str(run) / "candidate_taxonomy.json"
            if p.exists():
                return p, "labels_manifest"
    latest = latest_run_dir(TAXONOMY_DIR / dataset_id / question_id, "candidate_taxonomy.json")
    if latest is None:
        raise FileNotFoundError(
            f"No taxonomy found for dataset {dataset_id} question {question_id}"
        )
    return latest / "candidate_taxonomy.json", "latest_taxonomy_fallback"


def summarize_question(dataset_id: str, question_id: str,
                       assignments_sink: dict[str, list[dict]] | None = None) -> dict:
    """One question's block: taxonomy structure + real counts from the latest
    assignments. Assignment ids not present in the paired taxonomy are counted
    and disclosed rather than silently dropped — a mismatch here means the
    pairing rule failed and the analyst should know.

    `assignments_sink`, if given, receives `{question_id: assignments}` for the
    run this summary actually used. Callers that need the assignment rows
    themselves (the ask path wants actionability and event flags) can then use
    the same parse instead of resolving "latest" and re-reading the file — one
    less chance of the two disagreeing, and one less parse of a 30k-row JSON."""
    labels_run = latest_run_dir(LABELS_DIR / dataset_id / question_id, "assignments.json")
    if labels_run is None:
        raise FileNotFoundError(
            f"No labels run for dataset {dataset_id} question {question_id}. "
            "Run scripts.label first."
        )
    tax_path, tax_how = resolve_taxonomy_path(labels_run, dataset_id, question_id)
    taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))
    assignments = json.loads((labels_run / "assignments.json").read_text(encoding="utf-8"))
    if assignments_sink is not None:
        assignments_sink[question_id] = assignments

    counts: dict[str, int] = {lab["label_id"]: 0 for lab in taxonomy["labels"]}
    unknown_ids: dict[str, int] = {}
    for a in assignments:
        for lid in a.get("label_ids") or []:
            if lid in counts:
                counts[lid] += 1
            else:
                unknown_ids[lid] = unknown_ids.get(lid, 0) + 1

    parents_by_id = {p["parent_id"]: p for p in taxonomy.get("parents", [])}
    entries = []
    for lab in taxonomy["labels"]:
        parent = parents_by_id.get(lab.get("parent_id"))
        entries.append({
            "label_id": lab["label_id"],
            "name": lab["name"],
            "parent_id": lab.get("parent_id"),
            "parent_name": parent["name"] if parent else None,
            "description": (lab.get("description") or "").replace("\n", " ").strip(),
            "count": counts[lab["label_id"]],
        })
    entries.sort(key=lambda e: (-e["count"], e["name"].lower()))

    n = len(assignments)
    n_unc = sum(1 for a in assignments if a.get("uncategorized"))
    return {
        "question_id": question_id,
        "question_text": taxonomy.get("question_text", ""),
        "labels_run": labels_run.name,
        "taxonomy_run": tax_path.parent.name,
        "taxonomy_resolved_via": tax_how,
        "n_responses": n,
        "n_uncategorized": n_unc,
        "unknown_assignment_ids": unknown_ids,
        "parents": [
            {"parent_id": p["parent_id"], "name": p["name"],
             "description": p.get("description", "")}
            for p in taxonomy.get("parents", [])
        ],
        "entries": entries,
    }


def build_summary(dataset_id: str, dataset_description: str = "",
                  assignments_sink: dict[str, list[dict]] | None = None) -> dict:
    """The whole artifact: every question that has labels, plus the lexicon
    concept list (names only — the router selects concepts, matching stays
    deterministic in code).

    `assignments_sink` is passed straight through to `summarize_question` —
    see there for why."""
    ds_dir = LABELS_DIR / dataset_id
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"No labels directory for dataset {dataset_id}")
    question_ids = sorted((d.name for d in ds_dir.iterdir() if d.is_dir()),
                          key=lambda q: (len(q), q))
    questions = [summarize_question(dataset_id, q, assignments_sink)
                 for q in question_ids]

    lexicon_concepts = []
    lex_path = LEXICON_DIR / dataset_id / "lexicon.json"
    if lex_path.exists():
        lexicon = json.loads(lex_path.read_text(encoding="utf-8"))
        lexicon_concepts = [
            {"name": c["name"], "n_terms": len(c.get("terms", []))}
            for c in lexicon.get("concepts", [])
        ]

    location_concepts, location_coverage = [], {}
    loc_path = LOCATIONS_DIR / dataset_id / "locations.json"
    if loc_path.exists():
        locations = json.loads(loc_path.read_text(encoding="utf-8"))
        # sweep counts + coverage come from the build-time manifest; they are
        # router-facing hints — evidence counts are recomputed live at ask time
        counts, coverage = {}, {}
        loc_manifest_path = LOCATIONS_DIR / dataset_id / "manifest.json"
        if loc_manifest_path.exists():
            m = json.loads(loc_manifest_path.read_text(encoding="utf-8"))
            counts = m.get("concept_counts", {})
            coverage = m.get("labeled_location_coverage", {})
        location_concepts = [
            {"name": c["name"], "kind": c["kind"],
             "n_spans": len(c.get("spans", [])),
             "count": counts.get(c["name"], 0)}
            for c in locations.get("concepts", [])
        ]
        location_coverage = coverage

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "dataset_id": dataset_id,
        "dataset_description": dataset_description,
        "questions": questions,
        "lexicon_concepts": lexicon_concepts,
        "location_concepts": location_concepts,
        "location_coverage": location_coverage,
    }


def render_summary(summary: dict) -> str:
    """The prompt-facing flat rendering. One line per label:
    `label_id | parent > name — description (n=count)`. Counts are computed;
    the router is told exactly that so it never re-estimates them."""
    lines: list[str] = []
    desc = (summary.get("dataset_description") or "").strip()
    if desc:
        lines += [f"Survey: {desc}", ""]
    for q in summary["questions"]:
        lines.append(
            f'Question {q["question_id"]}: "{q["question_text"]}" '
            f'({q["n_responses"]} coded responses, '
            f'{q["n_uncategorized"]} uncategorized)'
        )
        for e in q["entries"]:
            d = e["description"]
            if len(d) > MAX_DESC_CHARS:
                d = d[:MAX_DESC_CHARS] + "…"
            parent = e["parent_name"] or "(no parent)"
            lines.append(f'  {e["label_id"]} | {parent} > {e["name"]} — {d} (n={e["count"]})')
        lines.append("")
    if summary["lexicon_concepts"]:
        names = ", ".join(c["name"] for c in summary["lexicon_concepts"])
        lines.append(f"Lexicon concepts (deterministic keyword matcher): {names}")
    loc_concepts = summary.get("location_concepts") or []
    if loc_concepts:
        lines.append("")
        lines.append("Locations (verbatim place mentions; matched deterministically; "
                     "n = responses mentioning it):")
        for kind, label in [("named", "named places"), ("type", "place types")]:
            items = [c for c in loc_concepts if c["kind"] == kind]
            if items:
                rendered = ", ".join(f'{c["name"]} (n={c["count"]})' for c in items)
                lines.append(f"  {label}: {rendered}")
        cov = summary.get("location_coverage") or {}
        if cov:
            per_q = "; ".join(
                f'q{q}: {c["responses_with_location"]}/{c["responses"]}'
                for q, c in sorted(cov.items(), key=lambda kv: (len(kv[0]), kv[0])))
            lines.append(f"  responses naming any place: {per_q} — only these are "
                         "localizable; where-answers must disclose this denominator")
    return "\n".join(lines).rstrip() + "\n"


def write_summary(summary: dict, out_root: Path | None = None) -> Path:
    out_dir = (out_root or SUMMARY_DIR) / summary["dataset_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    return out_dir


def valid_label_ids(summary: dict) -> set[str]:
    return {e["label_id"] for q in summary["questions"] for e in q["entries"]}


def label_index(summary: dict) -> dict[str, dict]:
    """label_id -> entry, with the owning question_id attached."""
    out = {}
    for q in summary["questions"]:
        for e in q["entries"]:
            out[e["label_id"]] = {**e, "question_id": q["question_id"]}
    return out
