"""Incremental labeling for appended rows — the paid half of duplicate-aware
ingest.

When a file is appended to a dataset, only its genuinely new rows lack labels.
This module labels exactly those rows against the existing (latest) taxonomy,
and gives rows the taxonomy cannot place a second chance: an induction-style
MAP pass over the uncovered pool proposes new labels, which are appended to
the taxonomy with provenance {"source": "incremental"} and needs_review=True —
automatic, but never silent — and the pool is relabeled against the extended
taxonomy.

Mirrors scripts/review.py's cmd_apply step for step (it is the same
label-a-pool-and-overlay problem):
- the base assignment list is carried over verbatim; only fresh rows are new
  records, stamped "labeled_incrementally", and pool rows that went through
  the second pass are additionally stamped "relabelled_incremental";
- taxonomy + assignments are written under the SAME new run id (the caller's
  job — see scripts/label_incremental.py) so summary pairs them via
  same_run_id, exactly like a review run;
- old run dirs are never touched.

New labels never create new parents: parents are the shared, human-edited
layer, and an auto-created parent would bypass the review gate. A proposal the
ASSIGN call can't place under an existing parent becomes an orphan
(parent_id=None), which review.diagnose already surfaces.
"""

from __future__ import annotations

import copy
import json
import re

from app import induction, labeling
from app.induction import Candidate, ResponseRow

# Below this many uncovered rows, skip pool induction entirely: a MAP call
# over a handful of responses produces n=1 noise labels, and the review loop
# is the right place for a pool that small.
INCREMENTAL_MIN_POOL = 8

# The uncovered predicate — identical to review.diagnose's missing-category
# pool so the two loops never disagree about what "didn't fit" means.
def _uncovered(assignment: dict) -> bool:
    return bool(assignment.get("uncategorized")) or assignment.get("fit") == 1


def never_labeled_pairs(
    rows: list[ResponseRow], prior_assignments: list[dict]
) -> list[tuple[str, str]]:
    """Rows present in the corpus but absent from the prior labels run — the
    appended rows. The assignments artifact is the record; the parquet's
    label columns are derived from it and are not consulted."""
    prior_keys = {a.get("response_key") for a in prior_assignments}
    return [(r.response_key, r.text) for r in rows if r.response_key not in prior_keys]


def next_incremental_seq(taxonomy: dict) -> int:
    """Next free NNN for the {question_id}_iNNN id scheme. The 'i' prefix is
    structurally collision-free against induction's {q}_NNN and review's
    hand-picked ids; existing _i ids are scanned so repeated incremental runs
    keep counting up."""
    q = taxonomy["question_id"]
    pat = re.compile(rf"^{re.escape(str(q))}_i(\d+)$")
    seqs = [
        int(m.group(1))
        for lab in taxonomy["labels"]
        if (m := pat.match(str(lab["label_id"])))
    ]
    return max(seqs, default=0) + 1


def propose_new_labels(
    pool_rows: list[ResponseRow],
    taxonomy: dict,
    client,
    dataset_description: str = "",
) -> tuple[list[dict], dict]:
    """One MAP call over the uncovered pool (single chunk — evidence numbers
    resolve to response_keys in code, so hallucinated quotes stay structurally
    impossible), then one ASSIGN call to attach proposals to EXISTING parents.
    Returns (new label dicts, report). Proposals whose normalized name matches
    an existing label are dropped — the pool row just didn't get that label,
    which is the labeling pass's business, not a taxonomy gap."""
    question_text = taxonomy["question_text"]

    system, user, _ = induction.build_map_prompts(
        question_text, pool_rows, dataset_description
    )
    raw = client.complete(system, user)
    candidates, n_invalid = induction.parse_map_output(raw, 0, pool_rows)

    existing_names = {induction._norm_name(l["name"]) for l in taxonomy["labels"]}
    merged = induction.auto_merge(candidates)
    provisional = [
        p for p in merged if induction._norm_name(p.name) not in existing_names
    ]
    dropped_existing = len(merged) - len(provisional)

    report = {
        "pool_size": len(pool_rows),
        "map_candidates": len(candidates),
        "map_invalid_citations": n_invalid,
        "dropped_name_matches_existing": dropped_existing,
        "assign_warnings": [],
    }
    if not provisional:
        return [], report

    themes = [
        {"name": p["name"], "description": p.get("description", "")}
        for p in taxonomy.get("parents", [])
    ]
    parent_id_by_name = {p["name"]: p["parent_id"] for p in taxonomy.get("parents", [])}
    mapping: dict[str, str] = {}
    if themes:
        # Contained like induction's own ASSIGN batches: a malformed response
        # must not discard MAP's successful proposals — they become orphans
        # (parent_id=None), which review.diagnose already surfaces.
        try:
            system, user = induction.build_assign_batch_prompts(
                question_text, themes, provisional, dataset_description
            )
            raw = client.complete(system, user)
            mapping, warnings = induction.parse_assign_batch(
                raw, provisional, [t["name"] for t in themes]
            )
            report["assign_warnings"] = warnings
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            mapping = {}
            report["assign_warnings"] = [
                f"ASSIGN failed ({type(exc).__name__}: {exc}); proposals kept "
                "as orphans for review"
            ]

    seq = next_incremental_seq(taxonomy)
    existing_ids = {str(l["label_id"]) for l in taxonomy["labels"]}
    new_labels: list[dict] = []
    for offset, prov in enumerate(provisional):
        label_id = f"{taxonomy['question_id']}_i{seq + offset:03d}"
        if label_id in existing_ids:
            raise ValueError(f"incremental label id already exists: {label_id}")
        parent_name = mapping.get(prov.pid)
        examples = [
            {"response_key": r.response_key, "text": r.text}
            for m in prov.members
            for r in m.evidence
        ][: induction.MAX_EXAMPLES_PER_LABEL]
        new_labels.append(
            {
                "label_id": label_id,
                "parent_id": parent_id_by_name.get(parent_name),
                "name": prov.name,
                "description": prov.description,
                "include": prov.include,
                "exclude": prov.exclude,
                "examples": examples,
                "chunk_support": None,
                "chunk_support_note": (
                    f"proposed by the incremental pass over {len(pool_rows)} "
                    "uncovered appended rows, not from full induction"
                ),
                "singleton": False,
                "needs_review": True,
                "provenance": {
                    "source": "incremental",
                    "pool_size": len(pool_rows),
                },
            }
        )
    return new_labels, report


def extend_taxonomy(taxonomy: dict, new_labels: list[dict]) -> dict:
    """Append new labels to a deep copy — never mutate a prior run's artifact
    in memory either. Child ids are registered under their parents; orphans
    (parent_id None) stay orphans for review to catch."""
    tax = copy.deepcopy(taxonomy)
    parents_by_id = {p["parent_id"]: p for p in tax.get("parents", [])}
    for lab in new_labels:
        if any(str(l["label_id"]) == lab["label_id"] for l in tax["labels"]):
            raise ValueError(f"incremental label id already exists: {lab['label_id']}")
        tax["labels"].append(lab)
        pid = lab.get("parent_id")
        if pid is not None and pid in parents_by_id:
            parents_by_id[pid]["child_label_ids"].append(lab["label_id"])
    return tax


def run_incremental(
    rows: list[ResponseRow],
    taxonomy: dict,
    prior_assignments: list[dict],
    client,
    *,
    batch_size: int = labeling.DEFAULT_BATCH_SIZE,
    dataset_description: str = "",
    workers: int = labeling.DEFAULT_WORKERS,
    min_pool: int = INCREMENTAL_MIN_POOL,
) -> tuple[dict, list[dict], dict]:
    """The whole incremental pass for one question. Returns
    (taxonomy_out, merged_assignments, report). taxonomy_out is the input
    taxonomy (possibly extended); merged_assignments is prior + fresh with
    pool rows overlaid; report discloses every step, including a skipped pool
    induction ("empty results are correct results" applies to the pool too).

    Raises ValueError when there are no new rows — the caller decides whether
    that is a no-op or an error, and no run dir should be written for it."""
    new_pairs = never_labeled_pairs(rows, prior_assignments)
    if not new_pairs:
        raise ValueError("no never-labeled rows — nothing to do")

    fresh, label_report = labeling.run_labeling(
        new_pairs,
        taxonomy,
        client,
        batch_size=batch_size,
        dataset_description=dataset_description,
        workers=workers,
    )
    for a in fresh:
        a["labeled_incrementally"] = True

    pool_assignments = [
        a
        for a in fresh
        if _uncovered(a)
        and not a.get("not_returned")
        and not a.get("batch_failed")
    ]
    texts = dict(new_pairs)
    pool_rows = [
        ResponseRow(response_key=a["response_key"], text=texts[a["response_key"]])
        for a in pool_assignments
        if a["response_key"] in texts
    ]

    # `pool_induction` is always a dict, never a bare string: it lands verbatim
    # in the run manifest, and a field that is sometimes text and sometimes an
    # object forces every reader to type-check before it can say anything.
    report: dict = {
        "n_new_rows": len(new_pairs),
        "n_pool": len(pool_rows),
        "label_report": label_report,
        "pool_induction": {"pool_size": len(pool_rows), "outcome": ""},
        "new_label_ids": [],
        "relabel_report": None,
    }

    taxonomy_out = taxonomy
    if len(pool_rows) < min_pool:
        report["pool_induction"]["outcome"] = (
            f"skipped (pool of {len(pool_rows)} below floor {min_pool}); "
            "rows stay uncategorized for the review loop"
        )
        return taxonomy_out, prior_assignments + fresh, report

    # Contained like induction's failed MAP chunks: one malformed model
    # response must not kill the run after the paid labeling above succeeded.
    # The pool just stays uncategorized for the review loop — disclosed, never
    # silent.
    try:
        new_labels, proposal_report = propose_new_labels(
            pool_rows, taxonomy, client, dataset_description
        )
    except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
        report["pool_induction"] = {
            "pool_size": len(pool_rows),
            "outcome": (
                f"pool induction failed ({type(exc).__name__}: {exc}); "
                "pool stays uncategorized for the review loop"
            ),
        }
        return taxonomy_out, prior_assignments + fresh, report
    report["pool_induction"] = {**proposal_report, "outcome": ""}
    if not new_labels:
        report["pool_induction"]["outcome"] = (
            "no new labels proposed; pool stays uncategorized"
        )
        return taxonomy_out, prior_assignments + fresh, report

    taxonomy_out = extend_taxonomy(taxonomy, new_labels)
    report["new_label_ids"] = [l["label_id"] for l in new_labels]

    # Relabel ONLY the pool against the extended taxonomy — a pooled row can
    # land on a brand-new label or on a pre-existing one it missed the first
    # time; either is a win. Overlay exactly like scripts/review.py cmd_apply.
    pool_pairs = [(r.response_key, r.text) for r in pool_rows]
    relabeled, relabel_report = labeling.run_labeling(
        pool_pairs,
        taxonomy_out,
        client,
        batch_size=batch_size,
        dataset_description=dataset_description,
        workers=workers,
    )
    report["relabel_report"] = relabel_report
    by_key = {a["response_key"]: a for a in relabeled}
    fresh = [
        {
            **by_key[a["response_key"]],
            "labeled_incrementally": True,
            "relabelled_incremental": True,
        }
        if a["response_key"] in by_key
        else a
        for a in fresh
    ]
    return taxonomy_out, prior_assignments + fresh, report
