"""Post-labeling review: generate the defect report, apply approved edits.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.review --question 2                  # report + template
    python -m scripts.review --question 2 --edits e.json --dry-run
    python -m scripts.review --question 2 --edits e.json   # apply

Report mode is free (no model calls). Apply mode is free too unless the edits
add a new label, in which case only the uncovered pool (uncategorized/fit=1)
is relabeled — a few dozen responses, never the corpus.

Outputs:
    data/review/{dataset_id}/{question_id}/report.md
    data/review/{dataset_id}/{question_id}/edits_template.json
    data/taxonomy/{ds}/{q}/{run}_review/candidate_taxonomy.json   (on apply)
    data/labels/{ds}/{q}/{run}_review/assignments.json            (on apply)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from app import induction, labeling, review

REVIEW_DIR = induction.DATA_DIR / "review"
LABELS_DIR = induction.DATA_DIR / "labels"


def latest(pattern: str, what: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"No {what} found ({pattern}).")
    return Path(hits[-1])


def load_state(args, question: str):
    parquet = induction.discover_parquet(args.parquet)
    rows, meta, _ = induction.load_question(parquet, question)
    ds = meta["dataset_id"]
    tax_path = (Path(args.taxonomy) if args.taxonomy else latest(
        str(induction.TAXONOMY_DIR / ds / question / "*" / "candidate_taxonomy.json"),
        f"taxonomy for q{question}"))
    asg_path = (Path(args.assignments) if args.assignments else latest(
        str(LABELS_DIR / ds / question / "*" / "assignments.json"),
        f"assignments for q{question}"))
    taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))
    assignments = json.loads(asg_path.read_text(encoding="utf-8"))
    texts = {r.response_key: r.text for r in rows}
    return taxonomy, assignments, texts, tax_path, asg_path, meta


def cmd_report(args, question: str) -> None:
    taxonomy, assignments, texts, tax_path, asg_path, meta = load_state(args, question)
    rep = review.diagnose(taxonomy, assignments)
    out_dir = REVIEW_DIR / meta["dataset_id"] / question
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(
        review.render_report(taxonomy, rep, texts), encoding="utf-8")
    (out_dir / "edits_template.json").write_text(json.dumps({
        "_taxonomy": str(tax_path), "_assignments": str(asg_path),
        "edits": [], "suggestions": review.suggest_edits(rep),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"q{question}: {rep['n_responses']} responses, {rep['n_labels']} labels, "
          f"{rep['pct_uncategorized']:.0%} uncategorized")
    print(f"  merge candidates: {len(rep['merge_candidates'])} | "
          f"duplicate names: {len(rep['duplicate_names'])} | "
          f"orphans: {len(rep['orphan_labels'])} | "
          f"zero-count: {len(rep['zero_count_labels'])} | "
          f"uncovered pool: {len(rep['missing_category_pool'])}")
    print(f"  -> {out_dir / 'report.md'}")


def cmd_apply(args, question: str) -> None:
    taxonomy, assignments, texts, tax_path, asg_path, meta = load_state(args, question)
    edits = json.loads(Path(args.edits).read_text(encoding="utf-8"))["edits"]
    if not edits:
        raise SystemExit(f"{args.edits} has an empty 'edits' list — copy the ops "
                         "you approve from 'suggestions' into 'edits'.")

    before = review.diagnose(taxonomy, assignments)
    new_tax, new_asg, log = review.apply_edits(taxonomy, assignments, edits)
    for line in log:
        print(f"  {line}")

    added = review.new_label_ids(edits)
    pool_keys = set(before["missing_category_pool"]) if added else set()
    if args.dry_run:
        print(f"\nDRY RUN — {len(log)} ops valid. "
              f"{'Relabel of ' + str(len(pool_keys)) + ' pooled responses needed.' if added else 'No relabeling needed.'}")
        return

    relabel_report = None
    cost = 0.0
    if added:
        from app.llm import GeminiClient
        pool = [(k, texts[k]) for k in pool_keys if k in texts]
        print(f"\nrelabeling uncovered pool ({len(pool)} responses) against "
              f"updated taxonomy...")
        client = GeminiClient(model=args.model)
        t0 = time.time()
        pool_asg, relabel_report = labeling.run_labeling(
            pool, new_tax, client, dataset_description=args.description)
        cost = client.usage.cost_usd(args.price_in, args.price_out)
        print(f"  ${cost:.4f}, {time.time() - t0:.0f}s")
        by_key = {a["response_key"]: a for a in pool_asg}
        new_asg = [
            {**by_key[a["response_key"]], "relabelled_in_review": True}
            if a["response_key"] in by_key else a
            for a in new_asg
        ]

    after = review.diagnose(new_tax, new_asg)
    run_id = f"{induction.utc_now()}_review"
    ds = meta["dataset_id"]
    tax_dir = induction.TAXONOMY_DIR / ds / question / run_id
    asg_dir = LABELS_DIR / ds / question / run_id
    for d in (tax_dir, asg_dir):
        d.mkdir(parents=True, exist_ok=True)
    (tax_dir / "candidate_taxonomy.json").write_text(
        json.dumps(new_tax, indent=2, ensure_ascii=False), encoding="utf-8")
    (asg_dir / "assignments.json").write_text(
        json.dumps(new_asg, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "run_id": run_id, "created_utc": induction.utc_now(),
        "tool": "scripts.review", "schema_version": review.SCHEMA_VERSION,
        "derived_from": {"taxonomy": str(tax_path), "assignments": str(asg_path)},
        "edits_file": str(Path(args.edits).resolve()),
        "edits_applied": edits, "change_log": log,
        "relabelled_pool_size": len(pool_keys),
        "relabel_report": relabel_report,
        "relabel_cost_usd": round(cost, 4),
        "diagnostics_before": {k: v for k, v in before.items()
                               if k != "missing_category_pool"},
        "diagnostics_after": {k: v for k, v in after.items()
                              if k != "missing_category_pool"},
    }
    for d in (tax_dir, asg_dir):
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nbefore: {before['n_labels']} labels, "
          f"{before['pct_uncategorized']:.0%} uncategorized, "
          f"{len(before['merge_candidates'])} merge candidates, "
          f"{len(before['orphan_labels'])} orphans, "
          f"{len(before['duplicate_names'])} dup names")
    print(f"after:  {after['n_labels']} labels, "
          f"{after['pct_uncategorized']:.0%} uncategorized, "
          f"{len(after['merge_candidates'])} merge candidates, "
          f"{len(after['orphan_labels'])} orphans, "
          f"{len(after['duplicate_names'])} dup names")
    print(f"-> {tax_dir}\n-> {asg_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", required=True)
    ap.add_argument("--parquet")
    ap.add_argument("--taxonomy", help="explicit taxonomy path (default: most recent)")
    ap.add_argument("--assignments", help="explicit assignments path (default: most recent)")
    ap.add_argument("--edits", help="edits JSON to apply; omit for report mode")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--description", default=induction.DEFAULT_DATASET_DESCRIPTION)
    ap.add_argument("--price-in", type=float, default=0.30)
    ap.add_argument("--price-out", type=float, default=2.50)
    args = ap.parse_args()

    if args.edits:
        cmd_apply(args, args.question)
    else:
        cmd_report(args, args.question)


if __name__ == "__main__":
    sys.exit(main())
