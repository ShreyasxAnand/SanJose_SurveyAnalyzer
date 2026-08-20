"""Label only the never-labeled (appended) rows of a question, extending the
taxonomy when they don't fit.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.label_incremental --question 2 --dry-run
    python -m scripts.label_incremental --question 2 --parquet ../data/exports/1/responses.parquet

Requires an existing labels run for the question (that is what "never
labeled" is measured against) — a question with a taxonomy but no labels
belongs to scripts.label. Output mirrors a review run: taxonomy AND
assignments written under the same new `{utc}_incr` run id, so summary pairs
them via same_run_id and every downstream consumer picks them up as "latest"
with zero changes. Prior run dirs are never touched.

New labels (if the uncovered pool is big enough to induce over) carry
provenance {"source": "incremental"} and needs_review=True so they are easy
to find later — scripts.review remains the optional repair loop for them;
this pass just refuses to leave appended rows sitting in uncategorized when
the taxonomy has an obvious gap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app import incremental, induction, labeling
from app.llm import GeminiClient
from app.summary import LABELS_DIR, latest_run_dir
from scripts.label import latest_taxonomy


class NoPriorLabelsRun(Exception):
    """The question has a taxonomy but no labels run, so there is nothing to
    measure "never labeled" against — that is scripts.label's job. Raised
    rather than exiting so `--all` can skip the question and still finish (and
    still report) the ones it can do."""


def run_one(args, question: str) -> dict:
    parquet = induction.discover_parquet(args.parquet)
    rows, meta, _filtered = induction.load_question(parquet, question)
    ds, q = meta["dataset_id"], meta["question_id"]

    tax_path = Path(args.taxonomy) if args.taxonomy else latest_taxonomy(ds, q)
    taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))

    labels_run = latest_run_dir(LABELS_DIR / ds / q, "assignments.json")
    if labels_run is None:
        raise NoPriorLabelsRun(
            f"No labels run for dataset {ds} question {q} — incremental labeling "
            "needs a prior run to measure 'new' against. Run scripts.label first."
        )
    prior = json.loads((labels_run / "assignments.json").read_text(encoding="utf-8"))

    new_pairs = incremental.never_labeled_pairs(rows, prior)
    print(f"\nq{q}: \"{meta['question_text']}\" — {len(rows)} rows, "
          f"{len(prior)} already labelled ({labels_run.name}), "
          f"{len(new_pairs)} new")

    if not new_pairs:
        print("  nothing to do — no run written")
        return {}

    if args.dry_run:
        n_batches = -(-len({t for _, t in new_pairs}) // args.batch_size)
        print(f"  DRY RUN — would label {len(new_pairs)} rows in {n_batches} "
              f"batches; uncovered rows (unknown until run time, bounded by "
              f"{len(new_pairs)}) may add one MAP + one ASSIGN call and a "
              f"relabel of the pool")
        return {}

    client = GeminiClient(model=args.model,
                          max_output_tokens=args.max_output_tokens,
                          min_interval_s=args.min_interval)
    t0 = time.time()
    new_tax, merged, report = incremental.run_incremental(
        rows, taxonomy, prior, client,
        batch_size=args.batch_size,
        dataset_description=args.description,
        workers=args.workers,
        min_pool=args.min_pool,
    )
    elapsed = time.time() - t0
    cost = client.usage.cost_usd(args.price_in, args.price_out)

    run_id = f"{induction.utc_now()}_incr"
    tax_dir = induction.TAXONOMY_DIR / ds / q / run_id
    asg_dir = LABELS_DIR / ds / q / run_id
    for d in (tax_dir, asg_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Taxonomy always written, even when unchanged — uniform same_run_id
    # pairing, exactly like a review run.
    (tax_dir / "candidate_taxonomy.json").write_text(
        json.dumps(new_tax, indent=2, ensure_ascii=False), encoding="utf-8")
    (asg_dir / "assignments.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": induction.utc_now(),
        "tool": "scripts.label_incremental",
        "schema_version": labeling.SCHEMA_VERSION,
        "model_id": client.model_id,
        "temperature": 0.0,
        "prompt_sha256_16": labeling.prompt_hash(args.description),
        "induction_prompt_sha256_16": induction.prompt_hash(args.description),
        "dataset_description": args.description,
        # only the new rows (and the relabelled pool) went through the model;
        # everything else was carried over verbatim from derived_from
        "derived_from": {
            "taxonomy": str(tax_path),
            "assignments": str(labels_run / "assignments.json"),
        },
        "source": meta,
        "n_new_rows": report["n_new_rows"],
        "n_pool": report["n_pool"],
        "pool_induction": report["pool_induction"],
        "new_label_ids": report["new_label_ids"],
        "report": report["label_report"],
        "relabel_report": report["relabel_report"],
        "usage": {
            "calls": client.usage.calls,
            "input_tokens": client.usage.input_tokens,
            "output_tokens": client.usage.output_tokens,
            "est_cost_usd": round(cost, 4),
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    for d in (tax_dir, asg_dir):
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Best-effort: fold the merged labels into the export files so the
    # analyst-facing parquet/CSV never lag the labels JSON. Must never fail a
    # paid run — POST /datasets/{id}/export is the manual path.
    try:
        from app.db import SessionLocal
        from app.models import Dataset
        from app import ingest

        with SessionLocal() as db:
            ds_row = db.get(Dataset, int(ds))
            if ds_row is not None and ds_row.status == "ingested":
                ingest._write_exports(db, ds_row)
                print("  exports refreshed with labels")
    except Exception as exc:
        print(f"  NOTE: export refresh skipped ({exc}); "
              f"use POST /datasets/{ds}/export")

    lr = report["label_report"]
    print(f"  labelled {lr['responses_labelled']}/{lr['responses_total']} new rows, "
          f"uncategorized {lr['responses_uncategorized']}")
    if report["new_label_ids"]:
        print(f"  new labels (needs_review): {report['new_label_ids']}")
    else:
        print(f"  pool induction: "
              f"{report['pool_induction'].get('outcome') or 'ran, no labels added'}")
    print(f"  ${cost:.4f}, {elapsed:.0f}s -> {asg_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question")
    ap.add_argument("--all", action="store_true",
                    help="incrementally label every question in the parquet")
    ap.add_argument("--parquet",
                    help="explicit responses.parquet (the pipeline always passes "
                         "this; the mtime-based default is only safe with one "
                         "dataset on disk)")
    ap.add_argument("--taxonomy", help="explicit candidate_taxonomy.json (default: most recent)")
    ap.add_argument("--batch-size", type=int, default=labeling.DEFAULT_BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=labeling.DEFAULT_WORKERS)
    ap.add_argument("--min-pool", type=int, default=incremental.INCREMENTAL_MIN_POOL,
                    help="below this many uncovered rows, skip pool induction")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--min-interval", type=float, default=0.1)
    ap.add_argument("--description", default=None,
                    help="default: the export manifest's dataset_description")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--price-in", type=float, default=0.30)
    ap.add_argument("--price-out", type=float, default=2.50)
    args = ap.parse_args()

    if not args.question and not args.all:
        ap.error("pass --question or --all")

    parquet = induction.discover_parquet(args.parquet)
    args.description = induction.resolve_description(args.description, parquet)
    questions = ([q["question_id"] for q in induction.list_questions(parquet)]
                 if args.all else [args.question])
    total = 0.0
    skipped: list[str] = []
    for q in questions:
        try:
            m = run_one(args, q)
        except NoPriorLabelsRun as exc:
            # With one explicit --question this is a hard error; under --all it
            # must not discard the questions already processed in this run.
            if not args.all:
                raise SystemExit(str(exc))
            print(f"\nq{q}: SKIPPED — {exc}")
            skipped.append(q)
            continue
        total += (m.get("usage") or {}).get("est_cost_usd", 0.0)
    if len(questions) > 1 and not args.dry_run:
        print(f"\ntotal: ${total:.4f} across "
              f"{len(questions) - len(skipped)} question(s)")
    if skipped:
        print(f"skipped (no prior labels run — use scripts.label): "
              f"{', '.join(skipped)}")


if __name__ == "__main__":
    sys.exit(main())
