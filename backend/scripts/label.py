"""Label every response for a question against its induced taxonomy.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.label --question 2 --dry-run
    python -m scripts.label --question 2
    python -m scripts.label --all

Uses the most recent candidate_taxonomy.json for the question unless --taxonomy
is given. Output: data/labels/{dataset_id}/{question_id}/{run_id}/
        assignments.json  <- one record per response
        manifest.json     <- audit record + counts
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from app import induction, labeling, llm
from app.llm import GeminiClient

LABELS_DIR = induction.DATA_DIR / "labels"


def latest_taxonomy(dataset_id: str, question_id: str) -> Path:
    hits = sorted(glob.glob(
        str(induction.TAXONOMY_DIR / dataset_id / question_id / "*" / "candidate_taxonomy.json")))
    if not hits:
        raise SystemExit(f"No taxonomy for dataset {dataset_id} question {question_id}. "
                         f"Run scripts.induce first.")
    return Path(hits[-1])


def label_one(args, question: str) -> dict:
    parquet = induction.discover_parquet(args.parquet)
    rows, meta, _filtered = induction.load_question(parquet, question)
    tax_path = Path(args.taxonomy) if args.taxonomy else latest_taxonomy(
        meta["dataset_id"], meta["question_id"])
    taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))

    pairs = [(r.response_key, r.text) for r in rows]
    n_unique = len({t for _, t in pairs})
    n_batches = -(-n_unique // args.batch_size)
    print(f"\nq{question}: \"{meta['question_text']}\" — {len(pairs)} responses "
          f"({n_unique} unique texts), "
          f"{len(taxonomy['labels'])} labels, {n_batches} batches")
    print(f"  taxonomy: {tax_path.parent.name}")

    if args.dry_run:
        # Delegates to labeling.plan_labeling, which builds the prompts the run
        # will really send and counts them with Vertex's free countTokens.
        # What this replaced got two things wrong: it sampled its example
        # prompt from the RAW rows while the run batches deduped ones, so on
        # any corpus with repeated answers it measured a prompt that is never
        # sent; and it extrapolated the first batch's length to every batch,
        # which is false whenever response lengths vary and always false for
        # the short final batch.
        plan = labeling.plan_labeling(
            pairs, taxonomy, batch_size=args.batch_size,
            dataset_description=args.description)
        est_in, est_out = plan["est_input_tokens"], plan["est_output_tokens"]
        cost = est_in / 1e6 * args.price_in + est_out / 1e6 * args.price_out
        print(f"  DRY RUN — {plan['n_batches']} calls, ~{est_in:,} in "
              f"({plan['input_basis']}) / ~{est_out:,} out, ~${cost:.3f}")
        return {}

    client = GeminiClient(model=args.model, max_output_tokens=args.max_output_tokens,
                          min_interval_s=args.min_interval)
    t0 = time.time()
    assignments, report = labeling.run_labeling(
        pairs, taxonomy, client, batch_size=args.batch_size,
        dataset_description=args.description, workers=args.workers)
    elapsed = time.time() - t0
    cost = client.usage.cost_usd(args.price_in, args.price_out)

    # stamped with the LABELING prompt hash, not induction's — an edit to
    # LABEL_SYSTEM has to change the run id, or two runs that disagree on a
    # third of their rows land in indistinguishable directories
    run_id = f"{induction.utc_now()}_{labeling.prompt_hash(args.description)[:8]}"
    out_dir = LABELS_DIR / meta["dataset_id"] / meta["question_id"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assignments.json").write_text(
        json.dumps(assignments, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_utc": induction.utc_now(),
        "tool": "scripts.label",
        "schema_version": labeling.SCHEMA_VERSION,
        "model_id": client.model_id,
        "temperature": 0.0,
        "prompt_sha256_16": labeling.prompt_hash(args.description),
        # the taxonomy this run consumed was built by a different prompt set;
        # recording both makes a labels run traceable to each independently
        "induction_prompt_sha256_16": induction.prompt_hash(args.description),
        "dataset_description": args.description,
        "taxonomy_run": tax_path.parent.name,
        "taxonomy_path": str(tax_path),
        "source": meta,
        "report": report,
        "usage": {
            "calls": client.usage.calls,
            "input_tokens": client.usage.input_tokens,
            "output_tokens": client.usage.output_tokens,
            "est_cost_usd": round(cost, 4),
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Best-effort: fold the fresh labels into the export files so the
    # analyst-facing parquet/CSV never lag the labels JSON. Must never fail
    # a paid labeling run — POST /datasets/{id}/export is the manual path.
    try:
        from app.db import SessionLocal
        from app.models import Dataset
        from app import ingest

        with SessionLocal() as db:
            ds = db.get(Dataset, int(meta["dataset_id"]))
            if ds is not None and ds.status == "ingested":
                ingest._write_exports(db, ds)
                print("  exports refreshed with labels")
    except Exception as exc:
        print(f"  NOTE: export refresh skipped ({exc}); "
              f"use POST /datasets/{meta['dataset_id']}/export")

    r = report
    print(f"  labelled {r['responses_labelled']}/{r['responses_total']}, "
          f"uncategorized {r['responses_uncategorized']} "
          f"({r['responses_uncategorized']/r['responses_total']:.0%}), "
          f"mean {r['labels_per_response_mean']} labels/response")
    print(f"  invalid ids dropped: {r['invalid_ids_dropped']} | "
          f"not returned: {r['responses_not_returned']} | "
          f"zero-count labels: {len(r['labels_with_zero_responses'])}")
    print(f"  locations: {r['responses_with_locations']} responses mention one | "
          f"hallucinated locations dropped: {r['invalid_locations_dropped']}")
    print(f"  time context: {r['responses_with_time_context']} responses | "
          f"hallucinated dropped: {r['invalid_time_context_dropped']} | "
          f"actionability: {r['actionability_distribution']}")
    print(f"  events: {r['responses_with_event_occurred']} of "
          f"{r['responses_event_coded']} coded responses report an incident")
    print(f"  fit: {r['fit_distribution']}")
    if r["failed_batches"]:
        print(f"  BATCHES FAILED: {len(r['failed_batches'])} — those responses are unlabelled")
    print(f"  ${cost:.3f}, {elapsed:.0f}s -> {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question")
    ap.add_argument("--all", action="store_true", help="label every question in the parquet")
    ap.add_argument("--parquet")
    ap.add_argument("--taxonomy", help="explicit candidate_taxonomy.json (default: most recent)")
    ap.add_argument("--batch-size", type=int, default=labeling.DEFAULT_BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=labeling.DEFAULT_WORKERS,
                    help="concurrent batches (1 = serial)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--min-interval", type=float, default=0.1,
                    help="minimum seconds between API requests across all workers (0 disables)")
    ap.add_argument("--description", default=None,
                    help="default: the export manifest's dataset_description")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    args = ap.parse_args()
    # $/MTok: the flags when given, otherwise the configured rate for the
    # model this run will actually use. Every script used to default these
    # to a hand-copied 0.30/2.50, which was silently wrong the moment
    # --model pointed elsewhere and never tracked config.json at all.
    args.price_in, args.price_out = llm.resolve_prices(
        llm.resolve_model(args.model), args.price_in, args.price_out)

    if not args.question and not args.all:
        ap.error("pass --question or --all")

    parquet = induction.discover_parquet(args.parquet)
    args.description = induction.resolve_description(args.description, parquet)
    questions = ([q["question_id"] for q in induction.list_questions(parquet)]
                 if args.all else [args.question])
    total = 0.0
    for q in questions:
        m = label_one(args, q)
        total += (m.get("usage") or {}).get("est_cost_usd", 0.0)
    if len(questions) > 1 and not args.dry_run:
        print(f"\ntotal: ${total:.3f} across {len(questions)} questions")


if __name__ == "__main__":
    sys.exit(main())
