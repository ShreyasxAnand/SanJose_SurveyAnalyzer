"""Induce a candidate taxonomy for one question.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.induce --list                       # see questions
    python -m scripts.induce --question q4oe --dry-run    # plan + cost, no API
    python -m scripts.induce --question q4oe              # real run

Needs GEMINI_API_KEY (or GOOGLE_API_KEY) for real runs, taken from the
environment or from the gitignored .env at the repo root.
Output: data/taxonomy/{dataset_id}/{question_id}/{run_id}/
        candidate_taxonomy.json  <- hand-edit this, it is the review artifact
        manifest.json            <- audit record, do not edit
"""
from __future__ import annotations

import argparse
import sys
import time

from app import induction
from app.llm import GeminiClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", help="question_id to induce (e.g. q4oe)")
    ap.add_argument("--parquet", help="path to long-format parquet (default: auto-discover under data/exports)")
    ap.add_argument("--list", action="store_true", help="list questions in the parquet and exit")
    ap.add_argument("--chunk-size", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=None, help="Gemini model id (default: env GEMINI_MODEL or gemini-3.5-flash-lite)")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--description", default=induction.DEFAULT_DATASET_DESCRIPTION,
                    help="what the survey is and who answered it (NOT what you hope to find). "
                         "Temporary: Phase 1 will collect this at upload. Pass '' to disable.")
    ap.add_argument("--dry-run", action="store_true", help="plan chunks + estimate cost; zero API calls")
    ap.add_argument("--price-in", type=float, default=0.30, help="$/MTok input, for cost reporting only")
    ap.add_argument("--price-out", type=float, default=2.50, help="$/MTok output, for cost reporting only")
    args = ap.parse_args()

    parquet = induction.discover_parquet(args.parquet)
    print(f"parquet: {parquet}")

    if args.list:
        for q in induction.list_questions(parquet):
            print(f"  {q['question_id']:<12} n={q['n_rows']:<6} {q['question_text']}")
        return
    if not args.question:
        ap.error("--question is required (or use --list)")

    rows, meta, filtered = induction.load_question(parquet, args.question)
    print(f"question: {meta['question_id']} — \"{meta['question_text']}\"")
    print(f"rows: {meta['rows_for_question']} total, {meta['rows_empty']} empty, "
          f"{meta['rows_sentinel_filtered']} sentinel non-answers filtered, "
          f"{meta['rows_used']} used")

    if args.dry_run:
        induction.estimate_dry_run(rows, meta, args.chunk_size, args.seed,
                                   args.price_in, args.price_out, args.description)
        return

    client = GeminiClient(model=args.model, max_output_tokens=args.max_output_tokens)
    print(f"model: {client.model_id} (temperature=0)")
    t0 = time.time()
    taxonomy, report = induction.run_induction(
        rows, meta, client, chunk_size=args.chunk_size, seed=args.seed,
        dataset_description=args.description,
    )
    elapsed = time.time() - t0

    cost = client.usage.cost_usd(args.price_in, args.price_out)
    manifest = {
        "run_id": f"{induction.utc_now()}_{induction.prompt_hash(args.description)[:8]}",
        "created_utc": induction.utc_now(),
        "tool": "scripts.induce",
        "schema_version": induction.SCHEMA_VERSION,
        "model_id": client.model_id,
        "temperature": 0.0,
        "prompt_sha256_16": induction.prompt_hash(args.description),
        "dataset_description": args.description,
        "source": meta,
        "sentinel_non_answers_filtered": filtered,
        "pii_note": (
            "Response text may contain first names and addresses. NOT scrubbed: "
            "San José streets/parks collide with person names (Julian, Story, "
            "Alma, Hedding, Kelley, Roosevelt) and naive NER destroys location "
            "signal. Resolve before any external sharing of artifacts."
        ),
        "run_report": report,
        "usage": {
            "calls": client.usage.calls,
            "input_tokens": client.usage.input_tokens,
            "output_tokens": client.usage.output_tokens,
            "est_cost_usd": round(cost, 4),
            "price_in_per_mtok": args.price_in,
            "price_out_per_mtok": args.price_out,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    out_dir = induction.write_artifacts(taxonomy, manifest)

    usage_line = (f"usage: {client.usage.calls} calls, "
                  f"{client.usage.input_tokens:,} in / {client.usage.output_tokens:,} out tokens, "
                  f"~${cost:.3f} at ${args.price_in}/${args.price_out} per MTok, "
                  f"{elapsed:.0f}s")
    induction.print_diagnostics(taxonomy, report, usage_line)
    print(f"\nwrote: {out_dir / 'candidate_taxonomy.json'}")
    print(f"       {out_dir / 'manifest.json'}")
    print("Next: hand-review candidate_taxonomy.json (see review_instructions inside).")


if __name__ == "__main__":
    sys.exit(main())
