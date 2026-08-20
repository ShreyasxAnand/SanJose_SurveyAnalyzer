"""Induce a candidate taxonomy for one question.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.induce --list                       # see questions
    python -m scripts.induce --question q4oe --dry-run    # plan + cost, no API
    python -m scripts.induce --question q4oe              # real run

Needs Google Application Default Credentials for real runs — run
`gcloud auth application-default login` once on this machine. The project
comes from GOOGLE_CLOUD_PROJECT (environment or the gitignored repo-root
.env) or the ADC file's quota project.
Output: data/taxonomy/{dataset_id}/{question_id}/{run_id}/
        candidate_taxonomy.json  <- hand-editable; labeling uses the latest
                                    run as-is, edits are optional
        manifest.json            <- audit record, do not edit
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

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
    ap.add_argument("--workers", type=int, default=induction.DEFAULT_WORKERS,
                    help="concurrent MAP calls (1 = serial)")
    ap.add_argument("--model", default=None, help="Gemini model id (default: env GEMINI_MODEL or gemini-3.5-flash-lite)")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--min-interval", type=float, default=0.1,
                    help="minimum seconds between API requests across all workers (0 disables)")
    ap.add_argument("--description", default=None,
                    help="what the survey is and who answered it (NOT what you hope to find). "
                         "Default: the export manifest's dataset_description "
                         "(collected in the ingest UI).")
    ap.add_argument("--dry-run", action="store_true", help="plan chunks + estimate cost; zero API calls")
    ap.add_argument("--resume", metavar="RUN_DIR",
                    help="resume a run whose MAP phase completed: skip MAP, run "
                         "consolidation from RUN_DIR/candidates_checkpoint.json")
    ap.add_argument("--price-in", type=float, default=0.30, help="$/MTok input, for cost reporting only")
    ap.add_argument("--price-out", type=float, default=2.50, help="$/MTok output, for cost reporting only")
    args = ap.parse_args()

    parquet = induction.discover_parquet(args.parquet)
    print(f"parquet: {parquet}")
    args.description = induction.resolve_description(args.description, parquet)

    if args.list:
        for q in induction.list_questions(parquet):
            print(f"  {q['question_id']:<12} n={q['n_rows']:<6} {q['question_text']}")
        return

    resume_flags: dict = {}
    if args.resume:
        run_dir = Path(args.resume)
        ckpt = run_dir / "candidates_checkpoint.json"
        if not ckpt.exists():
            raise SystemExit(f"No candidates_checkpoint.json in {run_dir}")
        candidates, map_report, meta, stored_sha = \
            induction.load_candidates_checkpoint(ckpt)
        current_sha = induction.prompt_hash(args.description)
        resume_flags = {"resumed_from_checkpoint": True}
        if stored_sha != current_sha:
            resume_flags["resumed_with_changed_prompts"] = True
            print(f"WARNING: prompts/description changed since the checkpointed "
                  f"MAP phase ran ({stored_sha} -> {current_sha}). Consolidation "
                  f"will use the NEW prompts against the OLD candidates.")
        run_id = run_dir.name
        print(f"resuming {run_id}: {len(candidates)} checkpointed candidate(s), "
              f"question {meta['question_id']}")
        rows, filtered = [], []
    else:
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
        # run_id and its directory are fixed BEFORE any spend so the MAP
        # checkpoint lands where --resume expects it
        run_id = f"{induction.utc_now()}_{induction.prompt_hash(args.description)[:8]}"
        run_dir = (induction.TAXONOMY_DIR / meta["dataset_id"]
                   / meta["question_id"] / run_id)

    client = GeminiClient(model=args.model, max_output_tokens=args.max_output_tokens,
                          min_interval_s=args.min_interval)
    print(f"model: {client.model_id} (temperature=0)")
    t0 = time.time()
    if args.resume:
        taxonomy, report = induction.run_consolidation(
            candidates, map_report, meta, client,
            dataset_description=args.description, workers=args.workers,
            checkpoint_hint=str(run_dir),
        )
    else:
        taxonomy, report = induction.run_induction(
            rows, meta, client, chunk_size=args.chunk_size, seed=args.seed,
            dataset_description=args.description, workers=args.workers,
            checkpoint_dir=run_dir,
        )
    elapsed = time.time() - t0

    cost = client.usage.cost_usd(args.price_in, args.price_out)
    manifest = {
        "run_id": run_id,
        "created_utc": induction.utc_now(),
        "tool": "scripts.induce",
        "schema_version": induction.SCHEMA_VERSION,
        "model_id": client.model_id,
        "temperature": 0.0,
        "prompt_sha256_16": induction.prompt_hash(args.description),
        "dataset_description": args.description,
        "source": meta,
        # counts + samples, not every full text — thousands of sentinels at
        # production scale would bloat the manifest for no audit value
        "sentinel_non_answers_filtered": {
            "count": len(filtered) if not args.resume else meta["rows_sentinel_filtered"],
            "sample": filtered[:20],
            "top_texts": Counter(f["text"].lower() for f in filtered).most_common(15),
        },
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
            # on resume this covers consolidation only — MAP spend is in the
            # original (aborted) run's terminal output, not here
            "covers": "consolidation_only" if args.resume else "full_run",
        },
        **resume_flags,
    }
    out_dir = induction.write_artifacts(taxonomy, manifest)

    usage_line = (f"usage: {client.usage.calls} calls, "
                  f"{client.usage.input_tokens:,} in / {client.usage.output_tokens:,} out tokens, "
                  f"~${cost:.3f} at ${args.price_in}/${args.price_out} per MTok, "
                  f"{elapsed:.0f}s")
    induction.print_diagnostics(taxonomy, report, usage_line)
    print(f"\nwrote: {out_dir / 'candidate_taxonomy.json'}")
    print(f"       {out_dir / 'manifest.json'}")
    print("Next: scripts.label uses this run as-is; hand-editing "
          "candidate_taxonomy.json first is optional (see review_instructions "
          "inside).")


if __name__ == "__main__":
    sys.exit(main())
