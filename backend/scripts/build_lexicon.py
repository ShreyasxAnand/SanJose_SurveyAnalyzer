"""Derive the keyword dictionary (layer 2) from the corpus.

    python -m scripts.build_lexicon              # build + report coverage
    python -m scripts.build_lexicon --show-terms # print raw extracted candidates

One model call, then matching is deterministic and free forever after.
Output: data/lexicon/{dataset_id}/lexicon.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from app import induction, lexicon as lex
from app.llm import GeminiClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet")
    ap.add_argument("--model", default=None)
    ap.add_argument("--description", default=induction.DEFAULT_DATASET_DESCRIPTION)
    ap.add_argument("--show-terms", action="store_true", help="print candidates and exit")
    ap.add_argument("--price-in", type=float, default=0.30)
    ap.add_argument("--price-out", type=float, default=2.50)
    args = ap.parse_args()

    parquet = induction.discover_parquet(args.parquet)
    df = pd.read_parquet(parquet)
    texts = [str(t) for t in df["response_text"].tolist()]
    keys = [str(k) for k in df["response_key"].tolist()]
    dataset_id = str(df["dataset_id"].iloc[0])
    print(f"parquet: {parquet}\ncorpus: {len(texts)} responses, dataset {dataset_id}")

    if args.show_terms:
        rendered, n = lex.render_candidates(lex.extract_candidates(texts))
        print(f"\n{n} candidate terms:\n{rendered}")
        return

    client = GeminiClient(model=args.model)
    t0 = time.time()
    lexicon, warnings = lex.build_lexicon(texts, client, args.description)
    elapsed = time.time() - t0
    cost = client.usage.cost_usd(args.price_in, args.price_out)

    hits = lex.match_responses(lexicon, keys, texts)
    covered = {k for ks in hits.values() for k in ks}

    manifest = {
        "created_utc": induction.utc_now(),
        "tool": "scripts.build_lexicon",
        "schema_version": lex.SCHEMA_VERSION,
        "model_id": client.model_id,
        "dataset_description": args.description,
        "n_responses": len(texts),
        "n_concepts": len(lexicon["concepts"]),
        "responses_matching_any_concept": len(covered),
        "concept_counts": {name: len(ks) for name, ks in hits.items()},
        "warnings": warnings,
        "usage": {"calls": client.usage.calls, "est_cost_usd": round(cost, 4),
                  "elapsed_seconds": round(elapsed, 1)},
    }
    out_dir = lex.write_lexicon(lexicon, manifest, dataset_id)

    print(f"\n{len(lexicon['concepts'])} concepts, "
          f"{len(covered)}/{len(texts)} responses match at least one "
          f"({len(covered)/len(texts):.0%})")
    for c in sorted(lexicon["concepts"], key=lambda c: -len(hits[c["name"]])):
        print(f"  {len(hits[c['name']]):>5}  {c['name']:<34} {', '.join(c['terms'][:7])}")
    for w in warnings[:10]:
        print(f"  warn: {w}")
    if len(warnings) > 10:
        print(f"  ... {len(warnings) - 10} more warnings")
    print(f"\n${cost:.4f}, {elapsed:.0f}s -> {out_dir / 'lexicon.json'}")


if __name__ == "__main__":
    sys.exit(main())
