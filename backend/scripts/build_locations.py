"""Canonicalize the location spans labeling extracted (the "where" layer).

    python -m scripts.build_locations              # build + report coverage
    python -m scripts.build_locations --show-spans # print raw spans and exit

One model call, then matching is deterministic and free forever after.
Output: data/locations/{dataset_id}/locations.json
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from app import induction, llm, locations as loc
from app.llm import GeminiClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="dataset id (default: from the parquet)")
    ap.add_argument("--parquet")
    ap.add_argument("--model", default=None)
    ap.add_argument("--description", default=None,
                    help="default: the export manifest's dataset_description")
    ap.add_argument("--show-spans", action="store_true", help="print spans and exit")
    ap.add_argument("--max-spans", type=int, default=loc.MAX_SPANS,
                    help="cap on spans in the grouping prompt (top-N by count)")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    args = ap.parse_args()
    # $/MTok: the flags when given, otherwise the configured rate for the
    # model this run will actually use. Every script used to default these
    # to a hand-copied 0.30/2.50, which was silently wrong the moment
    # --model pointed elsewhere and never tracked config.json at all.
    args.price_in, args.price_out = llm.resolve_prices(
        llm.resolve_model(args.model), args.price_in, args.price_out)

    parquet = induction.discover_parquet(args.parquet)
    args.description = induction.resolve_description(args.description, parquet)
    df = pd.read_parquet(parquet)
    texts = [str(t) for t in df["response_text"].tolist()]
    keys = [str(k) for k in df["response_key"].tolist()]
    dataset_id = args.dataset or str(df["dataset_id"].iloc[0])

    by_question = loc.load_assignments_by_question(dataset_id)
    span_counts, coverage = loc.collect_spans(by_question)
    n_resp = sum(c["responses"] for c in coverage.values())
    n_with = sum(c["responses_with_location"] for c in coverage.values())
    print(f"parquet: {parquet}\ndataset {dataset_id}: {n_with}/{n_resp} labeled "
          f"responses name a place, {len(span_counts)} unique spans")

    if args.show_spans:
        rendered, _, dropped, over_cap = loc.render_spans(span_counts, args.max_spans)
        print(f"\n{rendered}\n({dropped} singleton spans below the cut, "
              f"{len(over_cap)} over the {args.max_spans}-span cap)")
        return

    client = GeminiClient(model=args.model)
    t0 = time.time()
    locations, warnings = loc.build_locations(span_counts, client, args.description,
                                              max_spans=args.max_spans)
    elapsed = time.time() - t0
    cost = client.usage.cost_usd(args.price_in, args.price_out)

    hits = loc.match_locations(locations, keys, texts)
    covered = {k for ks in hits.values() for k in ks}
    # this sweep is exactly what the ask path needs — persist it here so the
    # first question after a rebuild doesn't pay for it again
    loc.write_members(hits, dataset_id, locations, keys, texts)

    manifest = {
        "created_utc": induction.utc_now(),
        "tool": "scripts.build_locations",
        "schema_version": loc.SCHEMA_VERSION,
        "model_id": client.model_id,
        "dataset_description": args.description,
        "n_responses": len(texts),
        "n_unique_spans": len(span_counts),
        "max_spans": args.max_spans,
        "n_spans_over_cap_dropped": locations["n_spans_over_cap_dropped"],
        "n_concepts": len(locations["concepts"]),
        "responses_matching_any_concept": len(covered),
        "labeled_location_coverage": coverage,
        "concept_counts": {name: len(ks) for name, ks in hits.items()},
        "members_cache": str(loc.members_path(dataset_id)),
        "warnings": warnings,
        "usage": {"calls": client.usage.calls, "est_cost_usd": round(cost, 4),
                  "elapsed_seconds": round(elapsed, 1)},
    }
    out_dir = loc.write_locations(locations, manifest, dataset_id)

    named = [c for c in locations["concepts"] if c["kind"] == "named"]
    types = [c for c in locations["concepts"] if c["kind"] == "type"]
    print(f"\n{len(named)} named places + {len(types)} place types; "
          f"{len(covered)}/{len(texts)} responses mention at least one "
          f"({len(covered)/len(texts):.0%} — sweep counts, superset of labeled spans)")
    for c in sorted(locations["concepts"], key=lambda c: -len(hits[c["name"]])):
        print(f"  {len(hits[c['name']]):>5}  [{c['kind']:<5}] {c['name']:<28} "
              f"{', '.join(c['spans'][:6])}")
    for w in warnings[:10]:
        print(f"  warn: {w}")
    if len(warnings) > 10:
        print(f"  ... {len(warnings) - 10} more warnings")
    print(f"\n${cost:.4f}, {elapsed:.0f}s -> {out_dir / 'locations.json'}")


if __name__ == "__main__":
    sys.exit(main())
