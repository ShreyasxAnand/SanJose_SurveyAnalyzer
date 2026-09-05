"""Post-process an existing sub-themes run with the automated cleanup pass.

For runs produced before the inline review existed (or with --no-review).
One LLM call per category catches same-idea duplicate sub-codes and
category restatements; edits are pure id rewrites over the already-coded
assignments, so nothing is relabelled. Writes a NEW `{utc}_review` run dir —
the source run is never modified, matching the labels `_review` convention.

From backend/:

    python -m scripts.subthemes_review --dataset 2 --all
    python -m scripts.subthemes_review --dataset 2 --question 9
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from app import induction, llm, subthemes, summary
from app.llm import GeminiClient

SUBTHEMES_DIR = induction.DATA_DIR / "subthemes"


def review_one(args, client: GeminiClient, question_id: str) -> dict | None:
    qdir = SUBTHEMES_DIR / args.dataset / question_id
    run = summary.latest_run_dir(qdir, "sub_taxonomy.json")
    if run is None:
        print(f"q{question_id}: no sub-themes run — skipped")
        return None
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    tax = json.loads((run / "sub_taxonomy.json").read_text(encoding="utf-8"))
    if tax.get("status") == "reviewed_auto" and not args.re_review:
        print(f"q{question_id}: {run.name} already reviewed — skipped")
        return None
    assignments = json.loads((run / "sub_assignments.json").read_text(encoding="utf-8"))
    description = manifest.get("dataset_description") or ""
    question_text = tax.get("question_text", "")

    # member texts for evidence-based pair rulings — best-effort: reviews
    # still run (names + overlap only) when the corpus isn't loadable here
    texts: dict[str, str] = {}
    try:
        from app import ask_service
        parquet = ask_service.dataset_parquet(args.dataset)
        rows, _meta, _f = induction.load_question(parquet, question_id)
        texts = {r.response_key: r.text for r in rows}
    except Exception as exc:
        print(f"  (no member examples — corpus not loadable: {exc})")

    print(f"q{question_id}: reviewing {run.name} "
          f"({sum(1 for c in tax['categories'] if c['outcome'] == 'ok')} categories)")
    logs: dict[str, dict] = {}
    wanted = set(args.categories.split(",")) if args.categories else None
    for cat in tax["categories"]:
        if cat["outcome"] != "ok" or not cat["sub_labels"]:
            continue
        if wanted is not None and cat["label_id"] not in wanted:
            continue
        lid = cat["label_id"]
        counts: dict[str, int] = {s["sub_label_id"]: 0 for s in cat["sub_labels"]}
        members_of: dict[str, set[str]] = {}
        examples_of: dict[str, list[str]] = {}
        for rec in assignments:
            for sid in rec.get("subs", {}).get(lid, []):
                if sid in counts:
                    counts[sid] += 1
                    members_of.setdefault(sid, set()).add(rec["response_key"])
                    ex = examples_of.setdefault(sid, [])
                    if len(ex) < 3 and rec["response_key"] in texts:
                        ex.append(texts[rec["response_key"]])
        forced = [tuple(p.split(":")) for p in (args.pairs or "").split(",")
                  if ":" in p]
        forced = [(a, b) for a, b in forced
                  if a.startswith(lid) and b.startswith(lid)]
        edits, failure = subthemes.review_subthemes(
            client, question_text, cat, cat["sub_labels"], counts, description,
            members=members_of, forced_pairs=forced or None,
            examples=examples_of or None)
        if failure:
            logs[lid] = {"failure": failure}
            print(f"  {lid}: review call failed — left untouched")
            continue
        kept, id_map, log = subthemes.apply_subtheme_review(
            cat["sub_labels"], edits, counts)
        logs[lid] = log
        if id_map:
            for rec in assignments:
                if lid in rec.get("subs", {}):
                    rec["subs"][lid] = subthemes.remap_sub_ids(rec["subs"][lid], id_map)
        if len(kept) < subthemes.MIN_SUBLABELS:
            cat["outcome"] = "does_not_decompose_after_review"
            cat["sub_labels"] = []
            for rec in assignments:
                rec.get("subs", {}).pop(lid, None)
                rec.get("sub_fit", {}).pop(lid, None)
            print(f"  {lid}: does not decompose after review — sub-codes dropped")
        else:
            cat["sub_labels"] = kept
            if log["merged"] or log["restated"] or log.get("renamed"):
                print(f"  {lid}: {len(log['merged'])} merge(s), "
                      f"{len(log['restated'])} restatement(s), "
                      f"{len(log.get('renamed', []))} rename(s)")

    run_id = f"{induction.utc_now()}_review"
    out_dir = qdir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tax["status"] = "reviewed_auto"
    tax["source_run"] = run.name
    (out_dir / "sub_taxonomy.json").write_text(
        json.dumps(tax, indent=2, ensure_ascii=False), encoding="utf-8")
    # drop records that no longer carry anything (subs emptied by a dropped category)
    assignments = [r for r in assignments if r.get("subs") or r.get("uncoded")]
    (out_dir / "sub_assignments.json").write_text(
        json.dumps(assignments, indent=2, ensure_ascii=False), encoding="utf-8")
    out_manifest = {
        "run_id": run_id,
        "created_utc": induction.utc_now(),
        "tool": "scripts.subthemes_review",
        "schema_version": subthemes.SCHEMA_VERSION,
        "model_id": client.model_id,
        "source_run": run.name,
        "source_manifest": manifest.get("run_id"),
        "review_logs": logs,
        "usage": {
            "calls": client.usage.calls,
            "input_tokens": client.usage.input_tokens,
            "output_tokens": client.usage.output_tokens,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_edits = sum(len(l.get("merged", [])) + len(l.get("restated", []))
                  for l in logs.values())
    print(f"  {n_edits} edit(s) applied -> {out_dir}")
    return out_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--question")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--categories",
                    help="comma-separated label_ids to restrict the review to")
    ap.add_argument("--pairs",
                    help="analyst-flagged sub-code pairs to force a ruling on, "
                         "as a:b comma-separated (e.g. 5_001s17:5_001s05)")
    ap.add_argument("--re-review", action="store_true",
                    help="review again even when the latest run is already "
                         "marked reviewed_auto (e.g. after strengthening the "
                         "review prompt)")
    ap.add_argument("--min-interval", type=float, default=0.1)
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

    base = SUBTHEMES_DIR / args.dataset
    if args.all:
        questions = sorted(d.name for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
    else:
        questions = [args.question]
    if not questions:
        raise SystemExit(f"No sub-themes runs under {base}")
    client = GeminiClient(model=args.model, min_interval_s=args.min_interval)
    t0 = time.time()
    for q in questions:
        review_one(args, client, q)
    cost = client.usage.cost_usd(args.price_in, args.price_out)
    print(f"\n${cost:.3f}, {time.time() - t0:.0f}s, {client.usage.calls} calls")


if __name__ == "__main__":
    sys.exit(main())
