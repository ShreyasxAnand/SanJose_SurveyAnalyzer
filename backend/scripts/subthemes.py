"""Induce and assign sub-themes within every large category of a question.

From backend/, inside the SurveyAnalyzer conda env:

    python -m scripts.subthemes --all --dry-run
    python -m scripts.subthemes --question 5
    python -m scripts.subthemes --all

Reads the most recent labels run (and its paired taxonomy) per question.
Output: data/subthemes/{dataset_id}/{question_id}/{run_id}/
        sub_taxonomy.json    <- sub-codes per large category (reviewable)
        sub_assignments.json <- one record per response that got sub-coded
        manifest.json        <- audit record + per-category reports + usage
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app import induction, llm, subthemes, summary
from app.induction import ResponseRow
from app.llm import GeminiClient

SUBTHEMES_DIR = induction.DATA_DIR / "subthemes"


def member_rows(assignments: list[dict], label_id: str,
                texts: dict[str, str]) -> list[ResponseRow]:
    """A category's member rows, in assignment order. Rows whose text is
    missing from the parquet (should not happen) are skipped, not invented."""
    out = []
    for a in assignments:
        if label_id in a.get("label_ids", []):
            t = texts.get(a["response_key"])
            if t:
                out.append(ResponseRow(response_key=a["response_key"], text=t))
    return out


def estimate_category(cat: dict, rows: list[ResponseRow], question_text: str,
                      description: str, batch_size: int) -> dict:
    """Call/token estimate for one category, from real prompt sizes — the
    same approach as scripts.label's dry run, not a hand-waved constant."""
    n = len(rows)
    n_unique = len({r.text for r in rows})
    n_map = -(-n // subthemes.SUB_CHUNK_SIZE)
    sample = rows[: subthemes.SUB_CHUNK_SIZE]
    map_sys, map_usr, _ = subthemes.build_submap_prompts(
        question_text, cat, sample, description)
    map_in = n_map * (len(map_sys) + len(map_usr)) // 4
    map_out = n_map * 1200          # ~7 sub-themes with include/exclude/evidence
    # candidates ~7/chunk, exact-merge keeps ~60%; dedup output is names+ids only
    n_dedup = max(1, -(-int(n_map * 7 * 0.6) // 40)) + (1 if n_map * 7 * 0.6 > 40 else 0)
    dedup_in = n_dedup * 1500
    dedup_out = n_dedup * 400
    n_batches = -(-n_unique // batch_size)
    fake_subs = [{"sub_label_id": f"{cat['label_id']}s{i:02d}",
                  "name": "placeholder sub-theme name",
                  "description": "placeholder description of the sub-theme."}
                 for i in range(1, 8)]
    lab_sys, lab_usr = subthemes.build_sublabel_prompts(
        question_text, cat, fake_subs, [(r.response_key, r.text) for r in sample[:batch_size]],
        description)
    lab_in = n_batches * (len(lab_sys) + len(lab_usr)) // 4
    lab_out = n_unique * 12         # {"n":..,"l":["5_002s01"],"f":3}
    return {
        "label_id": cat["label_id"], "name": cat["name"],
        "n_members": n, "n_unique": n_unique,
        "calls": n_map + n_dedup + n_batches,
        "est_in": map_in + dedup_in + lab_in,
        "est_out": map_out + dedup_out + lab_out,
    }


def subthemes_one(args, question: str) -> dict:
    parquet = induction.discover_parquet(args.parquet)
    rows, meta, _filtered = induction.load_question(parquet, question)
    dataset_id, question_id = meta["dataset_id"], meta["question_id"]
    question_text = meta["question_text"]

    labels_run = summary.latest_run_dir(
        summary.LABELS_DIR / dataset_id / question_id, "assignments.json")
    if labels_run is None:
        raise SystemExit(f"No labels run for dataset {dataset_id} question "
                         f"{question_id}. Run scripts.label first.")
    tax_path, tax_how = summary.resolve_taxonomy_path(labels_run, dataset_id, question_id)
    taxonomy = json.loads(tax_path.read_text(encoding="utf-8"))
    assignments = json.loads((labels_run / "assignments.json").read_text(encoding="utf-8"))

    counts: dict[str, int] = {}
    for a in assignments:
        for lid in a.get("label_ids", []):
            counts[lid] = counts.get(lid, 0) + 1
    eligible = subthemes.eligible_categories(taxonomy, counts, args.min_n)
    if args.categories:
        wanted = set(args.categories.split(","))
        eligible = [c for c in eligible if c["label_id"] in wanted]
    texts = {r.response_key: r.text for r in rows}

    # Carry-forward: categories a prior run already sub-coded (any recorded
    # outcome) are copied into the new run instead of re-paid-for. A category
    # is only re-run when --redo names it. This is what makes lowering
    # --min-n an incremental cost: only the newly-eligible band runs.
    prior_run = subthemes.latest_sub_run(dataset_id, question_id)
    carried: dict[str, dict] = {}
    prior_assignments: list[dict] = []
    if prior_run is not None and not args.no_carry:
        prior_tax = json.loads(
            (prior_run / "sub_taxonomy.json").read_text(encoding="utf-8"))
        redo = set((args.redo or "").split(",")) if args.redo else set()
        carried = {c["label_id"]: c for c in prior_tax.get("categories", [])
                   if c["label_id"] not in redo}
        prior_assignments = json.loads(
            (prior_run / "sub_assignments.json").read_text(encoding="utf-8"))
        eligible = [c for c in eligible if c["label_id"] not in carried]

    print(f"\nq{question_id}: \"{question_text}\" — {len(eligible)} categories "
          f"to run >= {args.min_n} members (of {len(taxonomy['labels'])} total"
          + (f"; {len(carried)} carried from {prior_run.name}" if carried else "")
          + ")")
    print(f"  labels run: {labels_run.name} | taxonomy via {tax_how}")

    if args.dry_run:
        total_calls = total_in = total_out = 0
        for cat in eligible:
            rows_c = member_rows(assignments, cat["label_id"], texts)
            e = estimate_category(cat, rows_c, question_text, args.description,
                                  args.batch_size)
            total_calls += e["calls"]; total_in += e["est_in"]; total_out += e["est_out"]
            print(f"    {cat['label_id']}  n={e['n_members']:<5} ({e['n_unique']} unique) "
                  f"~{e['calls']} calls  — {cat['name'][:55]}")
        cost = total_in / 1e6 * args.price_in + total_out / 1e6 * args.price_out
        print(f"  DRY RUN — {total_calls} calls, ~{total_in:,} in / ~{total_out:,} out, "
              f"~${cost:.2f}")
        return {"est_cost_usd": cost, "est_calls": total_calls}

    if not eligible:
        print("  nothing new to sub-code — prior run stays latest")
        return {}

    client = GeminiClient(model=args.model, max_output_tokens=args.max_output_tokens,
                          min_interval_s=args.min_interval)
    t0 = time.time()
    out_categories: list[dict] = []
    reports: list[dict] = []
    merged: dict[str, dict] = {}    # response_key -> sub-assignment record

    # seed with the carried categories' assignment records, restricted to the
    # carried lids so a --redo category's stale coding cannot leak through
    if carried:
        carried_lids = set(carried)
        for rec in prior_assignments:
            subs = {lid: v for lid, v in (rec.get("subs") or {}).items()
                    if lid in carried_lids}
            fits = {lid: v for lid, v in (rec.get("sub_fit") or {}).items()
                    if lid in carried_lids}
            unc = {lid: v for lid, v in (rec.get("uncoded") or {}).items()
                   if lid in carried_lids}
            if subs or unc:
                merged[rec["response_key"]] = {
                    "response_key": rec["response_key"],
                    "respondent_key": rec.get("respondent_key"),
                    "subs": subs, "sub_fit": fits, "uncoded": unc}

    for ci, cat in enumerate(eligible, start=1):
        lid = cat["label_id"]
        rows_c = member_rows(assignments, lid, texts)
        print(f"  [{ci}/{len(eligible)}] {lid} {cat['name'][:55]} (n={len(rows_c)}) …",
              flush=True)
        sub_labels, ind_report = subthemes.induce_subthemes(
            client, question_text, cat, rows_c,
            dataset_description=args.description, workers=args.workers)
        reports.append({"induction": ind_report})
        if not sub_labels:
            print(f"      {ind_report['outcome']} — no sub-codes kept")
            out_categories.append({"label_id": lid, "name": cat["name"],
                                   "n_members": len(rows_c),
                                   "outcome": ind_report["outcome"],
                                   "sub_labels": []})
            continue
        pairs = [(r.response_key, r.text) for r in rows_c]
        sub_assignments, lab_report = subthemes.run_sublabeling(
            pairs, question_text, cat, sub_labels, client,
            batch_size=args.batch_size, dataset_description=args.description,
            workers=args.workers)
        reports[-1]["sublabeling"] = lab_report

        # Automated cleanup: one call catching same-idea duplicates and
        # category restatements, applied as pure id rewrites (see
        # subthemes.py's SUBREVIEW note for why this layer auto-applies).
        if not args.no_review:
            members_of: dict[str, set[str]] = {}
            text_of = {r.response_key: r.text for r in rows_c}
            examples_of: dict[str, list[str]] = {}
            for a in sub_assignments:
                for sid in a["sub_label_ids"]:
                    members_of.setdefault(sid, set()).add(a["response_key"])
                    ex = examples_of.setdefault(sid, [])
                    if len(ex) < 3 and a["response_key"] in text_of:
                        ex.append(text_of[a["response_key"]])
            edits, rev_failure = subthemes.review_subthemes(
                client, question_text, cat, sub_labels,
                lab_report["sub_label_counts"], args.description,
                members=members_of, examples=examples_of)
            if rev_failure:
                reports[-1]["review"] = {"failure": rev_failure}
            else:
                sub_labels, id_map, rev_log = subthemes.apply_subtheme_review(
                    sub_labels, edits, lab_report["sub_label_counts"])
                reports[-1]["review"] = rev_log
                if id_map:
                    for a in sub_assignments:
                        a["sub_label_ids"] = subthemes.remap_sub_ids(
                            a["sub_label_ids"], id_map)
                    counts_after = {s["sub_label_id"]: 0 for s in sub_labels}
                    for a in sub_assignments:
                        for sid in a["sub_label_ids"]:
                            counts_after[sid] += 1
                    lab_report["sub_label_counts"] = counts_after
                    lab_report["responses_with_subthemes"] = sum(
                        1 for a in sub_assignments if a["sub_label_ids"])
                    lab_report["responses_generic"] = sum(
                        1 for a in sub_assignments if not a["sub_label_ids"]
                        and not a.get("not_returned") and not a.get("batch_failed"))
                    print(f"      review: {len(rev_log['merged'])} merge(s), "
                          f"{len(rev_log['restated'])} restatement(s) folded to generic")
        if len(sub_labels) < subthemes.MIN_SUBLABELS:
            print("      does_not_decompose after review — sub-codes dropped")
            out_categories.append({"label_id": lid, "name": cat["name"],
                                   "n_members": len(rows_c),
                                   "outcome": "does_not_decompose_after_review",
                                   "sub_labels": []})
            continue
        for a in sub_assignments:
            rec = merged.setdefault(a["response_key"], {
                "response_key": a["response_key"],
                "respondent_key": subthemes.respondent_key(a["response_key"]),
                "subs": {}, "sub_fit": {}, "uncoded": {}})
            if a.get("not_returned") or a.get("batch_failed"):
                rec["uncoded"][lid] = ("not_returned" if a.get("not_returned")
                                       else "batch_failed")
            else:
                rec["subs"][lid] = a["sub_label_ids"]
                if a.get("fit") is not None:
                    rec["sub_fit"][lid] = a["fit"]
        top = sorted(lab_report["sub_label_counts"].items(), key=lambda kv: -kv[1])[:3]
        print(f"      {len(sub_labels)} sub-codes | "
              f"{lab_report['responses_with_subthemes']} specific / "
              f"{lab_report['responses_generic']} generic of "
              f"{lab_report['responses_coded']} coded | top: "
              + ", ".join(f"{sid.rpartition('s')[2]}={n}" for sid, n in top))
        out_categories.append({"label_id": lid, "name": cat["name"],
                               "n_members": len(rows_c), "outcome": "ok",
                               "sub_labels": sub_labels})

    elapsed = time.time() - t0
    cost = client.usage.cost_usd(args.price_in, args.price_out)
    run_id = f"{induction.utc_now()}_{subthemes.prompt_hash(args.description)[:8]}"
    out_dir = SUBTHEMES_DIR / dataset_id / question_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_taxonomy = {
        "schema_version": subthemes.SCHEMA_VERSION,
        # inline auto-review already ran unless --no-review; the post-processor
        # (scripts.subthemes_review) keys off this to skip self-cleaned runs
        "status": "candidate_for_review" if args.no_review else "reviewed_auto",
        "dataset_id": dataset_id,
        "question_id": question_id,
        "question_text": question_text,
        "min_n": args.min_n,
        "labels_run": labels_run.name,
        "taxonomy_run": tax_path.parent.name,
        "review_instructions": (
            "Hand-edit like candidate_taxonomy.json: rename sub-labels, tighten "
            "descriptions, delete by removing the object. sub_label_id values "
            "freeze at approval — sub-assignments key off them. A category with "
            "outcome != 'ok' kept no sub-codes; that is recorded, not an error."
        ),
        "carried_from": prior_run.name if carried else None,
        "categories": sorted(list(carried.values()) + out_categories,
                             key=lambda c: -c.get("n_members", 0)),
    }
    (out_dir / "sub_taxonomy.json").write_text(
        json.dumps(sub_taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sub_assignments.json").write_text(
        json.dumps(list(merged.values()), indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_utc": induction.utc_now(),
        "tool": "scripts.subthemes",
        "schema_version": subthemes.SCHEMA_VERSION,
        "model_id": client.model_id,
        "temperature": 0.0,
        "prompt_sha256_16": subthemes.prompt_hash(args.description),
        "dataset_description": args.description,
        "min_n": args.min_n,
        "batch_size": args.batch_size,
        "labels_run": labels_run.name,
        "taxonomy_run": tax_path.parent.name,
        "taxonomy_resolved_via": tax_how,
        "carried_from": prior_run.name if carried else None,
        "categories_carried": len(carried),
        "source": meta,
        "category_reports": reports,
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
    n_ok = sum(1 for c in out_categories if c["outcome"] == "ok")
    print(f"  {n_ok}/{len(eligible)} categories sub-coded, "
          f"{len(merged)} responses carry sub-codes")
    print(f"  ${cost:.3f}, {elapsed:.0f}s -> {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question")
    ap.add_argument("--all", action="store_true",
                    help="every question with a labels run")
    ap.add_argument("--parquet")
    ap.add_argument("--categories",
                    help="comma-separated label_ids to restrict to (default: all eligible)")
    ap.add_argument("--redo",
                    help="comma-separated label_ids to re-run even though a "
                         "prior run already sub-coded them")
    ap.add_argument("--no-carry", action="store_true",
                    help="ignore prior runs entirely and re-run every eligible category")
    ap.add_argument("--min-n", type=int, default=subthemes.DEFAULT_MIN_N)
    ap.add_argument("--batch-size", type=int, default=subthemes.DEFAULT_BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=subthemes.DEFAULT_WORKERS)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--min-interval", type=float, default=0.1)
    ap.add_argument("--description", default=None,
                    help="default: the export manifest's dataset_description")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the automated duplicate/restatement cleanup pass")
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
        m = subthemes_one(args, q)
        total += (m.get("usage") or {}).get("est_cost_usd", m.get("est_cost_usd", 0.0))
    if len(questions) > 1:
        kind = "estimated " if args.dry_run else ""
        print(f"\n{kind}total: ${total:.2f} across {len(questions)} questions")


if __name__ == "__main__":
    sys.exit(main())
