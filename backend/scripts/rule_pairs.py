"""Run the sameness-ruling pass for one dataset.

Nominates candidate pairs by name similarity (plus anything the analyst flags),
rules each with one batched model call, and writes the verdicts to
data/rulings/{dataset_id}/rulings.json. Plan build reads that store; nothing
here touches labels, assignments or counts.

Run it after a taxonomy or sub-theme build, not at ask time — rulings are
definitional, so they are computed once per taxonomy version and reused.

    python -m scripts.rule_pairs --dataset 2
    python -m scripts.rule_pairs --dataset 2 --dry-run
    python -m scripts.rule_pairs --dataset 2 --force-pair 5_001 6_014
    python -m scripts.rule_pairs --dataset 2 --stale-report

Re-running is cheap and safe: pairs whose definitions and prompt are unchanged
keep their existing keys and are skipped unless --re-rule is passed. Analyst
overrides survive every re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import rulings, subthemes
from app.llm import GeminiClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_categories(dataset_id: str) -> tuple[list[dict], dict[str, int], str]:
    """Category definitions, member counts and the dataset description."""
    path = REPO_ROOT / "data" / "summary" / str(dataset_id) / "summary.json"
    if not path.exists():
        raise SystemExit(f"No summary for dataset {dataset_id} at {path}. "
                         "Run `python -m scripts.summarize` first.")
    s = json.loads(path.read_text(encoding="utf-8"))
    items, counts = [], {}
    for q in s.get("questions", []):
        for e in q.get("entries", []):
            items.append({"id": e["label_id"], "question_id": q["question_id"],
                          "name": e["name"],
                          "description": e.get("description", ""),
                          "include": e.get("include") or [],
                          "exclude": e.get("exclude") or []})
            counts[e["label_id"]] = e.get("count", 0)
    return items, counts, s.get("dataset_description", "")


def _load_subthemes(dataset_id: str) -> list[dict]:
    """Sub-theme definitions across every question that has a sub-themes run."""
    root = REPO_ROOT / "data" / "subthemes" / str(dataset_id)
    if not root.exists():
        return []
    items = []
    for qdir in sorted(p for p in root.iterdir() if p.is_dir()):
        run = subthemes.latest_sub_run(dataset_id, qdir.name)
        if run is None:
            continue
        tax = json.loads((run / "sub_taxonomy.json").read_text(encoding="utf-8"))
        for cat in tax.get("categories", []):
            if cat.get("outcome") != "ok":
                continue
            for sl in cat.get("sub_labels", []):
                items.append({"id": sl["sub_label_id"], "label_id": cat["label_id"],
                              "question_id": qdir.name, "name": sl["name"],
                              "description": sl.get("description", ""),
                              "include": sl.get("include") or [],
                              "exclude": sl.get("exclude") or []})
    return items


def _samples(dataset_id: str, cat_items, sub_items) -> dict[str, list[str]]:
    """Sample member responses per id — the evidence the ruling rests on.

    Categories draw from the taxonomy's stored examples; sub-themes from their
    own. Both are already verbatim rows resolved at induction time, so nothing
    here can introduce a quote the corpus does not contain."""
    out: dict[str, list[str]] = {}
    tax_root = REPO_ROOT / "data" / "taxonomy" / str(dataset_id)
    if tax_root.exists():
        for qdir in sorted(p for p in tax_root.iterdir() if p.is_dir()):
            runs = sorted(r for r in qdir.iterdir()
                          if (r / "candidate_taxonomy.json").exists())
            if not runs:
                continue
            tax = json.loads(
                (runs[-1] / "candidate_taxonomy.json").read_text(encoding="utf-8"))
            for lab in tax.get("labels", []):
                out[lab["label_id"]] = [e["text"] for e in lab.get("examples", [])]
    root = REPO_ROOT / "data" / "subthemes" / str(dataset_id)
    if root.exists():
        for qdir in sorted(p for p in root.iterdir() if p.is_dir()):
            run = subthemes.latest_sub_run(dataset_id, qdir.name)
            if run is None:
                continue
            tax = json.loads((run / "sub_taxonomy.json").read_text(encoding="utf-8"))
            for cat in tax.get("categories", []):
                for sl in cat.get("sub_labels", []):
                    out[sl["sub_label_id"]] = [e["text"]
                                               for e in sl.get("examples", [])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="nominate and print, make no model calls")
    ap.add_argument("--re-rule", action="store_true",
                    help="re-rule pairs that already have a current verdict")
    ap.add_argument("--stale-report", action="store_true",
                    help="list rulings whose membership moved a lot, and exit")
    ap.add_argument("--force-pair", nargs=2, action="append", metavar=("ID_A", "ID_B"),
                    help="rule this pair regardless of similarity (repeatable)")
    ap.add_argument("--min-sim", type=float, default=rulings.NOMINATE_MIN)
    ap.add_argument("--workers", type=int, default=rulings.DEFAULT_WORKERS)
    ap.add_argument("--model", default=None,
                    help="override the ruling model (default: pipeline default)")
    ap.add_argument("--min-interval", type=float, default=0.0)
    args = ap.parse_args()

    ds = str(args.dataset)
    cat_items, counts, description = _load_categories(ds)
    sub_items = _load_subthemes(ds)
    store = rulings.load_store(ds)

    if args.stale_report:
        defs = {i["id"]: i for i in cat_items + sub_items}
        flagged = rulings.RulingIndex(store, defs, description).stale(counts)
        if not flagged:
            print("no rulings flagged stale")
            return 0
        print(f"{len(flagged)} ruling(s) flagged — membership moved past "
              f"{rulings.MEMBER_DRIFT_FLAG:.0%}. Advisory only; re-rule with "
              f"--force-pair if you want a fresh verdict.")
        for f in flagged:
            print(f"  {f['ids']}  {f['label_id']}: {f['was']} -> {f['now']}")
        return 0

    forced = [tuple(p) for p in (args.force_pair or [])]
    nominated = (
        rulings.nominate(cat_items, "category", forced, args.min_sim)
        + rulings.nominate(cat_items, "category_sameq", forced, args.min_sim)
        + rulings.nominate(sub_items, "subtheme_xcat", forced, args.min_sim)
        + rulings.nominate(sub_items, "subtheme_xq", forced, args.min_sim)
    )
    print(f"dataset {ds}: {len(cat_items)} categories, {len(sub_items)} sub-themes "
          f"-> {len(nominated)} nominated pair(s)")

    if not args.re_rule:
        defs = {i["id"]: i for i in cat_items + sub_items}
        idx = rulings.RulingIndex(store, defs, description)
        before = len(nominated)
        nominated = [p for p in nominated
                     if idx.lookup(p["a"]["id"], p["b"]["id"]) is None]
        if before != len(nominated):
            print(f"  {before - len(nominated)} already ruled and current, skipped "
                  f"(--re-rule to redo)")

    by_species: dict[str, int] = {}
    for p in nominated:
        by_species[p["species"]] = by_species.get(p["species"], 0) + 1
    for sp, n in sorted(by_species.items()):
        print(f"  {sp}: {n}")

    if not nominated:
        print("nothing to rule")
        return 0

    samples = _samples(ds, cat_items, sub_items)
    if args.dry_run:
        print("\n--dry-run: pairs that would be ruled\n")
        for p in nominated:
            print(f"  [{p['name_sim']:.2f}] {p['a']['id']} {p['a']['name']!r}")
            print(f"{'':>9}vs {p['b']['id']} {p['b']['name']!r}")
        est = -(-len(nominated) // rulings.PAIRS_PER_CALL)
        print(f"\n{len(nominated)} pairs -> {est} model call(s)")
        return 0

    client = GeminiClient(model=args.model, min_interval_s=args.min_interval)
    print(f"model: {client.model_id} (temperature=0)")
    rows, report = rulings.rule_pairs(client, nominated, samples, counts,
                                      description, workers=args.workers)
    merged = rulings.merge_rows(store, rows)
    path = rulings.save_store(ds, merged)

    print(f"\nruled {report['pairs_ruled']}/{report['pairs_nominated']}: "
          f"{report['same_idea']} same_idea, {report['distinct']} distinct, "
          f"{report['pairs_unruled']} unruled (unruled = unfused)")
    if report["failed_batches"]:
        print(f"  {len(report['failed_batches'])} failed batch(es) — those pairs "
              f"stay unruled, so they stay unfused")
    for w in report["warnings"][:10]:
        print(f"  warning: {w}")
    print(f"wrote {path.relative_to(REPO_ROOT)} "
          f"({len(merged['rows'])} rows, {len(merged['history'])} archived)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
