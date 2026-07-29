"""Phase 4: regenerate the taxonomy summary artifact.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.summarize
    python -m scripts.summarize --dataset 1

Deterministic and free — no model calls. Reads the latest labels run and its
paired taxonomy for every question, computes real counts, and writes:

    data/summary/{dataset_id}/summary.json   <- machine-readable
    data/summary/{dataset_id}/summary.md     <- the prompt-facing rendering

`scripts.ask` rebuilds this on every question, so running this by hand is
only needed when you want to look at the artifact itself.
"""
from __future__ import annotations

import argparse
import sys

from app import induction, summary


def discover_dataset_ids() -> list[str]:
    if not summary.LABELS_DIR.is_dir():
        return []
    return sorted(d.name for d in summary.LABELS_DIR.iterdir() if d.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="dataset id (default: every dataset with labels)")
    ap.add_argument("--description", default=None,
                    help="default: the export manifest's dataset_description")
    args = ap.parse_args()

    dataset_ids = [args.dataset] if args.dataset else discover_dataset_ids()
    if not dataset_ids:
        raise SystemExit(f"No labeled datasets under {summary.LABELS_DIR}. "
                         "Run scripts.label first.")

    for ds in dataset_ids:
        description = induction.resolve_description(
            args.description,
            induction.DATA_DIR / "exports" / ds / "responses.parquet")
        s = summary.build_summary(ds, description)
        out_dir = summary.write_summary(s)
        n_labels = sum(len(q["entries"]) for q in s["questions"])
        print(f"dataset {ds}: {len(s['questions'])} questions, {n_labels} labels, "
              f"{len(s['lexicon_concepts'])} lexicon concepts -> {out_dir}")
        for q in s["questions"]:
            flag = ""
            if q["unknown_assignment_ids"]:
                flag = f"  <-- WARN {sum(q['unknown_assignment_ids'].values())} " \
                       f"assignments reference unknown label ids"
            if q["taxonomy_resolved_via"] == "latest_taxonomy_fallback":
                flag += "  <-- WARN taxonomy paired by fallback, check runs"
            print(f'  q{q["question_id"]}: {q["n_responses"]} responses, '
                  f'{len(q["entries"])} labels, {q["n_uncategorized"]} uncategorized '
                  f'(labels {q["labels_run"]} / taxonomy {q["taxonomy_run"]}){flag}')


if __name__ == "__main__":
    sys.exit(main())
