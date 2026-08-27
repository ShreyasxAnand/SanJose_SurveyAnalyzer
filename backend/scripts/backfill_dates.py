"""Backfill a response-date column into an already-ingested dataset.

For datasets ingested before date columns existed (dataset 2's `stopdate`
was silently dropped at upload): register the column as date-typed metadata,
parse every stored upload's cells to ISO, bulk-insert RespondentAttribute
rows, default the period config to quarter bucketing, and rewrite the
exports. The labels/taxonomy/sub-theme artifacts are never touched — the
export re-join is keyed by response_key, which is stable — so no LLM stage
re-runs. Verify with `labels_unmatched_keys: 0` in the printed summary.

    cd backend
    python -m scripts.backfill_dates --dataset 2 --column stopdate --label Period

Idempotent: re-running re-parses and rewrites the same column's rows
(the upsert path is the same one select_columns uses). Refuses a column
already ingested as a question, and warns rather than proceeding if the
dataset already has OTHER metadata columns whose rows the per-upload
delete would drop (none of the legacy datasets do).
"""

import argparse
import json

from app import dates as dates_module
from app.db import SessionLocal, init_db
from app.ingest import (
    _ensure_uploads,
    _read_upload_frame,
    _upsert_upload_metadata,
    _write_exports,
)
from app.models import Dataset, MetadataColumn, QuestionColumn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=int, required=True)
    ap.add_argument("--column", required=True,
                    help="source column holding the response date, e.g. stopdate")
    ap.add_argument("--label", default=None,
                    help="display name for the derived period filter "
                         "(default: the column name)")
    ap.add_argument("--granularity", default="quarter",
                    choices=("quarter", "month", "year"),
                    help="default period bucketing stored on the dataset")
    args = ap.parse_args()

    init_db()  # applies the additive value_type/date_ranges_json columns
    db = SessionLocal()
    try:
        dataset = db.get(Dataset, args.dataset)
        if dataset is None:
            raise SystemExit(f"No dataset {args.dataset}")
        question_cols = {
            q.source_column
            for q in db.query(QuestionColumn)
            .filter(QuestionColumn.dataset_id == dataset.id)
            .all()
        }
        if args.column in question_cols:
            raise SystemExit(
                f"'{args.column}' is ingested as a question column — a column "
                "cannot be both a question and a date attribute"
            )
        other_meta = (
            db.query(MetadataColumn)
            .filter(
                MetadataColumn.dataset_id == dataset.id,
                MetadataColumn.source_column != args.column,
            )
            .all()
        )
        if other_meta:
            # _upsert_upload_metadata clears each upload's attribute rows for
            # ALL columns before rewriting the ones it is given — running it
            # with only the date column would drop the others' values. Those
            # datasets should re-run column selection instead.
            raise SystemExit(
                f"Dataset {dataset.id} already has metadata columns "
                f"({[m.source_column for m in other_meta]}) — re-run "
                "POST /datasets/{id}/columns with the full selection instead"
            )

        uploads = _ensure_uploads(db, dataset)
        frames = [(u, _read_upload_frame(u)) for u in uploads]
        for upload, df in frames:
            if args.column not in df.columns:
                raise SystemExit(
                    f"Column '{args.column}' missing from upload "
                    f"'{upload.stored_filename}'"
                )

        # Distinct parsed dates across every upload — the same cardinality
        # contract select_columns applies to a date column.
        raw: set[str] = set()
        for _u, df in frames:
            raw |= {v.strip() for v in df[args.column].tolist() if str(v).strip()}
        parsed = {iso for iso in (dates_module.parse_date(v) for v in raw) if iso}
        if not parsed:
            raise SystemExit(f"No cell of '{args.column}' parses as a date")
        unparseable = len(raw) - len(
            {v for v in raw if dates_module.parse_date(v)}
        )

        col = (
            db.query(MetadataColumn)
            .filter(
                MetadataColumn.dataset_id == dataset.id,
                MetadataColumn.source_column == args.column,
            )
            .one_or_none()
        )
        if col is None:
            col = MetadataColumn(
                dataset_id=dataset.id,
                source_column=args.column,
                label=(args.label or args.column).strip(),
                position=0,
                n_distinct=len(parsed),
                value_type="date",
            )
            db.add(col)
            db.flush()
        else:
            col.label = (args.label or col.label).strip()
            col.n_distinct = len(parsed)
            col.value_type = "date"

        for upload, df in frames:
            _upsert_upload_metadata(db, dataset, upload, df,
                                    {args.column: col})
        if not dataset.date_ranges_json:
            dataset.date_ranges_json = json.dumps(
                {"mode": "bucket", "granularity": args.granularity}
            )
        db.commit()

        from app.models import RespondentAttribute

        stored = (
            db.query(RespondentAttribute)
            .filter(
                RespondentAttribute.dataset_id == dataset.id,
                RespondentAttribute.metadata_column_id == col.id,
            )
            .count()
        )
        print(f"dataset {dataset.id}: stored {stored} dates "
              f"({len(parsed)} distinct, {unparseable} distinct raw values "
              f"unparseable) from {len(frames)} upload(s)")

        if dataset.status == "ingested":
            _write_exports(db, dataset)
            from app.db import EXPORTS_DIR

            manifest = json.loads(
                (EXPORTS_DIR / str(dataset.id) / "manifest.json")
                .read_text("utf-8")
            )
            print(f"exports rewritten — labels_unmatched_keys: "
                  f"{manifest['labels_unmatched_keys']}, date_ranges: "
                  f"{manifest['date_ranges']}")
        else:
            print("dataset not ingested — no exports to rewrite")
    finally:
        db.close()


if __name__ == "__main__":
    main()
