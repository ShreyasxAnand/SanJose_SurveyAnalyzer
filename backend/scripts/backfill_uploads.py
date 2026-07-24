"""Backfill Upload + RowHash rows for datasets ingested before those tables
existed, so future uploads can be duplicate-checked against them.

For each dataset that has responses but no Upload row: parse its stored
original file, create Upload #1 (row_offset=0, every row new), bulk-insert
one RowHash per raw row, and stamp responses.upload_id. Idempotent — datasets
that already have an Upload row are skipped.

    cd backend
    python -m scripts.backfill_uploads

One-time prerequisite on a live DB (fresh DBs get everything from
create_all()):

    ALTER TABLE responses ADD COLUMN upload_id INTEGER REFERENCES uploads(id);
"""

from app.db import Base, REPO_ROOT, SessionLocal, engine
from app.ingest import _read_dataframe
from app.models import ColumnFingerprint, Dataset, Response, RowHash, Upload
from app.rowhash import fingerprint_columns, hash_dataframe


def main() -> None:
    Base.metadata.create_all(bind=engine)  # creates uploads/row_hashes if absent
    db = SessionLocal()
    try:
        for dataset in db.query(Dataset).order_by(Dataset.id).all():
            existing = (
                db.query(Upload).filter(Upload.dataset_id == dataset.id).count()
            )
            if existing:
                print(f"dataset {dataset.id} ({dataset.name}): already has "
                      f"{existing} upload(s), skipping")
                continue
            if not dataset.original_path:
                print(f"dataset {dataset.id} ({dataset.name}): no stored file, skipping")
                continue
            path = REPO_ROOT / dataset.original_path
            if not path.exists():
                print(f"dataset {dataset.id} ({dataset.name}): stored file missing "
                      f"({path}), skipping")
                continue

            df = _read_dataframe(path)
            hashes = hash_dataframe(df)
            upload = Upload(
                dataset_id=dataset.id,
                stored_filename=dataset.original_filename,
                stored_path=dataset.original_path,
                sheet_name=dataset.sheet_name,
                row_offset=0,
                row_count=len(df),
                new_row_count=len(df),
                duplicate_row_count=0,
                uploaded_at=dataset.uploaded_at,
            )
            db.add(upload)
            db.flush()  # assign upload.id
            db.bulk_insert_mappings(
                RowHash,
                [
                    {
                        "dataset_id": dataset.id,
                        "upload_id": upload.id,
                        "row_index": i,
                        "row_hash": h,
                        "is_duplicate": False,
                    }
                    for i, h in enumerate(hashes)
                ],
            )
            n_responses = (
                db.query(Response)
                .filter(Response.dataset_id == dataset.id)
                .update({"upload_id": upload.id}, synchronize_session=False)
            )
            db.commit()
            print(f"dataset {dataset.id} ({dataset.name}): upload #{upload.id}, "
                  f"{len(hashes)} row hashes, {n_responses} responses stamped")

        # Second idempotent pass: column fingerprints for any upload that
        # predates the column_fingerprints table (2026-08-03). Independent of
        # the Upload backfill above so it also covers uploads created between
        # the two features.
        for upload in db.query(Upload).order_by(Upload.id).all():
            existing = (
                db.query(ColumnFingerprint)
                .filter(ColumnFingerprint.upload_id == upload.id)
                .count()
            )
            if existing:
                continue
            path = REPO_ROOT / upload.stored_path
            if not path.exists():
                print(f"upload {upload.id} ({upload.stored_filename}): stored "
                      f"file missing ({path}), skipping fingerprints")
                continue
            fingerprints = fingerprint_columns(_read_dataframe(path))
            db.bulk_insert_mappings(
                ColumnFingerprint,
                [
                    {
                        "dataset_id": upload.dataset_id,
                        "upload_id": upload.id,
                        "column_name": col,
                        "fingerprint": fp,
                    }
                    for col, fp in fingerprints.items()
                ],
            )
            db.commit()
            print(f"upload {upload.id} ({upload.stored_filename}): "
                  f"{len(fingerprints)} column fingerprints")
    finally:
        db.close()


if __name__ == "__main__":
    main()
