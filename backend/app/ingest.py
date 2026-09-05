import datetime as dt
import io
import json
import shutil
import uuid
from pathlib import Path
from typing import IO

import ftfy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import dates as dates_module
from app import purge, rowhash
from app.auth import require_admin
from app.db import DATA_DIR, EXPORTS_DIR, UPLOADS_DIR, get_db
from app.models import (
    ColumnFingerprint,
    Dataset,
    MetadataColumn,
    QuestionColumn,
    RespondentAttribute,
    Response,
    RowHash,
    Upload,
)
from app.nonanswer import is_nonanswer_text
from app.schemas import (
    AppendRequest,
    AppendResponse,
    ColumnMatch,
    ColumnPreview,
    DatasetHistoryOut,
    DatasetMatch,
    DatasetMetadataPatch,
    DatasetOut,
    DeletionPreview,
    DeletionPreviewRoot,
    DeletionResult,
    DuplicateCheck,
    ExportInfo,
    MetadataColumnOut,
    MetadataValueCount,
    QuestionColumnOut,
    SelectColumnsRequest,
    UndeletedRoot,
    UploadHistoryEntry,
    UploadResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
REPO_ROOT = DATA_DIR.parent

# Export schema v2 — aligned with labeling SCHEMA_VERSION 2 (which dropped
# sentiment and added fit/time_context/actionability/event_occurred).
# `child_category_ids` became `label_ids` to match assignments.json exactly.
# The label block is None (not []) on rows no labeling run has covered, so
# "never labeled" and "labeled, no category" stay distinguishable. Exports
# are wipe-and-rewrite artifacts and load_question inspects the actual
# schema, so v1 exports keep working until regenerated.
# v3 added response_date: the respondent's ISO survey-completion date from
# the dataset's date-typed metadata column (the first, by position, if
# several), None when the dataset has none. It joins the fixed schema —
# unlike per-dataset demographics — because its name and meaning are the
# same for every dataset, and "% by wave" analysis straight off the flat
# file was the point of ingesting dates at all.
EXPORT_SCHEMA_VERSION = 3
LIST_COLUMNS = ("label_ids", "locations", "time_context")
# Demographics sidecar — see the comment at its write site in _write_exports
# for why this is a separate file rather than columns on the response schema.
RESPONDENT_PARQUET_SCHEMA = pa.schema(
    [
        ("respondent_key", pa.string()),
        ("source_row_index", pa.int64()),
        ("field", pa.string()),
        ("value", pa.string()),
    ]
)

RESPONSE_PARQUET_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.int64()),
        ("question_id", pa.int64()),
        ("question_label", pa.string()),
        ("source_column", pa.string()),
        ("source_row_index", pa.int64()),
        ("response_key", pa.string()),
        ("respondent_key", pa.string()),
        ("respondent_id", pa.string()),
        ("response_text", pa.string()),
        ("raw_text_original", pa.string()),
        ("was_encoding_repaired", pa.bool_()),
        ("is_nonanswer", pa.bool_()),
        # labeling v2 fields — null until a labels run exists for the row
        ("label_ids", pa.list_(pa.string())),
        ("uncategorized", pa.bool_()),
        ("fit", pa.int64()),
        ("locations", pa.list_(pa.string())),
        ("time_context", pa.list_(pa.string())),
        ("actionability", pa.string()),
        ("event_occurred", pa.bool_()),
        # v3 — ISO respondent survey-completion date, None when untracked
        ("response_date", pa.string()),
    ]
)


def _parse_dataframe(buffer: IO[bytes], suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        df = pd.read_csv(buffer, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(buffer, dtype=str, keep_default_na=False)
    # Excel, unlike CSV, can still yield NaN (a float) for a genuinely blank
    # cell even with dtype=str/keep_default_na=False — missing-value
    # handling happens independently of dtype coercion there. Normalize to
    # str everywhere so every downstream .strip() call is safe regardless
    # of source format. fillna("") first so a blank cell becomes "" rather
    # than the literal string "nan".
    return df.fillna("").astype(str)


def _read_dataframe(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        return _parse_dataframe(f, path.suffix.lower())


def _repair_mojibake(text: str) -> tuple[str, bool]:
    """Undo UTF-8 bytes that were previously mis-decoded through a
    single-byte codepage, e.g. "donâ€™t" -> "don't". Delegates to ftfy
    rather than a hand-rolled cp1252/latin-1 round trip: Python's strict
    cp1252 codec raises on bytes 0x81/0x8D/0x8F/0x90/0x9D (undefined in
    that table), but real-world corruption — e.g. from browsers/JS, which
    follow the WHATWG windows-1252 spec — maps those bytes to their raw C1
    control codepoints instead of erroring. A naive round trip bails on
    text containing them (this is exactly what silently left curly
    double-quotes unrepaired). ftfy.fix_encoding handles this and other
    mojibake patterns without touching anything beyond encoding repair
    (unlike ftfy.fix_text, which also normalizes whitespace/quotes/etc.).
    Returns (repaired_text, was_repaired)."""
    repaired = ftfy.fix_encoding(text)
    return repaired, repaired != text


def _response_key(dataset_id: int, question_id: int, source_row_index: int) -> str:
    return f"{dataset_id}:{question_id}:{source_row_index}"


def _get_dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def _ensure_uploads(db: Session, dataset: Dataset) -> list[Upload]:
    """Return the dataset's uploads ordered by row_offset, lazily synthesizing
    Upload #1 for a dataset ingested before the uploads table existed (the
    scripts.backfill_uploads self-heal, so an un-backfilled machine degrades
    gracefully instead of crashing on re-select)."""
    uploads = (
        db.query(Upload)
        .filter(Upload.dataset_id == dataset.id)
        .order_by(Upload.row_offset)
        .all()
    )
    if uploads:
        return uploads

    # The stored original is what the synthesized Upload #1 is derived from. If
    # it is gone or unreadable, say which file and why — the alternative is an
    # unhandled traceback out of select_columns / append / history, which tells
    # the analyst nothing about what to put back.
    source = REPO_ROOT / (dataset.original_path or "")
    if not dataset.original_path or not source.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Dataset {dataset.id} has no readable original file "
            f"({source if dataset.original_path else 'no path recorded'}). It "
            "predates the uploads table and cannot be back-filled without it; "
            "restore the file, or re-ingest the dataset.",
        )
    try:
        df = _read_dataframe(source)
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset {dataset.id}'s original file could not be parsed "
            f"({source}): {exc}",
        ) from exc
    hashes = rowhash.hash_dataframe(df)
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
    db.flush()
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
    db.query(Response).filter(
        Response.dataset_id == dataset.id, Response.upload_id.is_(None)
    ).update({"upload_id": upload.id}, synchronize_session=False)
    return [upload]


def _read_upload_frame(upload: Upload) -> pd.DataFrame:
    """One upload's stored file, or a 409 naming it. Re-select and append both
    re-read every file a dataset was built from; a file that has gone missing
    is an operator-fixable state, not a server fault, and the message has to
    say which file so it can be put back."""
    path = REPO_ROOT / upload.stored_path
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Stored file for upload {upload.id} "
            f"('{upload.stored_filename}') is missing at {path}. Restore it, "
            "or re-ingest the dataset.",
        )
    try:
        return _read_dataframe(path)
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Stored file for upload {upload.id} "
            f"('{upload.stored_filename}') could not be parsed: {exc}",
        ) from exc


def _insert_column_fingerprints(
    db: Session, dataset_id: int, upload_id: int, fingerprints: dict[str, str]
) -> None:
    db.bulk_insert_mappings(
        ColumnFingerprint,
        [
            {
                "dataset_id": dataset_id,
                "upload_id": upload_id,
                "column_name": col,
                "fingerprint": fp,
            }
            for col, fp in fingerprints.items()
        ],
    )


def _ensure_column_fingerprints(db: Session) -> None:
    """Lazily compute column fingerprints for ingested datasets' uploads that
    predate the column_fingerprints table (same self-heal pattern as
    _ensure_uploads) — each stored file is read once, ever. An unreadable
    file is skipped: matching then simply can't see that upload, which
    degrades to the pre-fingerprint behaviour instead of failing the upload."""
    uploads = (
        db.query(Upload)
        .join(Dataset, Dataset.id == Upload.dataset_id)
        .outerjoin(ColumnFingerprint, ColumnFingerprint.upload_id == Upload.id)
        .filter(Dataset.status == "ingested", ColumnFingerprint.id.is_(None))
        .all()
    )
    for upload in uploads:
        path = REPO_ROOT / upload.stored_path
        try:
            df = _read_dataframe(path)
        except Exception:
            continue
        _insert_column_fingerprints(
            db, upload.dataset_id, upload.id, rowhash.fingerprint_columns(df)
        )
    if uploads:
        db.commit()


def _owned_row_indices(db: Session, upload: Upload) -> list[int]:
    """Global row indices this upload contributed to the dataset (its
    non-duplicate rows), ascending. Local index into the upload's file is
    global - row_offset."""
    rows = (
        db.query(RowHash.row_index)
        .filter(RowHash.upload_id == upload.id, RowHash.is_duplicate == False)  # noqa: E712
        .order_by(RowHash.row_index)
        .all()
    )
    return [r[0] for r in rows]


# Cardinality above which a metadata column is flagged as probably too
# fine-grained to filter usefully. Advisory ONLY — nothing refuses a column
# past it. A survey with 100 districts is a real survey; a column with 8,000
# distinct values is almost certainly free text or an exact age, and the
# analyst gets told so and decides. See docs/DEMOGRAPHICS_PLAN.md §3.
METADATA_MAX_DISTINCT = 100
# How many (value, count) pairs travel in a DTO. The database keeps them all.
METADATA_VALUES_IN_DTO = 200


def _upsert_upload_metadata(
    db: Session,
    dataset: Dataset,
    upload: Upload,
    df: pd.DataFrame,
    columns: dict[str, "MetadataColumn"],
) -> None:
    """Store one upload's demographic values, mirroring
    _upsert_upload_responses: bulk mappings, deletion scoped to this upload
    so re-reading one file can never drop another file's rows.

    A column absent from THIS file is skipped rather than erroring — an
    appended wave may not carry every demographic the first file had, and
    those respondents simply have no value for it (missing, not "Unknown").
    """
    owned = _owned_row_indices(db, upload)
    db.query(RespondentAttribute).filter(
        RespondentAttribute.dataset_id == dataset.id,
        RespondentAttribute.upload_id == upload.id,
    ).delete(synchronize_session=False)
    if not columns:
        return

    rows: list[dict] = []
    for source_column, col in columns.items():
        if source_column not in df.columns:
            continue
        values = df[source_column].tolist()
        is_date = col.value_type == "date"
        for global_index in owned:
            local_index = global_index - upload.row_offset
            raw = values[local_index] if 0 <= local_index < len(values) else ""
            # same mojibake repair the response text gets — a demographic
            # value is displayed and filtered on, so "Distrito Três" must not
            # arrive mangled and split one real group into two
            value, _repaired = _repair_mojibake(str(raw).strip())
            value = value.strip()
            if is_date:
                # canonical ISO or nothing — an unparseable date cell is
                # missing data, the same asymmetry a blank cell has
                value = dates_module.parse_date(value) or ""
            if not value:
                continue          # blank = missing, never a stored category
            rows.append(
                {
                    "dataset_id": dataset.id,
                    "upload_id": upload.id,
                    "metadata_column_id": col.id,
                    "source_row_index": global_index,
                    "value": value,
                }
            )
    if rows:
        db.bulk_insert_mappings(RespondentAttribute, rows)


def _upsert_upload_responses(
    db: Session,
    dataset: Dataset,
    upload: Upload,
    df: pd.DataFrame,
    questions: dict[str, QuestionColumn],
    respondent_id_column: str | None,
) -> dict[int, int]:
    """Upsert Response rows for one upload's owned rows across the selected
    questions; delete rows this upload previously contributed that are no
    longer present. Deletion is scoped to this upload_id — re-reading one
    file can never delete rows another file contributed. Returns
    {question_id: inserted_row_count} (new rows only, for append reporting).

    Every cell is re-cleaned on every run so edits to the mojibake repair
    logic apply retroactively. Bulk mappings instead of per-row ORM objects:
    at 30k rows x several questions the unit-of-work bookkeeping is minutes,
    the bulk path is seconds.
    """
    owned = _owned_row_indices(db, upload)
    respondent_ids = (
        df[respondent_id_column].tolist()
        if respondent_id_column and respondent_id_column in df.columns
        else None
    )
    inserted_per_question: dict[int, int] = {}

    for column, question in questions.items():
        texts = df[column].tolist()
        existing_ids = dict(
            db.query(Response.source_row_index, Response.id)
            .filter(
                Response.dataset_id == dataset.id,
                Response.question_id == question.id,
                Response.upload_id == upload.id,
            )
            .all()
        )
        inserts: list[dict] = []
        updates: list[dict] = []
        seen_rows: set[int] = set()

        for global_index in owned:
            local_index = global_index - upload.row_offset
            raw_text = texts[local_index]
            if not raw_text.strip():
                continue
            seen_rows.add(global_index)

            cleaned_text, was_repaired = _repair_mojibake(raw_text.strip())
            fields = {
                "respondent_id": (
                    respondent_ids[local_index] if respondent_ids else None
                ),
                "raw_text_original": raw_text,
                "response_text": cleaned_text,
                "was_encoding_repaired": was_repaired,
                "is_nonanswer": is_nonanswer_text(cleaned_text),
            }
            existing_id = existing_ids.get(global_index)
            if existing_id is not None:
                updates.append({"id": existing_id, **fields})
            else:
                inserts.append(
                    {
                        "dataset_id": dataset.id,
                        "question_id": question.id,
                        "upload_id": upload.id,
                        "source_row_index": global_index,
                        "response_key": _response_key(
                            dataset.id, question.id, global_index
                        ),
                        **fields,
                    }
                )

        db.bulk_insert_mappings(Response, inserts)
        db.bulk_update_mappings(Response, updates)
        inserted_per_question[question.id] = len(inserts)

        stale_ids = [rid for idx, rid in existing_ids.items() if idx not in seen_rows]
        # chunk the IN() list — SQLite's default parameter limit is 999
        for i in range(0, len(stale_ids), 900):
            db.query(Response).filter(
                Response.id.in_(stale_ids[i : i + 900])
            ).delete(synchronize_session=False)

    return inserted_per_question


def _load_label_assignments(dataset_id: int) -> tuple[dict[str, dict], dict[str, str]]:
    """(response_key -> latest assignment record, question_id -> labels run
    id) for every question with a labels run on disk. Same latest-run
    convention as summary/locations. Lazy import keeps ingest usable without
    the pipeline modules loaded."""
    from app.summary import LABELS_DIR, latest_run_dir

    by_key: dict[str, dict] = {}
    runs: dict[str, str] = {}
    ds_dir = LABELS_DIR / str(dataset_id)
    if not ds_dir.exists():
        return by_key, runs
    for qdir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
        run_dir = latest_run_dir(qdir, "assignments.json")
        if run_dir is None:
            continue
        runs[qdir.name] = run_dir.name
        assignments = json.loads(
            (run_dir / "assignments.json").read_text(encoding="utf-8")
        )
        for a in assignments:
            key = a.get("response_key")
            if key:
                by_key[key] = a
    return by_key, runs


def _write_exports(db: Session, dataset: Dataset) -> ExportInfo:
    """Regenerate responses.parquet, responses.csv, and manifest.json from
    the current database state, left-joining the latest labeling run per
    question (rows without one get None for the whole label block). The
    database stays the source of truth — these files are reproducible
    exports, safe to delete and regenerate.

    The export directory is wiped and rewritten from scratch each time,
    rather than overwritten file-by-file — otherwise a re-ingest that drops
    a question would leave that question's rows stranded in the old export
    files even though the manifest only reflects the current selection."""
    export_dir = EXPORTS_DIR / str(dataset.id)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    questions = (
        db.query(QuestionColumn)
        .filter(QuestionColumn.dataset_id == dataset.id)
        .order_by(QuestionColumn.position)
        .all()
    )
    question_by_id = {q.id: q for q in questions}
    # Column tuples, not ORM objects — at 30k rows x several questions the
    # identity-map bookkeeping is the dominant cost of the old query.
    responses = (
        db.query(
            Response.question_id,
            Response.source_row_index,
            Response.response_key,
            Response.respondent_id,
            Response.response_text,
            Response.raw_text_original,
            Response.was_encoding_repaired,
            Response.is_nonanswer,
        )
        .filter(Response.dataset_id == dataset.id)
        .order_by(Response.question_id, Response.source_row_index)
        .all()
    )

    labels_by_key, labels_runs = _load_label_assignments(dataset.id)
    matched_keys = 0

    # Metadata columns drive two things below: the demographics sidecar and
    # the flat response_date column (from the first date-typed column, which
    # is per-respondent, so one lookup by row index serves every question).
    meta_cols = (
        db.query(MetadataColumn)
        .filter(MetadataColumn.dataset_id == dataset.id)
        .order_by(MetadataColumn.position)
        .all()
    )
    date_col = next((m for m in meta_cols if m.value_type == "date"), None)
    date_by_row: dict[int, str] = {}
    if date_col is not None:
        date_by_row = dict(
            db.query(
                RespondentAttribute.source_row_index, RespondentAttribute.value
            )
            .filter(
                RespondentAttribute.dataset_id == dataset.id,
                RespondentAttribute.metadata_column_id == date_col.id,
            )
            .all()
        )

    per_question_counts: dict[str, int] = {q.label: 0 for q in questions}
    per_question_nonanswer_counts: dict[str, int] = {q.label: 0 for q in questions}
    encoding_repairs_applied = 0
    records = []
    for r in responses:
        q = question_by_id[r.question_id]
        per_question_counts[q.label] += 1
        if r.is_nonanswer:
            per_question_nonanswer_counts[q.label] += 1
        if r.was_encoding_repaired:
            encoding_repairs_applied += 1
        a = labels_by_key.get(r.response_key)
        if a is not None:
            matched_keys += 1
        records.append(
            {
                "dataset_id": dataset.id,
                "question_id": r.question_id,
                "question_label": q.label,
                "source_column": q.source_column,
                "source_row_index": r.source_row_index,
                "response_key": r.response_key,
                "respondent_key": f"{dataset.id}:{r.source_row_index}",
                "respondent_id": r.respondent_id,
                "response_text": r.response_text,
                "raw_text_original": r.raw_text_original,
                "was_encoding_repaired": r.was_encoding_repaired,
                "is_nonanswer": r.is_nonanswer,
                "label_ids": a.get("label_ids") if a else None,
                "uncategorized": a.get("uncategorized") if a else None,
                "fit": a.get("fit") if a else None,
                "locations": a.get("locations") if a else None,
                "time_context": a.get("time_context") if a else None,
                "actionability": a.get("actionability") if a else None,
                "event_occurred": a.get("event_occurred") if a else None,
                "response_date": date_by_row.get(r.source_row_index),
            }
        )

    parquet_path = export_dir / "responses.parquet"
    table = pa.Table.from_pylist(records, schema=RESPONSE_PARQUET_SCHEMA)
    pq.write_table(table, parquet_path)

    # Demographics go in a SIDECAR, not into RESPONSE_PARQUET_SCHEMA. That
    # schema is fixed and every artifact and the whole ask path depend on it;
    # per-dataset demographic columns cannot live in a fixed schema, and
    # widening it for a map column would rewrite the contract for every
    # dataset that has no demographics at all. Long format, one row per
    # (respondent, field) — blanks simply absent. Written only when the
    # dataset has metadata columns, so nothing changes for datasets without.
    # (response_date above is the one deliberate exception: uniform name and
    # meaning across datasets, so it may live in the fixed schema.)
    if meta_cols:
        label_by_id = {m.id: m.label for m in meta_cols}
        attrs = (
            db.query(
                RespondentAttribute.metadata_column_id,
                RespondentAttribute.source_row_index,
                RespondentAttribute.value,
            )
            .filter(RespondentAttribute.dataset_id == dataset.id)
            .order_by(
                RespondentAttribute.source_row_index,
                RespondentAttribute.metadata_column_id,
            )
            .all()
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "respondent_key": f"{dataset.id}:{a.source_row_index}",
                        "source_row_index": a.source_row_index,
                        "field": label_by_id[a.metadata_column_id],
                        "value": a.value,
                    }
                    for a in attrs
                ],
                schema=RESPONDENT_PARQUET_SCHEMA,
            ),
            export_dir / "respondents.parquet",
        )

    csv_path = export_dir / "responses.csv"
    csv_records = [
        {
            **rec,
            **{
                col: json.dumps(rec[col]) if rec[col] is not None else ""
                for col in LIST_COLUMNS
            },
        }
        for rec in records
    ]
    pd.DataFrame(
        csv_records, columns=[f.name for f in RESPONSE_PARQUET_SCHEMA]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "dataset_id": dataset.id,
        "source_filename": dataset.original_filename,
        "sheet": dataset.sheet_name,
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "respondent_id_column": dataset.respondent_id_column,
        "dataset_description": dataset.description or "",
        # Catalog metadata — audit record only, never read by any prompt path.
        "dataset_name": dataset.name,
        "dataset_department": dataset.department or "",
        "dataset_notes": dataset.notes or "",
        "survey_start_date": dataset.survey_start_date or "",
        "survey_end_date": dataset.survey_end_date or "",
        "column_mapping": [
            {"question_id": q.id, "source_column": q.source_column, "label": q.label}
            for q in questions
        ],
        # Which demographic fields the sidecar carries and how each is typed —
        # the ask layer needs value_type to know a field holds ISO dates whose
        # facets should be derived period labels, not raw dates.
        "metadata_columns": [
            {
                "source_column": m.source_column,
                "label": m.label,
                "value_type": m.value_type,
                "n_distinct": m.n_distinct,
            }
            for m in meta_cols
        ],
        # Period-labeling config for date-typed fields (app/dates.py shapes);
        # null when unconfigured (readers fall back to quarter bucketing).
        "date_ranges": (
            json.loads(dataset.date_ranges_json)
            if dataset.date_ranges_json
            else None
        ),
        "per_question_counts": per_question_counts,
        "per_question_nonanswer_counts": per_question_nonanswer_counts,
        "total_row_count": len(records),
        "encoding_repairs_applied": encoding_repairs_applied,
        # which labeling run each question's label columns came from, and how
        # many assignment keys had no DB row (stale run vs re-ingest) — never
        # dropped silently
        "labels_runs": labels_runs,
        "labels_unmatched_keys": len(labels_by_key) - matched_keys,
        # every source file merged into this dataset, with how many of its
        # rows were new vs already-ingested duplicates that were skipped
        "uploads": [
            {
                "upload_id": u.id,
                "filename": u.stored_filename,
                "uploaded_at": u.uploaded_at.isoformat() if u.uploaded_at else None,
                "row_offset": u.row_offset,
                "row_count": u.row_count,
                "new_row_count": u.new_row_count,
                "duplicate_row_count": u.duplicate_row_count,
                "note": u.note,
            }
            for u in db.query(Upload)
            .filter(Upload.dataset_id == dataset.id)
            .order_by(Upload.row_offset)
            .all()
        ],
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # The ask layer memoizes a dataset's loaded context; this call is the only
    # thing in-process that rewrites the corpus underneath it. Its own key
    # would catch this via the parquet's size/mtime, but mtime is not trusted
    # anywhere in this codebase — so say so explicitly rather than rely on it.
    from app import ask_service

    ask_service.invalidate_context_cache(dataset.id)

    return ExportInfo(
        csv_path=str(csv_path.relative_to(REPO_ROOT)),
        parquet_path=str(parquet_path.relative_to(REPO_ROOT)),
        manifest_path=str(manifest_path.relative_to(REPO_ROOT)),
        csv_download_url=f"/datasets/{dataset.id}/exports/csv",
        parquet_download_url=f"/datasets/{dataset.id}/exports/parquet",
        total_row_count=len(records),
        per_question_counts=per_question_counts,
    )


def _load_export_info(dataset_id: int) -> ExportInfo | None:
    manifest_path = EXPORTS_DIR / str(dataset_id) / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_path = EXPORTS_DIR / str(dataset_id) / "responses.csv"
    parquet_path = EXPORTS_DIR / str(dataset_id) / "responses.parquet"
    return ExportInfo(
        csv_path=str(csv_path.relative_to(REPO_ROOT)),
        parquet_path=str(parquet_path.relative_to(REPO_ROOT)),
        manifest_path=str(manifest_path.relative_to(REPO_ROOT)),
        csv_download_url=f"/datasets/{dataset_id}/exports/csv",
        parquet_download_url=f"/datasets/{dataset_id}/exports/parquet",
        total_row_count=manifest["total_row_count"],
        per_question_counts=manifest["per_question_counts"],
    )


def _dataset_out(db: Session, dataset: Dataset) -> DatasetOut:
    exports = _load_export_info(dataset.id)
    # One grouped COUNT instead of a per-question len(responses) property,
    # which materialized every Response row.
    counts_by_qid = dict(
        db.query(Response.question_id, func.count(Response.id))
        .filter(Response.dataset_id == dataset.id)
        .group_by(Response.question_id)
        .all()
    )
    questions = []
    for q in sorted(dataset.questions, key=lambda q: q.position):
        # Once exports exist, the manifest is the audited record of what was
        # actually written to disk — prefer it over the live DB count so the
        # API response and the manifest can never disagree.
        if exports is not None and q.label in exports.per_question_counts:
            count = exports.per_question_counts[q.label]
        else:
            count = counts_by_qid.get(q.id, 0)
        questions.append(
            QuestionColumnOut(
                id=q.id,
                source_column=q.source_column,
                label=q.label,
                response_count=count,
            )
        )
    # Demographic columns with their value distribution — one grouped COUNT
    # for the whole dataset rather than a query per column.
    meta_out: list[MetadataColumnOut] = []
    if dataset.metadata_columns:
        counts = (
            db.query(
                RespondentAttribute.metadata_column_id,
                RespondentAttribute.value,
                func.count(RespondentAttribute.id),
            )
            .filter(RespondentAttribute.dataset_id == dataset.id)
            .group_by(
                RespondentAttribute.metadata_column_id, RespondentAttribute.value
            )
            .all()
        )
        by_col: dict[int, list[MetadataValueCount]] = {}
        for col_id, value, n in counts:
            by_col.setdefault(col_id, []).append(
                MetadataValueCount(value=value, n_respondents=n)
            )
        for m in sorted(dataset.metadata_columns, key=lambda m: m.position):
            vals = sorted(by_col.get(m.id, []),
                          key=lambda v: (-v.n_respondents, v.value))
            meta_out.append(
                MetadataColumnOut(
                    id=m.id,
                    source_column=m.source_column,
                    label=m.label,
                    value_type=m.value_type,
                    n_distinct=m.n_distinct,
                    values=vals[:METADATA_VALUES_IN_DTO],
                    high_cardinality=m.n_distinct > METADATA_MAX_DISTINCT,
                )
            )

    return DatasetOut(
        id=dataset.id,
        name=dataset.name,
        metadata_columns=meta_out,
        original_filename=dataset.original_filename,
        status=dataset.status,
        uploaded_at=dataset.uploaded_at,
        respondent_id_column=dataset.respondent_id_column,
        description=dataset.description,
        department=dataset.department,
        notes=dataset.notes,
        survey_start_date=dataset.survey_start_date,
        survey_end_date=dataset.survey_end_date,
        date_ranges=(
            json.loads(dataset.date_ranges_json)
            if dataset.date_ranges_json
            else None
        ),
        questions=questions,
        exports=exports,
    )


@router.post("/upload", response_model=UploadResponse,
             dependencies=[Depends(require_admin)])
async def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Expected one of {sorted(SUPPORTED_SUFFIXES)}.",
        )

    contents = await file.read()
    try:
        df = _parse_dataframe(io.BytesIO(contents), suffix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no rows")

    sheet_name = None
    if suffix in {".xlsx", ".xls"}:
        try:
            sheet_name = pd.ExcelFile(io.BytesIO(contents)).sheet_names[0]
        except Exception:
            sheet_name = None

    safe_filename = Path(file.filename or f"{uuid.uuid4().hex}{suffix}").name

    # Duplicate check BEFORE this file's own hashes are stored (no self-match).
    # Only ingested datasets participate, so an abandoned provisional upload
    # can never claim rows as "already ingested".
    hashes = rowhash.hash_dataframe(df)
    check = rowhash.find_matches(db, hashes)
    fingerprints = rowhash.fingerprint_columns(df)

    # Column-level tier: only when row matching found nothing — a re-upload
    # with a column added/dropped/renamed changes every row hash, but the
    # untouched columns still fingerprint identically. Diagnostic only.
    column_matches: list[ColumnMatch] = []
    if check["outcome"] == "none":
        _ensure_column_fingerprints(db)
        column_matches = [
            ColumnMatch(**m) for m in rowhash.find_column_matches(db, fingerprints)
        ]

    duplicate_check = DuplicateCheck(
        outcome=check["outcome"],
        best_dataset_id=check["best_dataset_id"],
        matches=[
            DatasetMatch(
                dataset_id=m["dataset_id"],
                dataset_name=m["dataset_name"],
                matched_rows=m["matched_rows"],
                file_rows=m["file_rows"],
                dataset_rows=m["dataset_rows"],
                exact=m["exact"],
                columns_compatible=not [
                    c for c in m["selected_columns"] if c not in df.columns
                ],
                missing_columns=[
                    c for c in m["selected_columns"] if c not in df.columns
                ],
            )
            for m in check["matches"]
        ],
        column_matches=column_matches,
    )

    dataset = Dataset(
        name=file.filename or safe_filename,
        original_filename=safe_filename,
        original_path="",  # finalized below, once we have dataset.id
        sheet_name=sheet_name,
        status="uploaded",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    dataset_upload_dir = UPLOADS_DIR / str(dataset.id)
    dataset_upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = dataset_upload_dir / safe_filename
    stored_path.write_bytes(contents)
    dataset.original_path = str(stored_path.relative_to(REPO_ROOT))

    upload = Upload(
        dataset_id=dataset.id,
        stored_filename=safe_filename,
        stored_path=dataset.original_path,
        sheet_name=sheet_name,
        row_offset=0,
        row_count=len(df),
        new_row_count=len(df),
        duplicate_row_count=0,
    )
    db.add(upload)
    db.flush()
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
    _insert_column_fingerprints(db, dataset.id, upload.id, fingerprints)
    db.commit()

    columns = [
        ColumnPreview(
            column=col,
            sample_values=[v for v in df[col].head(5).tolist() if v != ""],
            non_null_count=int((df[col] != "").sum()),
        )
        for col in df.columns
    ]

    return UploadResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        row_count=len(df),
        columns=columns,
        duplicate_check=duplicate_check,
    )


@router.delete("/{dataset_id}", status_code=204,
               dependencies=[Depends(require_admin)])
def discard_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Discard a provisional upload. Only allowed before column selection.

    An ingested dataset has derived artifacts (taxonomies, labels, answers)
    and deleting it from THIS endpoint would orphan them silently — which is
    why permanent deletion is a separate, explicitly-named endpoint that
    clears all of them. This one stays narrow on purpose: the frontend fires
    it from `pagehide` when a tab closes mid-upload, and an abandoned-tab
    handler must not be able to destroy processed work.
    """
    dataset = _get_dataset_or_404(db, dataset_id)
    if dataset.status != "uploaded":
        raise HTTPException(
            status_code=409,
            detail=f"Dataset {dataset_id} is '{dataset.status}' — only provisional "
            "(status 'uploaded') datasets can be discarded. To remove an "
            "ingested dataset and everything derived from it, use "
            f"DELETE /datasets/{dataset_id}/permanently.",
        )
    db.delete(dataset)
    db.commit()
    upload_dir = UPLOADS_DIR / str(dataset_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)


@router.get("/{dataset_id}/deletion-preview", response_model=DeletionPreview)
def deletion_preview(dataset_id: int, db: Session = Depends(get_db)):
    """Exactly what deleting this dataset would destroy. Reads only.

    The confirm dialog is built from this rather than from guesses, because
    the numbers are the whole argument: rows and questions read as data you
    could re-upload, while recorded model spend reads as money already gone.
    """
    dataset = _get_dataset_or_404(db, dataset_id)
    files = purge.preview(dataset.id)
    n_responses = (
        db.query(func.count(Response.id))
        .filter(Response.dataset_id == dataset.id)
        .scalar() or 0
    )
    n_uploads = (
        db.query(func.count(Upload.id))
        .filter(Upload.dataset_id == dataset.id)
        .scalar() or 0
    )
    job = purge.running_job(dataset.id)
    return DeletionPreview(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        status=dataset.status,
        n_responses=int(n_responses),
        n_questions=len(dataset.questions),
        n_metadata_columns=len(dataset.metadata_columns),
        n_uploads=int(n_uploads),
        roots=[
            DeletionPreviewRoot(**vars(root))
            for root in files.roots if root.exists
        ],
        total_files=files.total_files,
        total_bytes=files.total_bytes,
        total_spent_usd=round(files.total_spent_usd, 4),
        blocked_by_running_job=job is not None,
    )


@router.delete("/{dataset_id}/permanently", response_model=DeletionResult,
               dependencies=[Depends(require_admin)])
def delete_dataset_permanently(
    dataset_id: int, confirm_name: str = "", db: Session = Depends(get_db)
):
    """Delete a dataset and every artifact derived from it. Irreversible.

    `confirm_name` must equal the dataset's name. It is not security — the
    caller already passed the admin gate — it is a speed bump on the one
    action in this app that cannot be undone, and it makes an accidental
    DELETE with the wrong id a 400 instead of a catastrophe.

    Order matters. The database rows go first, in child-first bulk statements
    rather than by ORM cascade: cascading a 30k-response dataset means loading
    every child row into the identity map and issuing one DELETE per row, and
    the request would still be running minutes later. Files go second, so a
    failure to unlink something leaves recoverable orphans rather than rows
    pointing at data that is already gone. Caches go last, once there is
    nothing left for them to be repopulated from.
    """
    dataset = _get_dataset_or_404(db, dataset_id)

    if confirm_name.strip() != dataset.name.strip():
        raise HTTPException(
            status_code=400,
            detail="The confirmation name does not match this dataset's name "
            f"('{dataset.name}'). Nothing was deleted.",
        )

    # Deleting a tree a pipeline subprocess is mid-write in is how you get a
    # half-removed run that still looks loadable. Same refusal as append.
    job = purge.running_job(dataset.id)
    if job is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline job is running for dataset {dataset.id} "
            f"(job {job.job_id}) — cancel it or wait for it to finish before "
            "deleting. Nothing was deleted.",
        )

    name = dataset.name
    for model in (Response, RespondentAttribute, ColumnFingerprint, RowHash,
                  QuestionColumn, MetadataColumn, Upload):
        db.query(model).filter(model.dataset_id == dataset.id).delete(
            synchronize_session=False)
    db.delete(dataset)
    db.commit()

    failures = purge.purge_files(dataset_id)
    purge.forget_in_memory(dataset_id)

    return DeletionResult(
        dataset_id=dataset_id,
        dataset_name=name,
        deleted=True,
        # An empty list is the whole point of reporting this: a dataset id is
        # reused by the next upload, so anything left behind would later be
        # adopted by unrelated data. Say which roots survived so it can be
        # cleaned up by hand.
        undeleted_roots=[
            UndeletedRoot(key=key, error=error)
            for key, error in sorted(failures.items())
        ],
    )


@router.post("/{dataset_id}/append", response_model=AppendResponse,
             dependencies=[Depends(require_admin)])
def append_upload(
    dataset_id: int, body: AppendRequest, db: Session = Depends(get_db)
):
    """Merge a provisional upload into an existing ingested dataset. Rows the
    dataset already holds (by whole-row hash, multiset semantics) are skipped;
    only genuinely new rows become Response rows, at global row indices past
    everything already there, so every existing response_key — and therefore
    every label — stays valid. The provisional dataset is consumed on success;
    its file is copied under the target first, byte-for-byte."""
    target = _get_dataset_or_404(db, dataset_id)
    provisional = _get_dataset_or_404(db, body.upload_dataset_id)
    if target.id == provisional.id:
        raise HTTPException(status_code=400, detail="Cannot append a dataset to itself")
    if target.status != "ingested":
        raise HTTPException(
            status_code=400,
            detail=f"Target dataset {target.id} is '{target.status}' — appends "
            "require an ingested dataset.",
        )
    if provisional.status != "uploaded":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {provisional.id} is '{provisional.status}' — only a "
            "provisional upload (status 'uploaded') can be appended.",
        )

    # Appending rewrites the export a running pipeline stage may be reading.
    from app import pipeline as pipeline_module

    job = pipeline_module.latest_job(str(target.id))
    if job is not None and job.status == "running":
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline job is running for dataset {target.id} — wait for "
            "it to finish before appending.",
        )

    source_path = REPO_ROOT / provisional.original_path
    if not source_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"The uploaded file for dataset {provisional.id} is missing "
            f"at {source_path}; re-upload it.",
        )
    df = _read_dataframe(source_path)

    questions = (
        db.query(QuestionColumn)
        .filter(QuestionColumn.dataset_id == target.id)
        .order_by(QuestionColumn.position)
        .all()
    )
    missing = [q.source_column for q in questions if q.source_column not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Appended file is missing the dataset's question columns: "
            f"{missing}",
        )

    warnings: list[str] = []
    if (
        target.respondent_id_column
        and target.respondent_id_column not in df.columns
    ):
        warnings.append(
            f"Respondent-id column '{target.respondent_id_column}' is not in the "
            "appended file; its rows carry no respondent id."
        )
    known_columns = {q.source_column for q in questions}
    if target.respondent_id_column:
        known_columns.add(target.respondent_id_column)
    extra = [c for c in df.columns if c not in known_columns]
    if extra:
        warnings.append(f"Ignored columns not selected in this dataset: {extra}")

    # Recompute duplicate flags against the target's CURRENT hash multiset —
    # the upload-time report may be stale by the time the analyst confirms.
    uploads = _ensure_uploads(db, target)
    hashes = rowhash.hash_dataframe(df)
    dup_flags = rowhash.split_new_rows(hashes, rowhash.stored_multiset(db, target.id))
    n_duplicates = sum(dup_flags)
    n_new = len(dup_flags) - n_duplicates

    row_offset = max(u.row_offset + u.row_count for u in uploads)

    upload = Upload(
        dataset_id=target.id,
        stored_filename=provisional.original_filename,
        stored_path="",  # finalized below, once we have upload.id
        sheet_name=provisional.sheet_name,
        row_offset=row_offset,
        row_count=len(df),
        new_row_count=n_new,
        duplicate_row_count=n_duplicates,
        note=body.note.strip() or None if body.note is not None else None,
    )
    db.add(upload)
    db.flush()

    # Copy the original bytes under the target (subdir per upload id, so two
    # files with the same name can't collide) before the provisional dataset
    # and its copy are deleted — the original stays immutable and auditable.
    append_dir = UPLOADS_DIR / str(target.id) / str(upload.id)
    append_dir.mkdir(parents=True, exist_ok=True)
    stored_path = append_dir / provisional.original_filename
    stored_path.write_bytes(source_path.read_bytes())
    upload.stored_path = str(stored_path.relative_to(REPO_ROOT))

    db.bulk_insert_mappings(
        RowHash,
        [
            {
                "dataset_id": target.id,
                "upload_id": upload.id,
                "row_index": row_offset + i,
                "row_hash": h,
                "is_duplicate": dup_flags[i],
            }
            for i, h in enumerate(hashes)
        ],
    )
    _insert_column_fingerprints(
        db, target.id, upload.id, rowhash.fingerprint_columns(df)
    )

    questions_by_column = {q.source_column: q for q in questions}
    inserted = _upsert_upload_responses(
        db, target, upload, df, questions_by_column, target.respondent_id_column
    )
    # The appended wave's demographics, for the columns the dataset already
    # has selected. A column this file lacks is skipped inside the helper —
    # its respondents get no value, which is missing data, not an error.
    _upsert_upload_metadata(
        db,
        target,
        upload,
        df,
        {m.source_column: m for m in db.query(MetadataColumn)
         .filter(MetadataColumn.dataset_id == target.id).all()},
    )
    labels_by_question = {q.id: q.label for q in questions}
    new_responses_per_question = {
        labels_by_question[qid]: n for qid, n in inserted.items()
    }

    # The provisional dataset served its purpose; its bytes now live under the
    # target. Cascades take its Upload/RowHash rows with it.
    provisional_id = provisional.id
    db.delete(provisional)
    db.commit()
    db.refresh(target)
    provisional_dir = UPLOADS_DIR / str(provisional_id)
    if provisional_dir.exists():
        shutil.rmtree(provisional_dir, ignore_errors=True)

    try:
        _write_exports(db, target)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Append committed to the database but export failed: {exc}. "
            f"Retry via POST /datasets/{target.id}/export.",
        ) from exc

    return AppendResponse(
        dataset=_dataset_out(db, target),
        upload_id=upload.id,
        appended_rows=n_new,
        skipped_duplicates=n_duplicates,
        new_responses_per_question=new_responses_per_question,
        warnings=warnings,
    )


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    return [_dataset_out(db, d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = _get_dataset_or_404(db, dataset_id)
    return _dataset_out(db, dataset)


@router.patch("/{dataset_id}", response_model=DatasetOut,
              dependencies=[Depends(require_admin)])
def update_dataset_metadata(
    dataset_id: int, body: DatasetMetadataPatch, db: Session = Depends(get_db)
):
    """Edit catalog metadata (name, department, notes, survey dates). The
    description is deliberately not editable here — it feeds every analysis
    prompt and is hashed into run ids, so it is fixed at column-select time.
    A real change on an ingested dataset rewrites the exports so the manifest
    (the audit record) never disagrees with the DB."""
    dataset = _get_dataset_or_404(db, dataset_id)

    # Work out what would change BEFORE touching the ORM object, so the 409
    # below never leaves dirty state in the session (which only stayed harmless
    # because get_db closes and rolls back).
    pending: dict[str, str | None] = {}
    if body.name is not None and body.name.strip() != dataset.name:
        pending["name"] = body.name.strip()
    for field, value in (
        ("department", body.department),
        ("notes", body.notes),
        ("survey_start_date", body.survey_start_date),
        ("survey_end_date", body.survey_end_date),
    ):
        if value is None:
            continue
        new = value.strip() or None
        if new != getattr(dataset, field):
            pending[field] = new
    if body.date_ranges is not None:
        new_ranges = json.dumps(body.date_ranges)
        if new_ranges != dataset.date_ranges_json:
            pending["date_ranges_json"] = new_ranges

    if not pending:
        return _dataset_out(db, dataset)

    if dataset.status == "ingested":
        # Rewriting the export a running pipeline stage may be reading is the
        # same hazard as appending — refuse rather than race.
        from app import pipeline as pipeline_module

        job = pipeline_module.latest_job(str(dataset.id))
        if job is not None and job.status == "running":
            raise HTTPException(
                status_code=409,
                detail=f"A pipeline job is running for dataset {dataset.id} — "
                "wait for it to finish before editing metadata.",
            )

    for field, new in pending.items():
        setattr(dataset, field, new)
    db.commit()
    db.refresh(dataset)

    if dataset.status == "ingested":
        try:
            _write_exports(db, dataset)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Metadata saved but export refresh failed: {exc}. "
                f"Retry via POST /datasets/{dataset.id}/export.",
            ) from exc

    return _dataset_out(db, dataset)


@router.get("/{dataset_id}/history", response_model=DatasetHistoryOut)
def dataset_history(dataset_id: int, db: Session = Depends(get_db)):
    """Every source file merged into this dataset, oldest first: the file the
    dataset was created from, then each append with its note and how many of
    its rows were new vs already-held duplicates. All counts are the stored
    Upload-row numbers."""
    dataset = _get_dataset_or_404(db, dataset_id)
    uploads = _ensure_uploads(db, dataset)
    db.commit()  # persist any lazily synthesized Upload #1
    entries = [
        UploadHistoryEntry(
            upload_id=u.id,
            filename=u.stored_filename,
            uploaded_at=u.uploaded_at,
            kind="created" if i == 0 else "appended",
            row_count=u.row_count,
            new_row_count=u.new_row_count,
            duplicate_row_count=u.duplicate_row_count,
            note=u.note,
        )
        for i, u in enumerate(sorted(uploads, key=lambda u: u.row_offset))
    ]
    return DatasetHistoryOut(dataset_id=dataset.id, entries=entries)


@router.post("/{dataset_id}/columns", response_model=DatasetOut,
             dependencies=[Depends(require_admin)])
def select_columns(
    dataset_id: int, body: SelectColumnsRequest, db: Session = Depends(get_db)
):
    dataset = _get_dataset_or_404(db, dataset_id)
    if not body.questions:
        raise HTTPException(status_code=400, detail="Select at least one question column")

    uploads = _ensure_uploads(db, dataset)
    frames: list[tuple[Upload, pd.DataFrame]] = [
        (u, _read_upload_frame(u)) for u in uploads
    ]
    first_df = frames[0][1]

    all_selected = [q.column for q in body.questions]
    all_selected += [m.column for m in body.metadata_columns]
    if body.respondent_id_column:
        all_selected.append(body.respondent_id_column)
    missing = [c for c in all_selected if c not in first_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found in file: {missing}")

    # A column cannot be both a question and a demographic: the first is
    # reshaped into Responses and analysed, the second is an attribute of the
    # respondent. Selecting one as both is a mistake with silent consequences
    # (its text would be induced over AND offered as a filter value).
    overlap = sorted({q.column for q in body.questions}
                     & {m.column for m in body.metadata_columns})
    if overlap:
        raise HTTPException(
            status_code=400,
            detail=f"Columns selected as both a question and a demographic: {overlap}",
        )

    # Question columns must exist in every appended file too — a re-select
    # reshapes all uploads, and a column one file lacks has no rows to give.
    # The respondent-id column is lenient past the first file (append allows a
    # file without it; those rows carry respondent_id=None).
    question_cols = [q.column for q in body.questions]
    for upload, upload_df in frames[1:]:
        missing = [c for c in question_cols if c not in upload_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found in appended file "
                f"'{upload.stored_filename}': {missing}",
            )

    # A selected column with zero non-empty cells would silently become a
    # question with zero responses — that is never what the analyst meant.
    # Checked across the union of all uploads: empty in one file but answered
    # in another is fine.
    empty_cols = [
        q.column
        for q in body.questions
        if sum(int((f[q.column].str.strip() != "").sum()) for _, f in frames) == 0
    ]
    if empty_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Selected columns have no non-empty values: {empty_cols}",
        )

    dataset.respondent_id_column = body.respondent_id_column
    if body.dataset_description is not None:
        dataset.description = body.dataset_description.strip() or None
    # Catalog metadata, same None-preserves / strip-or-None semantics as the
    # description above.
    if body.dataset_department is not None:
        dataset.department = body.dataset_department.strip() or None
    if body.dataset_notes is not None:
        dataset.notes = body.dataset_notes.strip() or None
    if body.survey_start_date is not None:
        dataset.survey_start_date = body.survey_start_date.strip() or None
    if body.survey_end_date is not None:
        dataset.survey_end_date = body.survey_end_date.strip() or None
    # Period-labeling config: an explicit config always wins; otherwise a
    # newly selected date column gets quarter bucketing so it is useful with
    # zero setup (the analyst can re-cut periods later — read-time derivation
    # means that is a metadata edit, never a re-ingest).
    if body.date_ranges is not None:
        dataset.date_ranges_json = json.dumps(body.date_ranges)
    elif (
        any(m.value_type == "date" for m in body.metadata_columns)
        and not dataset.date_ranges_json
    ):
        dataset.date_ranges_json = json.dumps(
            {"mode": "bucket", "granularity": "quarter"}
        )

    # Upsert QuestionColumn by (dataset_id, source_column) instead of
    # delete-then-insert, so a column that stays selected across re-runs
    # keeps the same id — and therefore its Response rows keep theirs too.
    # Only columns dropped from the new selection get deleted (cascades to
    # their responses).
    existing_questions = {
        q.source_column: q
        for q in db.query(QuestionColumn)
        .filter(QuestionColumn.dataset_id == dataset.id)
        .all()
    }
    desired_columns = {q.column for q in body.questions}

    kept_or_created: dict[str, QuestionColumn] = {}
    for position, q in enumerate(body.questions):
        existing = existing_questions.get(q.column)
        if existing is not None:
            existing.label = q.label
            existing.position = position
            kept_or_created[q.column] = existing
        else:
            question = QuestionColumn(
                dataset_id=dataset.id,
                source_column=q.column,
                label=q.label,
                position=position,
            )
            db.add(question)
            db.flush()  # assign question.id
            kept_or_created[q.column] = question

    for source_column, existing in existing_questions.items():
        if source_column not in desired_columns:
            db.delete(existing)
    db.flush()

    # Metadata columns: same upsert-by-source_column contract as questions, so
    # a demographic that survives a re-select keeps its id and its stored
    # values. Dropped ones cascade their RespondentAttribute rows away.
    existing_meta = {
        m.source_column: m
        for m in db.query(MetadataColumn)
        .filter(MetadataColumn.dataset_id == dataset.id)
        .all()
    }
    desired_meta = {m.column for m in body.metadata_columns}
    meta_kept: dict[str, MetadataColumn] = {}
    for position, m in enumerate(body.metadata_columns):
        # cardinality measured across every upload, not just the first: a
        # column is only as coarse as the union of the waves makes it. Date
        # columns count distinct PARSED dates — the stored values — so a
        # column of junk that never parses is caught here, not discovered as
        # an empty filter later.
        distinct: set[str] = set()
        for _u, f in frames:
            if m.column in f.columns:
                raw_values = {
                    v.strip() for v in f[m.column].tolist() if str(v).strip()
                }
                if m.value_type == "date":
                    raw_values = {
                        iso
                        for iso in (dates_module.parse_date(v) for v in raw_values)
                        if iso
                    }
                distinct |= raw_values
        if m.value_type == "date" and not distinct:
            raise HTTPException(
                status_code=400,
                detail=f"Date column '{m.column}' has no parseable dates "
                "(expected e.g. 2023-09-19 or 9/19/2023)",
            )
        existing = existing_meta.get(m.column)
        if existing is not None:
            existing.label = m.label.strip()
            existing.position = position
            existing.n_distinct = len(distinct)
            existing.value_type = m.value_type
            meta_kept[m.column] = existing
        else:
            col = MetadataColumn(
                dataset_id=dataset.id,
                source_column=m.column,
                label=m.label.strip(),
                position=position,
                n_distinct=len(distinct),
                value_type=m.value_type,
            )
            db.add(col)
            db.flush()
            meta_kept[m.column] = col

    for source_column, existing in existing_meta.items():
        if source_column not in desired_meta:
            db.delete(existing)
    db.flush()

    # Upsert Response rows upload by upload, keyed (dataset_id, question_id,
    # source_row_index) so a row that survives a re-run keeps its id and
    # response_key. Stale deletion inside the helper is scoped per upload_id —
    # re-reading upload #1's file can never delete rows an appended file
    # contributed.
    for upload, upload_df in frames:
        _upsert_upload_responses(
            db, dataset, upload, upload_df, kept_or_created, body.respondent_id_column
        )
        _upsert_upload_metadata(db, dataset, upload, upload_df, meta_kept)

    dataset.status = "ingested"
    db.commit()
    db.refresh(dataset)

    try:
        _write_exports(db, dataset)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest committed to the database but export failed: {exc}. Retry via POST /datasets/{dataset.id}/export.",
        ) from exc

    return _dataset_out(db, dataset)


@router.post("/{dataset_id}/export", response_model=DatasetOut,
             dependencies=[Depends(require_admin)])
def export_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = _get_dataset_or_404(db, dataset_id)
    if dataset.status != "ingested":
        raise HTTPException(
            status_code=400,
            detail="Dataset has not been ingested yet — select columns first",
        )
    _write_exports(db, dataset)
    return _dataset_out(db, dataset)


@router.get("/{dataset_id}/exports/csv")
def download_csv(dataset_id: int, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    path = EXPORTS_DIR / str(dataset_id) / "responses.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV export not found — commit the ingest first")
    return FileResponse(path, media_type="text/csv", filename="responses.csv")


@router.get("/{dataset_id}/exports/parquet")
def download_parquet(dataset_id: int, db: Session = Depends(get_db)):
    _get_dataset_or_404(db, dataset_id)
    path = EXPORTS_DIR / str(dataset_id) / "responses.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Parquet export not found — commit the ingest first")
    return FileResponse(
        path, media_type="application/octet-stream", filename="responses.parquet"
    )
