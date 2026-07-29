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

from app.db import DATA_DIR, EXPORTS_DIR, UPLOADS_DIR, get_db
from app.models import Dataset, QuestionColumn, Response
from app.nonanswer import is_nonanswer_text
from app.schemas import (
    ColumnPreview,
    DatasetOut,
    ExportInfo,
    QuestionColumnOut,
    SelectColumnsRequest,
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
EXPORT_SCHEMA_VERSION = 2
LIST_COLUMNS = ("label_ids", "locations", "time_context")
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
            }
        )

    parquet_path = export_dir / "responses.parquet"
    table = pa.Table.from_pylist(records, schema=RESPONSE_PARQUET_SCHEMA)
    pq.write_table(table, parquet_path)

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
        "column_mapping": [
            {"question_id": q.id, "source_column": q.source_column, "label": q.label}
            for q in questions
        ],
        "per_question_counts": per_question_counts,
        "per_question_nonanswer_counts": per_question_nonanswer_counts,
        "total_row_count": len(records),
        "encoding_repairs_applied": encoding_repairs_applied,
        # which labeling run each question's label columns came from, and how
        # many assignment keys had no DB row (stale run vs re-ingest) — never
        # dropped silently
        "labels_runs": labels_runs,
        "labels_unmatched_keys": len(labels_by_key) - matched_keys,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
    return DatasetOut(
        id=dataset.id,
        name=dataset.name,
        original_filename=dataset.original_filename,
        status=dataset.status,
        uploaded_at=dataset.uploaded_at,
        respondent_id_column=dataset.respondent_id_column,
        description=dataset.description,
        questions=questions,
        exports=exports,
    )


@router.post("/upload", response_model=UploadResponse)
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
    )


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    return [_dataset_out(db, d) for d in datasets]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = _get_dataset_or_404(db, dataset_id)
    return _dataset_out(db, dataset)


@router.post("/{dataset_id}/columns", response_model=DatasetOut)
def select_columns(
    dataset_id: int, body: SelectColumnsRequest, db: Session = Depends(get_db)
):
    dataset = _get_dataset_or_404(db, dataset_id)
    if not body.questions:
        raise HTTPException(status_code=400, detail="Select at least one question column")

    original_path = REPO_ROOT / dataset.original_path
    df = _read_dataframe(original_path)

    all_selected = [q.column for q in body.questions]
    if body.respondent_id_column:
        all_selected.append(body.respondent_id_column)
    missing = [c for c in all_selected if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found in file: {missing}")

    # A selected column with zero non-empty cells would silently become a
    # question with zero responses — that is never what the analyst meant.
    empty_cols = [
        q.column for q in body.questions
        if int((df[q.column].str.strip() != "").sum()) == 0
    ]
    if empty_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Selected columns have no non-empty values: {empty_cols}",
        )

    dataset.respondent_id_column = body.respondent_id_column
    if body.dataset_description is not None:
        dataset.description = body.dataset_description.strip() or None

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

    # Upsert Response by (dataset_id, question_id, source_row_index) for the
    # same reason. Every cell is re-cleaned on every run so edits to the
    # mojibake repair logic apply retroactively. Bulk mappings instead of
    # per-row ORM objects: at 30k rows x several questions the unit-of-work
    # bookkeeping is minutes, the bulk path is seconds. enumerate position
    # equals the old df.iterrows() index — these frames always carry the
    # default RangeIndex.
    respondent_ids = (
        df[body.respondent_id_column].tolist() if body.respondent_id_column else None
    )
    for column, question in kept_or_created.items():
        existing_ids = dict(
            db.query(Response.source_row_index, Response.id)
            .filter(
                Response.dataset_id == dataset.id, Response.question_id == question.id
            )
            .all()
        )
        inserts: list[dict] = []
        updates: list[dict] = []
        seen_rows: set[int] = set()

        for row_index, raw_text in enumerate(df[column].tolist()):
            if not raw_text.strip():
                continue
            seen_rows.add(row_index)

            cleaned_text, was_repaired = _repair_mojibake(raw_text.strip())
            fields = {
                "respondent_id": respondent_ids[row_index] if respondent_ids else None,
                "raw_text_original": raw_text,
                "response_text": cleaned_text,
                "was_encoding_repaired": was_repaired,
                "is_nonanswer": is_nonanswer_text(cleaned_text),
            }
            existing_id = existing_ids.get(row_index)
            if existing_id is not None:
                updates.append({"id": existing_id, **fields})
            else:
                inserts.append(
                    {
                        "dataset_id": dataset.id,
                        "question_id": question.id,
                        "source_row_index": row_index,
                        "response_key": _response_key(dataset.id, question.id, row_index),
                        **fields,
                    }
                )

        db.bulk_insert_mappings(Response, inserts)
        db.bulk_update_mappings(Response, updates)

        stale_ids = [rid for idx, rid in existing_ids.items() if idx not in seen_rows]
        # chunk the IN() list — SQLite's default parameter limit is 999
        for i in range(0, len(stale_ids), 900):
            db.query(Response).filter(
                Response.id.in_(stale_ids[i : i + 900])
            ).delete(synchronize_session=False)

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


@router.post("/{dataset_id}/export", response_model=DatasetOut)
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
