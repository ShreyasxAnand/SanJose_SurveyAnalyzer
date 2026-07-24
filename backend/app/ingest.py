import datetime as dt
import io
import json
import shutil
import uuid
from pathlib import Path
from typing import IO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import DATA_DIR, EXPORTS_DIR, UPLOADS_DIR, get_db
from app.models import Dataset, QuestionColumn, Response
from app.schemas import (
    ColumnPreview,
    DatasetOut,
    ExportInfo,
    SelectColumnsRequest,
    UploadResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
REPO_ROOT = DATA_DIR.parent

# Encodings a UTF-8 file commonly gets mis-decoded through, producing
# mojibake like "donâ€™t" for "don't". cp1252 first since it's the more
# common real-world corruption (Excel/Windows text handling); latin-1 as
# fallback.
_MOJIBAKE_ENCODINGS = ("cp1252", "latin-1")

RESPONSE_PARQUET_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.int64()),
        ("question_id", pa.int64()),
        ("question_label", pa.string()),
        ("source_column", pa.string()),
        ("source_row_index", pa.int64()),
        ("response_key", pa.string()),
        ("respondent_id", pa.string()),
        ("response_text", pa.string()),
        ("raw_text_original", pa.string()),
        ("was_encoding_repaired", pa.bool_()),
        # Phase-3 fields — always null/empty until labeling exists. Typed now
        # so the labeling pass can read and rewrite this file in place later.
        ("sentiment", pa.string()),
        ("child_category_ids", pa.list_(pa.string())),
        ("locations", pa.list_(pa.string())),
        ("uncategorized", pa.bool_()),
    ]
)


def _parse_dataframe(buffer: IO[bytes], suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(buffer, dtype=str, keep_default_na=False)
    return pd.read_excel(buffer, dtype=str, keep_default_na=False)


def _read_dataframe(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        return _parse_dataframe(f, path.suffix.lower())


def _repair_mojibake(text: str) -> tuple[str, bool]:
    """Undo UTF-8 bytes that were previously mis-decoded as cp1252/latin-1.
    Tries up to 3 passes to catch doubly mangled text. A round-trip through
    one of these single-byte encodings only changes the text when the bytes
    happen to form valid UTF-8 on decode, which real, non-mangled text
    essentially never does by chance — so this is safe against false
    positives in practice. Returns (repaired_text, was_repaired)."""
    repaired = text
    was_repaired = False
    for _ in range(3):
        candidate = None
        for enc in _MOJIBAKE_ENCODINGS:
            try:
                attempt = repaired.encode(enc).decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
            if attempt != repaired:
                candidate = attempt
                break
        if candidate is None:
            break
        repaired = candidate
        was_repaired = True
    return repaired, was_repaired


def _response_key(dataset_id: int, question_id: int, source_row_index: int) -> str:
    return f"{dataset_id}:{question_id}:{source_row_index}"


def _get_dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def _write_exports(db: Session, dataset: Dataset) -> ExportInfo:
    """Regenerate responses.parquet, responses.csv, and manifest.json from
    the current database state. Postgres/SQLite stays the source of truth —
    these files are reproducible exports, safe to delete and regenerate.

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
    responses = (
        db.query(Response)
        .filter(Response.dataset_id == dataset.id)
        .order_by(Response.question_id, Response.source_row_index)
        .all()
    )

    per_question_counts: dict[str, int] = {q.label: 0 for q in questions}
    encoding_repairs_applied = 0
    records = []
    for r in responses:
        q = question_by_id[r.question_id]
        per_question_counts[q.label] += 1
        if r.was_encoding_repaired:
            encoding_repairs_applied += 1
        records.append(
            {
                "dataset_id": dataset.id,
                "question_id": r.question_id,
                "question_label": q.label,
                "source_column": q.source_column,
                "source_row_index": r.source_row_index,
                "response_key": r.response_key,
                "respondent_id": r.respondent_id,
                "response_text": r.response_text,
                "raw_text_original": r.raw_text_original,
                "was_encoding_repaired": r.was_encoding_repaired,
                "sentiment": None,
                "child_category_ids": [],
                "locations": [],
                "uncategorized": None,
            }
        )

    parquet_path = export_dir / "responses.parquet"
    table = pa.Table.from_pylist(records, schema=RESPONSE_PARQUET_SCHEMA)
    pq.write_table(table, parquet_path)

    csv_path = export_dir / "responses.csv"
    csv_records = [
        {
            **rec,
            "child_category_ids": json.dumps(rec["child_category_ids"]),
            "locations": json.dumps(rec["locations"]),
        }
        for rec in records
    ]
    pd.DataFrame(
        csv_records, columns=[f.name for f in RESPONSE_PARQUET_SCHEMA]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "dataset_id": dataset.id,
        "source_filename": dataset.original_filename,
        "sheet": dataset.sheet_name,
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "respondent_id_column": dataset.respondent_id_column,
        "column_mapping": [
            {"question_id": q.id, "source_column": q.source_column, "label": q.label}
            for q in questions
        ],
        "per_question_counts": per_question_counts,
        "total_row_count": len(records),
        "encoding_repairs_applied": encoding_repairs_applied,
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
    out = DatasetOut.model_validate(dataset)
    exports = _load_export_info(dataset.id)
    out.exports = exports
    # Once exports exist, the manifest is the audited record of what was
    # actually written to disk — prefer it over the live DB count so the
    # API response and the manifest can never disagree.
    if exports is not None:
        for q in out.questions:
            if q.label in exports.per_question_counts:
                q.response_count = exports.per_question_counts[q.label]
    return out


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

    dataset.respondent_id_column = body.respondent_id_column

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
    # same reason. Every cell is re-cleaned on every run (cheap at this
    # scale) so edits to the mojibake repair logic apply retroactively.
    for column, question in kept_or_created.items():
        existing_responses = {
            r.source_row_index: r
            for r in db.query(Response)
            .filter(
                Response.dataset_id == dataset.id, Response.question_id == question.id
            )
            .all()
        }
        seen_rows: set[int] = set()

        for row_index, row in df.iterrows():
            raw_text = row[column]
            if not raw_text.strip():
                continue
            row_index = int(row_index)
            seen_rows.add(row_index)

            cleaned_text, was_repaired = _repair_mojibake(raw_text.strip())
            respondent_id = (
                row[body.respondent_id_column] if body.respondent_id_column else None
            )

            existing_response = existing_responses.get(row_index)
            if existing_response is not None:
                existing_response.raw_text_original = raw_text
                existing_response.response_text = cleaned_text
                existing_response.was_encoding_repaired = was_repaired
                existing_response.respondent_id = respondent_id
            else:
                db.add(
                    Response(
                        dataset_id=dataset.id,
                        question_id=question.id,
                        source_row_index=row_index,
                        response_key=_response_key(dataset.id, question.id, row_index),
                        respondent_id=respondent_id,
                        raw_text_original=raw_text,
                        response_text=cleaned_text,
                        was_encoding_repaired=was_repaired,
                    )
                )

        for row_index, existing_response in existing_responses.items():
            if row_index not in seen_rows:
                db.delete(existing_response)

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
