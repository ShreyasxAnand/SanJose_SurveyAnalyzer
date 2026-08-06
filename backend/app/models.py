import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Dataset(Base):
    """A single uploaded survey file. The original file on disk is never
    modified after upload — reshaped data lives in separate tables that
    reference it."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    original_path: Mapped[str] = mapped_column(String, nullable=False)
    sheet_name: Mapped[str | None] = mapped_column(String, nullable=True)
    respondent_id_column: Mapped[str | None] = mapped_column(String, nullable=True)
    # Free-text survey description collected at column-select time. Purely
    # descriptive (what the survey is, who answered) — it feeds every prompt
    # via context_block and is hashed into run ids, so editing it re-versions
    # induction/labeling runs automatically.
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Catalog metadata: shown in the UI and written to the export manifest,
    # NEVER fed to prompts (unlike description above) — editing these must
    # not re-version any pipeline run. Dates are ISO "YYYY-MM-DD" strings
    # (SQLite stores text anyway; <input type="date"> emits exactly this).
    department: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    survey_start_date: Mapped[str | None] = mapped_column(String, nullable=True)
    survey_end_date: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="uploaded")
    uploaded_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    questions: Mapped[list["QuestionColumn"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    responses: Mapped[list["Response"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    uploads: Mapped[list["Upload"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="Upload.row_offset",
    )
    row_hashes: Mapped[list["RowHash"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Upload(Base):
    """One uploaded source file merged into a dataset. The first upload is the
    file the dataset was created from; later ones are appends whose duplicate
    rows were skipped. Each file's rows occupy the global row-index range
    [row_offset, row_offset + row_count), so appended rows can never collide
    with earlier ones and every response_key stays stable forever."""

    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String, nullable=False)
    stored_path: Mapped[str] = mapped_column(String, nullable=False)  # rel. REPO_ROOT
    sheet_name: Mapped[str | None] = mapped_column(String, nullable=True)
    row_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    new_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    duplicate_row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Analyst-supplied note collected at append time ("2026 Q3 wave", …);
    # None for uploads made before the field existed or left blank. Shown in
    # the dataset's history alongside the counts above.
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    dataset: Mapped["Dataset"] = relationship(back_populates="uploads")
    row_hashes: Mapped[list["RowHash"]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )
    column_fingerprints: Mapped[list["ColumnFingerprint"]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )


class RowHash(Base):
    """SHA-256 of one whole raw row of one uploaded file — the duplicate-
    detection record. Every raw row of every upload gets one (duplicates
    flagged rather than omitted, so what was skipped is auditable); match
    detection only counts rows with is_duplicate=False, which is exactly the
    multiset of rows the dataset actually ingested."""

    __tablename__ = "row_hashes"
    __table_args__ = (
        UniqueConstraint("dataset_id", "row_index", name="uq_row_hashes_dataset_row"),
        Index("ix_row_hashes_hash", "row_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"), nullable=False)
    # Global dataset row index = upload.row_offset + local index in the file.
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    row_hash: Mapped[str] = mapped_column(String, nullable=False)
    # True = this row was already in the dataset when its file was appended;
    # it produced no Response rows.
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="row_hashes")
    upload: Mapped["Upload"] = relationship(back_populates="row_hashes")


class ColumnFingerprint(Base):
    """SHA-256 of one column's full value sequence in one uploaded file — the
    column-level duplicate-detection record. When whole-row matching finds
    nothing (a re-upload with a column added/dropped/renamed changes every
    row hash), identical column fingerprints still identify the file as data
    the system already holds, so the analyst is warned before paying for a
    second pipeline run. Diagnostic only: it never enables append."""

    __tablename__ = "column_fingerprints"
    __table_args__ = (
        UniqueConstraint(
            "upload_id", "column_name", name="uq_column_fingerprints_upload_col"
        ),
        Index("ix_column_fingerprints_fp", "fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id"), nullable=False)
    column_name: Mapped[str] = mapped_column(String, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String, nullable=False)

    upload: Mapped["Upload"] = relationship(back_populates="column_fingerprints")


class QuestionColumn(Base):
    """One selected open-ended question column from the wide-format upload.
    Upserted by (dataset_id, source_column) on re-ingest so its id — and
    therefore the id of every Response that references it — stays stable
    across repeated column-selection runs on the same dataset."""

    __tablename__ = "question_columns"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "source_column", name="uq_question_columns_dataset_source"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    source_column: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0)

    dataset: Mapped["Dataset"] = relationship(back_populates="questions")
    responses: Mapped[list["Response"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class Response(Base):
    """One respondent's answer to one selected question — the reshaped,
    long-format unit the rest of the pipeline builds on.

    Identity is (dataset_id, question_id, source_row_index), also exposed as
    the deterministic `response_key` string. Re-ingest upserts against this
    key rather than deleting and reinserting, so a future labeling pass that
    keys off these ids/keys doesn't get orphaned by re-running column
    selection.

    `raw_text_original` is the exact cell value as read from the source
    file, before any cleaning. `response_text` is that value after mojibake
    repair — kept alongside, not overwritten, so repair is auditable and
    reversible."""

    __tablename__ = "responses"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "question_id",
            "source_row_index",
            name="uq_responses_dataset_question_row",
        ),
        UniqueConstraint("response_key", name="uq_responses_response_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("question_columns.id"), nullable=False
    )
    # Which uploaded file this row came from. Nullable only for rows created
    # before the uploads table existed (backfilled by scripts.backfill_uploads);
    # select_columns scopes its stale-row deletion by upload_id so re-reading
    # one file can never delete rows another file contributed.
    upload_id: Mapped[int | None] = mapped_column(
        ForeignKey("uploads.id"), nullable=True
    )
    source_row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    response_key: Mapped[str] = mapped_column(String, nullable=False)
    respondent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_text_original: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    was_encoding_repaired: Mapped[bool] = mapped_column(Boolean, default=False)
    # Sentinel non-answer ("n/a", ".", "idk", ...) — stored and exported like
    # any response (original data immutable), but flagged so induction and
    # labeling can skip it without re-deriving the predicate.
    is_nonanswer: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="responses")
    question: Mapped["QuestionColumn"] = relationship(back_populates="responses")
