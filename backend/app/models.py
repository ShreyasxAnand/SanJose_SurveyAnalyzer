import datetime as dt

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
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

    @property
    def response_count(self) -> int:
        return len(self.responses)


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
    source_row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    response_key: Mapped[str] = mapped_column(String, nullable=False)
    respondent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_text_original: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    was_encoding_repaired: Mapped[bool] = mapped_column(Boolean, default=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="responses")
    question: Mapped["QuestionColumn"] = relationship(back_populates="responses")
