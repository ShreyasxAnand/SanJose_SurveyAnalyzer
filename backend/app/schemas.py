import datetime as dt
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class ColumnPreview(BaseModel):
    column: str
    sample_values: list[str]
    non_null_count: int


class UploadResponse(BaseModel):
    dataset_id: int
    name: str
    original_filename: str
    row_count: int
    columns: list[ColumnPreview]


class QuestionColumnSelection(BaseModel):
    column: str
    label: str

    @model_validator(mode="after")
    def _label_must_be_real_wording(self) -> Self:
        label = self.label.strip()
        if not label:
            raise ValueError(f"Question wording is required for column '{self.column}'")
        if label.lower() == self.column.strip().lower():
            raise ValueError(
                f"Question wording for '{self.column}' must be the actual question "
                "text, not the raw column name"
            )
        return self


class SelectColumnsRequest(BaseModel):
    respondent_id_column: str | None = None
    questions: list[QuestionColumnSelection]


class QuestionColumnOut(BaseModel):
    id: int
    source_column: str
    label: str
    response_count: int

    model_config = ConfigDict(from_attributes=True)


class ExportInfo(BaseModel):
    csv_path: str
    parquet_path: str
    manifest_path: str
    csv_download_url: str
    parquet_download_url: str
    total_row_count: int
    per_question_counts: dict[str, int]


class DatasetOut(BaseModel):
    id: int
    name: str
    original_filename: str
    status: str
    uploaded_at: dt.datetime
    respondent_id_column: str | None
    questions: list[QuestionColumnOut]
    exports: ExportInfo | None = None

    model_config = ConfigDict(from_attributes=True)


class ResponseOut(BaseModel):
    id: int
    question_id: int
    source_row_index: int
    response_key: str
    respondent_id: str | None
    raw_text_original: str
    response_text: str
    was_encoding_repaired: bool

    model_config = ConfigDict(from_attributes=True)
