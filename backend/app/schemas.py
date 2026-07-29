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
    # Free-text survey description (what the survey is, who answered it —
    # never what the analyst hopes to find). Optional; None leaves any
    # previously saved description untouched.
    dataset_description: str | None = None


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
    description: str | None = None
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


# --- Phase 5: two-step ask (stateless — the browser carries the proposal) ---


class AskRouteRequest(BaseModel):
    question: str

    @model_validator(mode="after")
    def _question_required(self) -> Self:
        if not self.question.strip():
            raise ValueError("Question is required")
        return self


class AskCandidateOut(BaseModel):
    """One proposed category, enriched with everything the review screen
    shows: real count, names, and the model's rationale."""
    label_id: str
    name: str
    parent_name: str | None
    question_id: str
    question_text: str
    count: int
    relevance: str
    rationale: str


class AskLocationOut(BaseModel):
    """One location concept the review screen can offer as a place filter:
    kind separates named places from kinds of place, count is the real
    corpus-wide mention count (deterministic keyword sweep)."""
    name: str
    kind: str
    count: int


class AskRouteResponse(BaseModel):
    answerable: bool
    route: str
    reason: str
    candidates: list[AskCandidateOut]
    lexicon_concepts: list[str]
    available_lexicon_concepts: list[str]
    group_by: str = "category"
    location_filter: list[str] = []
    available_locations: list[AskLocationOut] = []
    # "" | "specific" | "general" — restrict the evidence to responses the
    # labeling pass marked as proposing a concrete action (or not).
    actionability_filter: str = ""
    # corpus-wide counts per value, so the review screen can show what the
    # filter would cost before the analyst ticks it. Empty when no labels run
    # carries the field, which is also how the UI knows to hide the control.
    available_actionability: dict[str, int] = {}
    # "" | "reported" — restrict to responses recounting a first-hand
    # incident. There is deliberately no negative direction: not describing an
    # incident is not evidence that none occurred.
    event_filter: str = ""
    # (responses reporting an incident, responses actually checked); absent
    # when no labels run carries the field, which is how the UI hides it
    available_events: dict[str, int] = {}
    warnings: list[str]


class AskSelectedCandidate(BaseModel):
    """A ticked checkbox coming back from the review screen. relevance and
    rationale are echoes of the proposal, kept for the audit manifest — the
    server recomputes everything that matters."""
    label_id: str
    relevance: str = "medium"
    rationale: str = ""


class AskAnswerRequest(BaseModel):
    question: str
    route: str
    reason: str = ""
    selected: list[AskSelectedCandidate]
    lexicon_concepts: list[str] = []
    group_by: str = "category"
    location_filter: list[str] = []
    actionability_filter: str = ""
    event_filter: str = ""
    proposed_label_ids: list[str] = []

    @model_validator(mode="after")
    def _selection_required(self) -> Self:
        if not self.question.strip():
            raise ValueError("Question is required")
        if not self.selected:
            raise ValueError("Select at least one category")
        return self


class AskSourceOut(BaseModel):
    n: int
    response_key: str
    label_id: str
    text: str
    location: str | None = None


class AskAnswerStats(BaseModel):
    """Computed facts about how the answer was assembled — every figure is
    counted from the run's own data, so the UI's method strip can never
    disagree with what actually happened."""
    categories_searched: int
    categories_total: int
    unique_responses: int
    quotes_shown: int
    quotes_cited: int


class AskLexiconCountOut(BaseModel):
    concept: str
    mentions_total: int
    mentions_by_question: dict[str, int]


class AskGroupCountOut(BaseModel):
    name: str
    count_unique_responses: int


class AskAnswerResponse(BaseModel):
    run_id: str
    # the answer body only — sources arrive structured in `sources`, and the
    # UI renders them itself (answer.md on disk keeps the embedded section)
    answer_markdown: str
    process_note: str
    stats: AskAnswerStats
    sources: list[AskSourceOut]
    counts: dict[str, int]
    # with a location_filter, `counts` are filtered; this holds each
    # category's full size so the UI can show "n of N in category"
    counts_unfiltered: dict[str, int] = {}
    sampling_notes: dict[str, str]
    lexicon_counts: list[AskLexiconCountOut] = []
    group_counts: list[AskGroupCountOut] = []
    location_filter: list[str] = []
    location_counts: list[AskLocationOut] = []
    location_denominator: dict[str, int] | None = None
    actionability_filter: str = ""
    # {in_scope, coded, matching} — how many responses the filter kept, out of
    # how many it was applied to. `coded` < `in_scope` means some responses
    # were never marked either way: missing data, not evidence of absence.
    actionability_denominator: dict[str, int] | None = None
    event_filter: str = ""
    # {in_scope, coded, matching} — `matching` responses described an
    # incident. The remainder did not describe one; that is not the same as
    # nothing having happened, and the UI must not present it as such.
    event_denominator: dict[str, int] | None = None
    invalid_citations: int
    deselected: list[str]
    added: list[str]
