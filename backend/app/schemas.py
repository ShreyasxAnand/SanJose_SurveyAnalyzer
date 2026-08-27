import datetime as dt
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from app import dates as dates_module


def _validate_survey_dates(start: str | None, end: str | None) -> None:
    """Shared check for the survey date-range fields: None = leave unchanged,
    "" = clear, anything else must be an ISO YYYY-MM-DD date, and a range
    where both ends are real dates must not run backwards."""
    parsed: dict[str, dt.date] = {}
    for field, value in (("survey_start_date", start), ("survey_end_date", end)):
        if value is None or not value.strip():
            continue
        try:
            parsed[field] = dt.date.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(
                f"{field} must be an ISO date (YYYY-MM-DD), got '{value}'"
            ) from None
    if len(parsed) == 2 and parsed["survey_start_date"] > parsed["survey_end_date"]:
        raise ValueError("survey_start_date must not be after survey_end_date")


class ColumnPreview(BaseModel):
    column: str
    sample_values: list[str]
    non_null_count: int


class DatasetMatch(BaseModel):
    """How much of an uploaded file's row-hash multiset an existing ingested
    dataset already holds. All counts are computed over stored hashes — never
    estimated."""
    dataset_id: int
    dataset_name: str
    matched_rows: int
    file_rows: int
    dataset_rows: int
    exact: bool
    # append is only possible when every question column the target selected
    # exists in the new file; when it isn't, the missing ones are named
    columns_compatible: bool
    missing_columns: list[str] = []


class RenamedColumn(BaseModel):
    stored_name: str
    file_name: str


class ColumnMatch(BaseModel):
    """Same column contents, different column set: an ingested dataset one of
    whose uploaded files shares most of its columns byte-for-byte (by full-
    column fingerprint) with the new file, even though no whole row matches
    because a column was added/dropped/renamed. Diagnostic only — append is
    impossible (row identity differs); the point is warning the analyst
    before they pay to re-process data the system already holds."""

    dataset_id: int
    dataset_name: str
    upload_filename: str
    upload_rows: int
    matched_columns: list[str]
    renamed_columns: list[RenamedColumn] = []
    missing_columns: list[str] = []  # in the stored file, absent from this one
    added_columns: list[str] = []  # new in this file
    changed_columns: list[str] = []  # same name, different content


class DuplicateCheck(BaseModel):
    outcome: str  # "none" | "exact" | "partial"
    best_dataset_id: int | None = None
    matches: list[DatasetMatch] = []
    # populated only when outcome == "none" — row evidence is stronger, so
    # column-level diagnosis only runs when row matching came up empty
    column_matches: list[ColumnMatch] = []


class UploadResponse(BaseModel):
    dataset_id: int
    name: str
    original_filename: str
    row_count: int
    columns: list[ColumnPreview]
    duplicate_check: DuplicateCheck = DuplicateCheck(outcome="none")


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


class MetadataValueCount(BaseModel):
    value: str
    n_respondents: int


class MetadataColumnSelection(BaseModel):
    """A demographic / respondent-attribute column. Unlike a question column
    the label MAY equal the raw column name — "District" is already the right
    display name, whereas a question column needs the actual question wording
    (a raw header like "Q2oe" tells a reader nothing).

    value_type "date" marks a response-date column ("stopdate"): cells are
    parsed to ISO at ingest (unparseable → no row) and the ask layer offers
    derived period labels instead of raw dates — see app/dates.py."""
    column: str
    label: str
    value_type: Literal["categorical", "date"] = "categorical"

    @model_validator(mode="after")
    def _label_required(self) -> Self:
        if not self.label.strip():
            raise ValueError(f"A name is required for metadata column '{self.column}'")
        return self


class SelectColumnsRequest(BaseModel):
    respondent_id_column: str | None = None
    questions: list[QuestionColumnSelection]
    # Demographic columns. Optional and independent of `questions`: a dataset
    # with none behaves exactly as before. Not named `metadata` — that shadows
    # BaseModel internals and reads ambiguously next to dataset_notes etc.
    metadata_columns: list[MetadataColumnSelection] = []
    # Free-text survey description (what the survey is, who answered it —
    # never what the analyst hopes to find). Optional; None leaves any
    # previously saved description untouched.
    dataset_description: str | None = None
    # Catalog metadata — displayed/edited in the UI and written to the export
    # manifest; NEVER fed to prompts (unlike dataset_description). Same
    # semantics as the description: None preserves the stored value, "" (or
    # whitespace) clears it. Dates are ISO YYYY-MM-DD.
    dataset_department: str | None = None
    dataset_notes: str | None = None
    survey_start_date: str | None = None
    survey_end_date: str | None = None
    # Period-labeling config for date-typed metadata columns (app/dates.py
    # shapes). None = keep whatever is stored (or, when a date column is
    # newly selected with nothing stored, default to quarter bucketing).
    date_ranges: dict | None = None

    @model_validator(mode="after")
    def _dates_are_valid(self) -> Self:
        _validate_survey_dates(self.survey_start_date, self.survey_end_date)
        if self.date_ranges is not None:
            self.date_ranges = dates_module.validate_ranges_config(self.date_ranges)
        return self


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


class MetadataColumnOut(BaseModel):
    id: int
    source_column: str
    label: str
    value_type: str = "categorical"
    n_distinct: int
    # value -> respondent count, most common first. Capped for transport; the
    # full set always lives in the database.
    values: list[MetadataValueCount] = []
    # True when n_distinct exceeds METADATA_MAX_DISTINCT. Advisory: the column
    # is stored and usable either way — see docs/DEMOGRAPHICS_PLAN.md §3.
    high_cardinality: bool = False

    model_config = ConfigDict(from_attributes=True)


class DatasetOut(BaseModel):
    id: int
    name: str
    original_filename: str
    status: str
    uploaded_at: dt.datetime
    respondent_id_column: str | None
    description: str | None = None
    department: str | None = None
    notes: str | None = None
    survey_start_date: str | None = None
    survey_end_date: str | None = None
    # Validated period-labeling config (app/dates.py shapes); None when the
    # dataset has no date-typed metadata column or nothing was configured.
    date_ranges: dict | None = None
    questions: list[QuestionColumnOut]
    metadata_columns: list[MetadataColumnOut] = []
    exports: ExportInfo | None = None

    model_config = ConfigDict(from_attributes=True)


class DatasetMetadataPatch(BaseModel):
    """Editable catalog metadata. Every field optional: None = leave
    unchanged, "" = clear (except name, which is non-nullable — blank is a
    422). The description is deliberately NOT here: it is part of every
    analysis prompt and each run's identity (prompt_hash), so it is only
    settable at column-select time; an unknown "description" key in a PATCH
    body is ignored by pydantic and changes nothing."""

    name: str | None = None
    department: str | None = None
    notes: str | None = None
    survey_start_date: str | None = None
    survey_end_date: str | None = None
    # Period-labeling config (app/dates.py shapes). None = leave unchanged.
    # Editable here because period labels are presentation config derived at
    # read time — changing them re-exports but never re-versions a run.
    date_ranges: dict | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.name is not None and not self.name.strip():
            raise ValueError("Dataset name cannot be blank")
        _validate_survey_dates(self.survey_start_date, self.survey_end_date)
        if self.date_ranges is not None:
            self.date_ranges = dates_module.validate_ranges_config(self.date_ranges)
        return self


class UploadHistoryEntry(BaseModel):
    """One source file merged into a dataset, straight off its Upload row —
    every count is a stored number, nothing derived."""

    upload_id: int
    filename: str
    uploaded_at: dt.datetime | None
    kind: str  # "created" (the dataset's first file) | "appended"
    row_count: int
    new_row_count: int
    duplicate_row_count: int
    note: str | None = None


class DatasetHistoryOut(BaseModel):
    dataset_id: int
    entries: list[UploadHistoryEntry]


class AppendRequest(BaseModel):
    # the provisional dataset created by the upload whose file is being
    # appended; consumed (deleted) on success
    upload_dataset_id: int
    # free-text history note ("2026 Q3 wave", "late responses from dept X");
    # stored on the Upload row and shown in the dataset's history. Optional —
    # forcing text would invent content.
    note: str | None = None


class AppendResponse(BaseModel):
    dataset: DatasetOut
    upload_id: int
    appended_rows: int
    skipped_duplicates: int
    new_responses_per_question: dict[str, int]
    warnings: list[str] = []


# --- Phase 5: two-step ask (stateless — the browser carries the proposal) ---


class AskRouteRequest(BaseModel):
    question: str
    # Survey question ids the ask is restricted to. Enforced at the proposal:
    # the router only sees the scoped questions' summary, so an out-of-scope
    # category cannot be proposed. Empty = all questions.
    question_scope: list[str] = []
    # Analyst-picked demographic restriction, {field: [values]} — the same
    # kind of scope selection as question_scope, never something the router
    # proposes. Values within one field are OR, fields are AND. Unknown
    # fields/values are a 422. Empty = everyone.
    demographic_filter: dict[str, list[str]] = {}

    @model_validator(mode="after")
    def _question_required(self) -> Self:
        if not self.question.strip():
            raise ValueError("Question is required")
        return self


class AskQuestionOut(BaseModel):
    """One survey question, for the scope selector on the ask form."""
    question_id: str
    question_text: str
    n_responses: int


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


class AskDemographicValueOut(BaseModel):
    """One value of a demographic field, with the number of RESPONDENTS
    carrying it — a demographic belongs to the person, not the response."""
    value: str
    n_respondents: int


class AskDemographicOut(BaseModel):
    """One demographic field the ask form offers as a respondent filter.
    Values are the dataset's actual non-blank values; a respondent who left
    the cell blank is missing data and appears under no value."""
    field: str
    values: list[AskDemographicValueOut]


class AskDemographicsRequest(BaseModel):
    """The ask form's current selection, so the counts come back faceted:
    each field recounted under the OTHER fields' ticked values. Empty = the
    unfiltered counts."""
    demographic_filter: dict[str, list[str]] = {}


class AskDemographicsResponse(BaseModel):
    fields: list[AskDemographicOut]
    # respondents matching the whole current filter; None when no filter is
    # active (the form shows it as a running "who am I asking about" figure)
    n_matching_respondents: int | None = None


class AskChildOut(BaseModel):
    """One child category on the review screen, proposed or not. `proposed`
    is what the router picked; the analyst can tick anything, and adding an
    unproposed category is logged the same way a deselection is — it is the
    router's recall signal, the mirror of the miscoding signal."""
    label_id: str
    name: str
    count: int
    description: str = ""
    proposed: bool = False
    # only meaningful when proposed — the router's reason for picking it
    relevance: str = ""
    rationale: str = ""


class AskParentGroupOut(BaseModel):
    """A top-level parent and its children, for one survey question.

    `count_unique_responses` is a real union over the children's response
    keys, computed in code — NOT the sum of child counts, which would
    double-count every multi-label response and invent a number no operation
    over the data produced."""
    question_id: str
    question_text: str
    parent_name: str
    count_unique_responses: int
    n_proposed: int
    children: list[AskChildOut]


class AskRouteResponse(BaseModel):
    # True when this proposal was served from the persistent ask cache —
    # identical question against identical data returns the stored routing
    # instead of re-rolling the model
    cached: bool = False
    answerable: bool
    route: str
    reason: str
    # route "aggregate_direct" answers from a coded tally with no synthesis
    # call; this names the tally ("location" | "time" | "event"). Empty for
    # every other route. Candidates may legitimately be empty on this route.
    aggregate_target: str = ""
    candidates: list[AskCandidateOut]
    # The whole taxonomy, grouped question > parent > child, so the review
    # screen can show every category rather than only the proposed ones. The
    # proposal is still the default selection; this is what makes it editable
    # in both directions instead of deselect-only.
    available_categories: list[AskParentGroupOut] = []
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
    # "" | "day" | "night" — restrict to responses whose verbatim time
    # mentions classify as day/night. No filter for responses naming no time.
    time_filter: str = ""
    # {"day": n, "night": n, "mentioned": n}; empty when nothing classifies,
    # which is how the UI hides the control
    available_time: dict[str, int] = {}
    # echo of the analyst-requested scope this proposal was made under
    question_scope: list[str] = []
    # echo of the analyst's demographic restriction (validated); the answer
    # step must carry it back unchanged for the filter to apply
    demographic_filter: dict[str, list[str]] = {}
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
    time_filter: str = ""
    question_scope: list[str] = []
    # {field: [values]} — validated server-side against the dataset's actual
    # demographics; part of the answer-cache key via ask_cache
    demographic_filter: dict[str, list[str]] = {}
    proposed_label_ids: list[str] = []
    aggregate_target: str = ""

    @model_validator(mode="after")
    def _selection_required(self) -> Self:
        if not self.question.strip():
            raise ValueError("Question is required")
        # aggregate_direct tallies the whole scope; categories only narrow it
        if not self.selected and self.route != "aggregate_direct":
            raise ValueError("Select at least one category")
        return self


class AskSourceOut(BaseModel):
    n: int
    response_key: str
    label_id: str
    text: str
    location: str | None = None
    # sub-themes this response was coded to (within its category) and the
    # survey question it answered — the same tags the quotes are grouped
    # under in the synthesis prompt
    subs: list[str] = []
    question_id: str = ""


class AskAnswerStats(BaseModel):
    """Computed facts about how the answer was assembled — every figure is
    counted from the run's own data, so the UI's method strip can never
    disagree with what actually happened."""
    categories_searched: int
    categories_total: int
    unique_responses: int
    quotes_shown: int
    quotes_cited: int
    # coverage guardrails: how much of the scoped questions' coded responses
    # the searched categories cover, and whether the base is small enough
    # (< ~200) that the UI should warn prominently
    scope_total: int = 0
    scope_coverage: float | None = None
    small_base: bool = False


class AskLexiconCountOut(BaseModel):
    concept: str
    mentions_total: int
    mentions_by_question: dict[str, int]


class AskGroupCountOut(BaseModel):
    name: str
    count_unique_responses: int


class AskSubCountOut(BaseModel):
    sub_label_id: str
    name: str
    count: int


class AskUncoveredOut(BaseModel):
    """An in-scope category the answer did NOT search — surfaced when the
    searched categories cover less than the coverage floor, so under-coverage
    is visible in the UI instead of only in a footnote."""
    label_id: str
    name: str
    count: int


class AskSubBreakdownOut(BaseModel):
    """Full-coverage sub-theme composition of one selected category, computed
    over the same filtered membership as the category's own count. `generic`
    responses raise the category without naming a specific sub-theme; members
    the sub-pass never coded are in neither figure (missing data). Sub-counts
    can sum past the category count — a response may raise several."""
    sub_counts: list[AskSubCountOut]
    generic: int
    coded: int


class AskAnswerResponse(BaseModel):
    # True when served from the persistent ask cache: the identical request
    # against identical data returns the ORIGINAL stored answer (same
    # run_id), byte-for-byte, with zero model calls
    cached: bool = False
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
    # label_id -> full-coverage sub-theme breakdown, for categories the
    # sub-theme layer has coded (app.subthemes); absent otherwise
    sub_breakdowns: dict[str, AskSubBreakdownOut] = {}
    # populated only when coverage fell below the floor: the largest in-scope
    # categories this answer does not cover
    uncovered_categories: list[AskUncoveredOut] = []
    # aggregate_direct answers only: the computed tally itself (target,
    # in_scope, per-place/time/event counts) as structured data
    aggregate: dict | None = None
    # {checked, violations} — the answer's inspection record: numbers traced
    # to computed counts, quoted spans checked against their cited sources,
    # quotes checked against the sub-theme they are cited under, sections
    # against their plan count. Findings are DISCLOSED; the answer is never
    # rewritten. None for deterministic tallies (nothing model-written).
    verification: dict | None = None
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
    time_filter: str = ""
    # {in_scope, mentioning, matching} — `matching` responses explicitly
    # mentioned the filtered time of day; responses naming no time say
    # nothing about when their experience happened.
    time_denominator: dict[str, int] | None = None
    demographic_filter: dict[str, list[str]] = {}
    # {in_scope, coded, matching} — `coded` responses belong to respondents
    # with a recorded value for every filtered field; the gap to in_scope is
    # missing data, never a group.
    demographic_denominator: dict[str, int] | None = None
    # True when the demographic filter left fewer matching responses than the
    # thin-cell notice threshold — disclosed, never blocked
    demographic_thin: bool = False
    # survey questions whose every response counted as mentioning the
    # filtered place because the question itself asks about it
    location_filter_implicit_questions: list[str] = []
    invalid_citations: int
    deselected: list[str]
    added: list[str]


# ---------------------------------------------------------------------------
# Pipeline runs (induce -> label -> lexicon -> locations), driven from the UI
# ---------------------------------------------------------------------------


class PipelineEstimateItem(BaseModel):
    """One row of the pre-run plan. `basis` is load-bearing: "planned" means
    the figure comes from the real chunking and prompt sizes the run will use,
    "projected" means it is extrapolated from a measured rate because the real
    dry-run needs an artifact that does not exist yet. The UI must keep the two
    distinguishable — a projection shown as a plan is an invented number."""
    stage: str
    question_id: str = ""
    question_text: str = ""
    detail: str = ""
    responses: int = 0
    est_cost_usd: float
    basis: str


class PipelineEstimate(BaseModel):
    dataset_id: str
    dataset_description: str = ""
    # "full" re-runs everything; "incremental" (post-append) only touches
    # never-labeled rows. Its items are basis="projected" — the pool size is a
    # run-time fact — except a brand-new column with no taxonomy, whose full
    # induction is planned exactly as it is in full mode.
    mode: str = "full"
    n_questions: int
    n_responses: int
    # rows the latest labels runs have never seen (incremental mode only)
    n_new_responses: int = 0
    items: list[PipelineEstimateItem]
    est_total_usd: float
    # questions that already have a taxonomy: running again adds a new
    # versioned run rather than doing nothing, so the analyst should know
    questions_with_existing_taxonomy: list[str] = []


class PipelineStageOut(BaseModel):
    key: str
    label: str
    status: str          # pending | running | done | failed | skipped
    detail: str = ""
    cost_usd: float | None = None
    seconds: float | None = None
    error: str = ""


class PipelineJobOut(BaseModel):
    job_id: str
    dataset_id: str
    status: str          # running | done | failed
    stages: list[PipelineStageOut]
    created_utc: str = ""
    finished_utc: str = ""
    error: str = ""
    # summed from each stage's own run manifest — real spend, not the estimate
    cost_usd: float = 0.0
    # whether the Ask tab will work for this dataset now
    is_processed: bool = False


class PipelineRunRequest(BaseModel):
    # 60 is the reliable labeling batch size; 80 produced deterministic
    # malformed-JSON failures on the 30k run
    batch_size: int = 60
    # "full" | "incremental" — incremental is the post-append path: label only
    # never-labeled rows, extend the taxonomy from the uncovered pool
    mode: str = "full"
