export interface ColumnPreview {
  column: string;
  sample_values: string[];
  non_null_count: number;
}

export interface DatasetMatch {
  dataset_id: number;
  dataset_name: string;
  matched_rows: number;
  file_rows: number;
  dataset_rows: number;
  exact: boolean;
  // append is only possible when every question column the target selected
  // exists in the new file
  columns_compatible: boolean;
  missing_columns: string[];
}

export interface RenamedColumn {
  stored_name: string;
  file_name: string;
}

// Same column contents, different column set — no row can match (a
// dropped/added/renamed column changes every row hash) but the untouched
// columns fingerprint identically. Diagnostic only: append is impossible,
// the point is warning before re-processing data the system already holds.
export interface ColumnMatch {
  dataset_id: number;
  dataset_name: string;
  upload_filename: string;
  upload_rows: number;
  matched_columns: string[];
  renamed_columns: RenamedColumn[];
  missing_columns: string[]; // in the stored file, absent from this one
  added_columns: string[]; // new in this file
  changed_columns: string[]; // same name, different content
}

export interface DuplicateCheck {
  outcome: "none" | "exact" | "partial";
  best_dataset_id: number | null;
  matches: DatasetMatch[];
  // populated only when outcome === "none"
  column_matches: ColumnMatch[];
}

export interface UploadResponse {
  dataset_id: number;
  name: string;
  original_filename: string;
  row_count: number;
  columns: ColumnPreview[];
  duplicate_check: DuplicateCheck;
}

export interface AppendResponse {
  dataset: DatasetOut;
  upload_id: number;
  appended_rows: number;
  skipped_duplicates: number;
  new_responses_per_question: Record<string, number>;
  warnings: string[];
}

export interface QuestionColumnOut {
  id: number;
  source_column: string;
  label: string;
  response_count: number;
}

export interface ExportInfo {
  csv_path: string;
  parquet_path: string;
  manifest_path: string;
  csv_download_url: string;
  parquet_download_url: string;
  total_row_count: number;
  per_question_counts: Record<string, number>;
}

export interface DatasetOut {
  id: number;
  name: string;
  original_filename: string;
  status: string;
  uploaded_at: string;
  respondent_id_column: string | null;
  description: string | null;
  // Catalog metadata — never used in analysis prompts (unlike description).
  department: string | null;
  notes: string | null;
  survey_start_date: string | null; // ISO YYYY-MM-DD
  survey_end_date: string | null;
  questions: QuestionColumnOut[];
  metadata_columns: MetadataColumnOut[];
  exports: ExportInfo | null;
}

export interface MetadataValueCount {
  value: string;
  n_respondents: number;
}

// A demographic / respondent-attribute column. `high_cardinality` is advisory
// only — the column is stored and usable either way (no hard blocks).
export interface MetadataColumnOut {
  id: number;
  source_column: string;
  label: string;
  n_distinct: number;
  values: MetadataValueCount[];
  high_cardinality: boolean;
}

// Editable catalog metadata. Omitted/null field = leave unchanged, "" =
// clear. The description is deliberately absent: it feeds every analysis
// prompt and is part of each run's identity, so it is fixed at ingest.
export interface DatasetMetadataPatch {
  name?: string;
  department?: string;
  notes?: string;
  survey_start_date?: string;
  survey_end_date?: string;
}

// One source file merged into a dataset — all counts are stored numbers.
export interface UploadHistoryEntry {
  upload_id: number;
  filename: string;
  uploaded_at: string | null;
  kind: "created" | "appended";
  row_count: number;
  new_row_count: number;
  duplicate_row_count: number;
  note: string | null;
}

export interface DatasetHistoryOut {
  dataset_id: number;
  entries: UploadHistoryEntry[];
}

// --- Phase 5: two-step ask ---

export interface AskCandidate {
  label_id: string;
  name: string;
  parent_name: string | null;
  question_id: string;
  question_text: string;
  count: number;
  relevance: string;
  rationale: string;
}

export interface AskLocation {
  name: string;
  kind: string; // "named" (a specific place) | "type" (a kind of place)
  count: number;
}

export interface AskChild {
  label_id: string;
  name: string;
  count: number;
  description: string;
  // whether the router proposed it; relevance and rationale are only
  // meaningful when it did
  proposed: boolean;
  relevance: string;
  rationale: string;
}

export interface AskParentGroup {
  question_id: string;
  question_text: string;
  parent_name: string;
  // a real union over the children's response keys, not the sum of their
  // counts — a response labelled with two children of one parent is one
  // response
  count_unique_responses: number;
  n_proposed: number;
  children: AskChild[];
}

export interface AskRouteResponse {
  // served from the persistent ask cache — same question, same data, same
  // routing, no model call
  cached: boolean;
  answerable: boolean;
  route: string;
  reason: string;
  // route "aggregate_direct" answers from a coded tally ("location" |
  // "time" | "event") with no synthesis call; candidates may be empty
  aggregate_target: string;
  candidates: AskCandidate[];
  // the whole taxonomy, so the review screen shows every parent category with
  // its children rather than only what the router proposed
  available_categories: AskParentGroup[];
  lexicon_concepts: string[];
  available_lexicon_concepts: string[];
  group_by: string; // "category" | "location"
  location_filter: string[];
  available_locations: AskLocation[];
  // "" | "specific" | "general" — restrict evidence to responses marked as
  // proposing a concrete action (or as raising a general concern)
  actionability_filter: string;
  // corpus-wide counts per value; empty when no labels run carries the field
  available_actionability: Record<string, number>;
  // "" | "reported" — no negative direction by design: not describing an
  // incident is not evidence that none occurred
  event_filter: string;
  // { reported, coded }; empty when no labels run carries the field
  available_events: Record<string, number>;
  // "" | "day" | "night" — classified from verbatim time mentions; no filter
  // for responses naming no time (that says nothing about when it happened)
  time_filter: string;
  // { day, night, mentioned }; empty when nothing classifies
  available_time: Record<string, number>;
  // survey question ids this proposal was scoped to; empty = all questions
  question_scope: string[];
  // the analyst's respondent restriction, validated and echoed — the answer
  // step must carry it back unchanged for the filter to apply
  demographic_filter: Record<string, string[]>;
  warnings: string[];
}

// One demographic field the ask form offers as a respondent filter. Counts
// are RESPONDENTS (the person, not their per-question responses); a blank
// cell is missing data and appears under no value.
export interface AskDemographicValue {
  value: string;
  n_respondents: number;
}

export interface AskDemographic {
  field: string;
  values: AskDemographicValue[];
}

// Faceted counts for the ask form: each field's values recounted under the
// OTHER fields' ticked values, plus how many respondents match the whole
// filter (null when nothing is ticked).
export interface AskDemographicsResponse {
  fields: AskDemographic[];
  n_matching_respondents: number | null;
}

export interface AskQuestionOut {
  question_id: string;
  question_text: string;
  n_responses: number;
}

export interface AskSelectedCandidate {
  label_id: string;
  relevance: string;
  rationale: string;
}

export interface AskAnswerRequest {
  question: string;
  route: string;
  reason: string;
  selected: AskSelectedCandidate[];
  lexicon_concepts: string[];
  group_by: string;
  location_filter: string[];
  actionability_filter: string;
  event_filter: string;
  time_filter: string;
  question_scope: string[];
  // {field: [values]} — validated server-side against the dataset's actual
  // demographics; values within a field are OR, fields are AND
  demographic_filter: Record<string, string[]>;
  proposed_label_ids: string[];
  aggregate_target: string;
}

export interface AskSource {
  n: number;
  response_key: string;
  label_id: string;
  text: string;
  location?: string | null;
}

export interface AskAnswerStats {
  categories_searched: number;
  categories_total: number;
  unique_responses: number;
  quotes_shown: number;
  quotes_cited: number;
  // coverage guardrails: how much of the scoped questions' coded responses
  // the searched categories cover; small_base flags an answer resting on
  // fewer than ~200 responses
  scope_total: number;
  scope_coverage: number | null;
  small_base: boolean;
}

export interface AskSubCount {
  sub_label_id: string;
  name: string;
  count: number;
}

// Full-coverage sub-theme composition of one searched category. `generic`
// responses raised the category without naming a specific sub-theme;
// sub-counts can sum past the category count (a response may raise several).
export interface AskSubBreakdown {
  sub_counts: AskSubCount[];
  generic: number;
  coded: number;
}

export interface AskUncovered {
  label_id: string;
  name: string;
  count: number;
}

export interface AskLexiconCount {
  concept: string;
  mentions_total: number;
  mentions_by_question: Record<string, number>;
}

export interface AskGroupCount {
  name: string;
  count_unique_responses: number;
}

export interface AskAnswerResponse {
  // served from the persistent ask cache — the original stored answer,
  // byte-identical, zero model calls
  cached: boolean;
  run_id: string;
  // answer body only — sources arrive separately and the UI renders them
  answer_markdown: string;
  process_note: string;
  stats: AskAnswerStats;
  sources: AskSource[];
  counts: Record<string, number>;
  // with a location_filter, counts are filtered; this holds each category's
  // full size so the UI can show "n of N in category"
  counts_unfiltered: Record<string, number>;
  sampling_notes: Record<string, string>;
  lexicon_counts: AskLexiconCount[];
  group_counts: AskGroupCount[];
  location_filter: string[];
  location_counts: AskLocation[];
  location_denominator: { in_scope: number; naming_any: number } | null;
  actionability_filter: string;
  actionability_denominator: {
    in_scope: number;
    coded: number;
    matching: number;
  } | null;
  event_filter: string;
  event_denominator: {
    in_scope: number;
    coded: number;
    matching: number;
  } | null;
  time_filter: string;
  time_denominator: {
    in_scope: number;
    mentioning: number;
    matching: number;
  } | null;
  demographic_filter: Record<string, string[]>;
  // {in_scope, coded, matching} — `coded` responses belong to respondents
  // with a recorded value for every filtered field; the gap to in_scope is
  // missing data, never a group
  demographic_denominator: {
    in_scope: number;
    coded: number;
    matching: number;
  } | null;
  // the demographic filter left fewer matching responses than the thin-cell
  // notice threshold — disclosed prominently, never blocked
  demographic_thin: boolean;
  // survey questions whose every response counted as mentioning the filtered
  // place because the question itself asks about it
  location_filter_implicit_questions: string[];
  invalid_citations: number;
  deselected: string[];
  added: string[];
  // label_id -> full-coverage sub-theme breakdown (categories the sub-theme
  // layer has coded); empty object otherwise
  sub_breakdowns: Record<string, AskSubBreakdown>;
  // largest in-scope categories NOT searched — present only when coverage
  // fell below the floor
  uncovered_categories: AskUncovered[];
  // aggregate_direct answers only: the computed tally as structured data
  aggregate: Record<string, unknown> | null;
  // the answer's inspection record: numbers traced to computed counts,
  // quoted spans checked against their cited sources; null for
  // deterministic tallies (nothing model-written to verify)
  verification: {
    checked: boolean;
    violations: { kind: string; value: string | number; detail: string }[];
    repaired: boolean;
    residual: { kind: string; value: string | number; detail: string }[];
  } | null;
}

// --- Pipeline runs (induce -> label -> lexicon -> locations) ---

export interface PipelineEstimateItem {
  stage: string;
  question_id: string;
  question_text: string;
  detail: string;
  responses: number;
  est_cost_usd: number;
  // "planned"  = from the real chunking and prompt sizes the run will use
  // "projected" = extrapolated from a measured rate, because the real dry-run
  //               needs an artifact that doesn't exist yet
  // The UI must keep these visually distinct.
  basis: string;
}

export interface PipelineEstimate {
  dataset_id: string;
  dataset_description: string;
  // "incremental" (post-append) plans only never-labeled rows; all its items
  // are basis="projected"
  mode: "full" | "incremental";
  n_questions: number;
  n_responses: number;
  // rows the latest labels runs have never seen (incremental mode only)
  n_new_responses: number;
  items: PipelineEstimateItem[];
  est_total_usd: number;
  questions_with_existing_taxonomy: string[];
}

export interface PipelineStage {
  key: string;
  label: string;
  status: string; // pending | running | done | failed | skipped
  detail: string;
  cost_usd: number | null;
  seconds: number | null;
  error: string;
}

export interface PipelineJob {
  job_id: string;
  dataset_id: string;
  status: string; // running | done | failed
  stages: PipelineStage[];
  created_utc: string;
  finished_utc: string;
  error: string;
  // summed from each stage's own run manifest — real spend, not the estimate
  cost_usd: number;
  is_processed: boolean;
}
