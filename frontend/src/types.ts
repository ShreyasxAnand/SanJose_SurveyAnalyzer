export interface ColumnPreview {
  column: string;
  sample_values: string[];
  non_null_count: number;
}

export interface UploadResponse {
  dataset_id: number;
  name: string;
  original_filename: string;
  row_count: number;
  columns: ColumnPreview[];
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
  questions: QuestionColumnOut[];
  exports: ExportInfo | null;
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

export interface AskRouteResponse {
  answerable: boolean;
  route: string;
  reason: string;
  candidates: AskCandidate[];
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
  warnings: string[];
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
  proposed_label_ids: string[];
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
  invalid_citations: number;
  deselected: string[];
  added: string[];
}
