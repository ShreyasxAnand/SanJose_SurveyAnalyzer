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
  questions: QuestionColumnOut[];
  exports: ExportInfo | null;
}
