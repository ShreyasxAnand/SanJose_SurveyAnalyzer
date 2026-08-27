import type {
  AppendResponse,
  AskAnswerRequest,
  AskAnswerResponse,
  AskDemographicsResponse,
  AskQuestionOut,
  AskRouteResponse,
  DatasetHistoryOut,
  DatasetMetadataPatch,
  DatasetOut,
  DateRangesConfig,
  PipelineEstimate,
  PipelineJob,
  UploadResponse,
} from "./types";

const BASE = "/api";

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/* Admin passcode. When the server has ADMIN_PASSCODE configured, the
   dataset-changing endpoints (upload, reshape, append, discard, metadata
   edit, export, pipeline run/cancel) answer 401 without it. The passcode is
   kept per browser tab (sessionStorage); on a 401 the stored value is
   cleared, the user is asked once, and the request retries. When the server
   has no passcode configured, the header is simply absent and everything
   works as before. */
const ADMIN_KEY = "admin_passcode";

function adminHeaders(): Record<string, string> {
  const p = sessionStorage.getItem(ADMIN_KEY);
  return p ? { "X-Admin-Passcode": p } : {};
}

async function adminFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const send = (extra: Record<string, string>) =>
    fetch(url, {
      ...init,
      headers: { ...(init.headers as Record<string, string> | undefined), ...extra },
    });
  let res = await send(adminHeaders());
  if (res.status === 401) {
    sessionStorage.removeItem(ADMIN_KEY);
    const entered = window.prompt(
      "This action needs the admin passcode.\nEnter it to continue:",
    );
    if (entered && entered.trim()) {
      sessionStorage.setItem(ADMIN_KEY, entered.trim());
      res = await send({ "X-Admin-Passcode": entered.trim() });
    }
  }
  return res;
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await adminFetch(`${BASE}/datasets/upload`, {
    method: "POST",
    body: form,
  });
  return unwrap<UploadResponse>(res);
}

export interface QuestionSelection {
  column: string;
  label: string;
}

// A demographic or date column selection. value_type defaults to
// "categorical" server-side; "date" cells are parsed to ISO at ingest.
export interface MetadataSelection {
  column: string;
  label: string;
  value_type?: "categorical" | "date";
}

// Catalog metadata collected alongside the column selection. null = leave
// any previously saved value untouched (the description semantics).
export interface DatasetMeta {
  description: string | null;
  department: string | null;
  notes: string | null;
  surveyStartDate: string | null;
  surveyEndDate: string | null;
  // Period-labeling config for a date-typed column. null = keep stored
  // config (a newly selected date column defaults to quarter bucketing).
  dateRanges: DateRangesConfig | null;
}

export async function selectColumns(
  datasetId: number,
  respondentIdColumn: string | null,
  questions: QuestionSelection[],
  meta: Partial<DatasetMeta> = {},
  metadataColumns: MetadataSelection[] = [],
): Promise<DatasetOut> {
  const res = await adminFetch(`${BASE}/datasets/${datasetId}/columns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      respondent_id_column: respondentIdColumn,
      questions,
      metadata_columns: metadataColumns,
      dataset_description: meta.description ?? null,
      dataset_department: meta.department ?? null,
      dataset_notes: meta.notes ?? null,
      survey_start_date: meta.surveyStartDate ?? null,
      survey_end_date: meta.surveyEndDate ?? null,
      date_ranges: meta.dateRanges ?? null,
    }),
  });
  return unwrap<DatasetOut>(res);
}

export async function appendToDataset(
  targetDatasetId: number,
  uploadDatasetId: number,
  note: string | null = null,
): Promise<AppendResponse> {
  const res = await adminFetch(`${BASE}/datasets/${targetDatasetId}/append`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_dataset_id: uploadDatasetId, note }),
  });
  return unwrap<AppendResponse>(res);
}

export async function discardDataset(datasetId: number): Promise<void> {
  const res = await adminFetch(`${BASE}/datasets/${datasetId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
}

export function discardDatasetOnClose(datasetId: number): void {
  /* Fired from pagehide when the tab closes with an unconfirmed provisional
     upload — keepalive lets the request outlive the page. Best effort, no
     response handling possible. The server refuses to delete anything that
     isn't a provisional (status "uploaded"), so a race with an in-flight
     commit can never delete ingested data. */
  // adminHeaders, not adminFetch: the page is closing, so there is nobody to
  // prompt — send the stored passcode if there is one and accept best effort
  void fetch(`${BASE}/datasets/${datasetId}`, {
    method: "DELETE",
    keepalive: true,
    headers: adminHeaders(),
  });
}

export async function exportDataset(datasetId: number): Promise<DatasetOut> {
  const res = await adminFetch(`${BASE}/datasets/${datasetId}/export`, {
    method: "POST",
  });
  return unwrap<DatasetOut>(res);
}

export async function listDatasets(): Promise<DatasetOut[]> {
  const res = await fetch(`${BASE}/datasets`);
  return unwrap<DatasetOut[]>(res);
}

export async function getDataset(datasetId: number): Promise<DatasetOut> {
  const res = await fetch(`${BASE}/datasets/${datasetId}`);
  return unwrap<DatasetOut>(res);
}

export async function patchDataset(
  datasetId: number,
  patch: DatasetMetadataPatch,
): Promise<DatasetOut> {
  const res = await adminFetch(`${BASE}/datasets/${datasetId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return unwrap<DatasetOut>(res);
}

export async function getDatasetHistory(
  datasetId: number,
): Promise<DatasetHistoryOut> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/history`);
  return unwrap<DatasetHistoryOut>(res);
}

export async function askQuestions(
  datasetId: number | string,
): Promise<AskQuestionOut[]> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/ask/questions`);
  return unwrap<AskQuestionOut[]>(res);
}

export async function askDemographics(
  datasetId: number | string,
  demographicFilter: Record<string, string[]> = {},
): Promise<AskDemographicsResponse> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/ask/demographics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ demographic_filter: demographicFilter }),
  });
  return unwrap<AskDemographicsResponse>(res);
}

export async function askRoute(
  datasetId: number | string,
  question: string,
  questionScope: string[] = [],
  demographicFilter: Record<string, string[]> = {},
): Promise<AskRouteResponse> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/ask/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      question_scope: questionScope,
      demographic_filter: demographicFilter,
    }),
  });
  return unwrap<AskRouteResponse>(res);
}

export async function askAnswer(
  datasetId: number | string,
  payload: AskAnswerRequest,
): Promise<AskAnswerResponse> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/ask/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return unwrap<AskAnswerResponse>(res);
}

export async function pipelineEstimate(
  datasetId: number | string,
  mode: "full" | "incremental" = "full",
): Promise<PipelineEstimate> {
  const res = await fetch(
    `${BASE}/datasets/${datasetId}/pipeline/estimate?mode=${mode}`,
    { method: "POST" },
  );
  return unwrap<PipelineEstimate>(res);
}

export async function pipelineRun(
  datasetId: number | string,
  batchSize = 60,
  mode: "full" | "incremental" = "full",
): Promise<PipelineJob> {
  const res = await adminFetch(`${BASE}/datasets/${datasetId}/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ batch_size: batchSize, mode }),
  });
  return unwrap<PipelineJob>(res);
}

export async function pipelineStatus(
  datasetId: number | string,
): Promise<PipelineJob | null> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/pipeline/status`);
  return unwrap<PipelineJob | null>(res);
}

export async function pipelineProcessed(
  datasetId: number | string,
): Promise<{ dataset_id: string; is_processed: boolean }> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/pipeline/processed`);
  return unwrap<{ dataset_id: string; is_processed: boolean }>(res);
}
