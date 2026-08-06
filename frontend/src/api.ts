import type {
  AppendResponse,
  AskAnswerRequest,
  AskAnswerResponse,
  AskQuestionOut,
  AskRouteResponse,
  DatasetHistoryOut,
  DatasetMetadataPatch,
  DatasetOut,
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

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/datasets/upload`, {
    method: "POST",
    body: form,
  });
  return unwrap<UploadResponse>(res);
}

export interface QuestionSelection {
  column: string;
  label: string;
}

// Catalog metadata collected alongside the column selection. null = leave
// any previously saved value untouched (the description semantics).
export interface DatasetMeta {
  description: string | null;
  department: string | null;
  notes: string | null;
  surveyStartDate: string | null;
  surveyEndDate: string | null;
}

export async function selectColumns(
  datasetId: number,
  respondentIdColumn: string | null,
  questions: QuestionSelection[],
  meta: Partial<DatasetMeta> = {},
): Promise<DatasetOut> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/columns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      respondent_id_column: respondentIdColumn,
      questions,
      dataset_description: meta.description ?? null,
      dataset_department: meta.department ?? null,
      dataset_notes: meta.notes ?? null,
      survey_start_date: meta.surveyStartDate ?? null,
      survey_end_date: meta.surveyEndDate ?? null,
    }),
  });
  return unwrap<DatasetOut>(res);
}

export async function appendToDataset(
  targetDatasetId: number,
  uploadDatasetId: number,
  note: string | null = null,
): Promise<AppendResponse> {
  const res = await fetch(`${BASE}/datasets/${targetDatasetId}/append`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_dataset_id: uploadDatasetId, note }),
  });
  return unwrap<AppendResponse>(res);
}

export async function discardDataset(datasetId: number): Promise<void> {
  const res = await fetch(`${BASE}/datasets/${datasetId}`, { method: "DELETE" });
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
  void fetch(`${BASE}/datasets/${datasetId}`, {
    method: "DELETE",
    keepalive: true,
  });
}

export async function exportDataset(datasetId: number): Promise<DatasetOut> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/export`, {
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
  const res = await fetch(`${BASE}/datasets/${datasetId}`, {
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

export async function askRoute(
  datasetId: number | string,
  question: string,
  questionScope: string[] = [],
): Promise<AskRouteResponse> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/ask/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, question_scope: questionScope }),
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
  const res = await fetch(`${BASE}/datasets/${datasetId}/pipeline/run`, {
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
