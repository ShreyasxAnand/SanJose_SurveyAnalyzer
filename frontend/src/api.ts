import type {
  AppendResponse,
  AskAnswerRequest,
  AskAnswerResponse,
  AskDemographicsResponse,
  AskQuestionOut,
  AskRouteResponse,
  CalibrationInfo,
  ConfigPatch,
  DatasetHistoryOut,
  DatasetMetadataPatch,
  DatasetOut,
  DateRangesConfig,
  DeletionPreview,
  DeletionResult,
  PipelineEstimate,
  PipelineJob,
  ServerConfig,
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

/* The Settings screen needs to show whether this tab is unlocked and to let
   someone unlock it deliberately rather than by tripping a 401. Deliberately
   still sessionStorage: a passcode that outlives the tab is a passcode
   somebody else finds on a shared machine. */
export function hasStoredPasscode(): boolean {
  return !!sessionStorage.getItem(ADMIN_KEY);
}

export function setStoredPasscode(passcode: string): void {
  sessionStorage.setItem(ADMIN_KEY, passcode.trim());
}

export function clearStoredPasscode(): void {
  sessionStorage.removeItem(ADMIN_KEY);
}

/* Asking for the passcode.

   This module has no React, and the 401 handler has to pause mid-request
   until a person types something — which is why it used to call
   window.prompt. A browser prompt cannot mask its input, so the passcode was
   displayed in the clear as it was typed, in the one place the app asks for
   it most often. PasscodeGate registers a masked dialog here instead.

   There is deliberately no window.prompt fallback: a fallback that leaks the
   thing it is protecting is not a fallback. With no dialog registered the
   401 is returned as-is and surfaces as an error telling the user where to
   set the passcode. */
export type PasscodeAsker = (context: { reason: string }) => Promise<string | null>;

let askForPasscode: PasscodeAsker | null = null;
let pendingAsk: Promise<string | null> | null = null;

export function registerPasscodeAsker(fn: PasscodeAsker | null): void {
  askForPasscode = fn;
}

async function requestPasscode(reason: string): Promise<string | null> {
  if (!askForPasscode) return null;
  /* Concurrent 401s share one dialog. The catalog can fire several admin
     requests at once, and stacking a modal per request would make the user
     type the same passcode three times to dismiss them. */
  if (!pendingAsk) {
    pendingAsk = askForPasscode({ reason }).finally(() => {
      pendingAsk = null;
    });
  }
  return pendingAsk;
}

async function detailOf(res: Response): Promise<string> {
  // require_admin's 401 detail is written to be shown to a person verbatim,
  // so prefer it over anything this file could invent. Cloned because the
  // caller may still want to read the body.
  try {
    const body = await res.clone().json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON — fall through to the generic wording */
  }
  return "This action needs the admin passcode.";
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
    const entered = await requestPasscode(await detailOf(res));
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

/* --- server settings ---------------------------------------------------- */

export async function getConfig(): Promise<ServerConfig> {
  // admin-gated like the writes: this is the only read that describes the
  // lock rather than the data behind it
  const res = await adminFetch(`${BASE}/config`);
  return unwrap<ServerConfig>(res);
}

export async function putConfig(patch: ConfigPatch): Promise<ServerConfig> {
  const res = await adminFetch(`${BASE}/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return unwrap<ServerConfig>(res);
}

export async function recalibrateCosts(): Promise<CalibrationInfo> {
  const res = await adminFetch(`${BASE}/config/recalibrate`, { method: "POST" });
  return unwrap<CalibrationInfo>(res);
}

/* --- permanent dataset deletion ----------------------------------------- */

export async function deletionPreview(
  datasetId: number,
): Promise<DeletionPreview> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/deletion-preview`);
  return unwrap<DeletionPreview>(res);
}

export async function deleteDatasetPermanently(
  datasetId: number,
  confirmName: string,
): Promise<DeletionResult> {
  /* The name goes in the query string, not a body: DELETE with a body is
     under-specified and some proxies drop it. It is not a security control —
     the admin gate already ran — it is the speed bump on the one irreversible
     action in the app. */
  const res = await adminFetch(
    `${BASE}/datasets/${datasetId}/permanently?confirm_name=${encodeURIComponent(
      confirmName,
    )}`,
    { method: "DELETE" },
  );
  return unwrap<DeletionResult>(res);
}
