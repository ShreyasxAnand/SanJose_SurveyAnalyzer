import type { DatasetOut, UploadResponse } from "./types";

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

export async function selectColumns(
  datasetId: number,
  respondentIdColumn: string | null,
  questions: QuestionSelection[],
): Promise<DatasetOut> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/columns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      respondent_id_column: respondentIdColumn,
      questions,
    }),
  });
  return unwrap<DatasetOut>(res);
}

export async function exportDataset(datasetId: number): Promise<DatasetOut> {
  const res = await fetch(`${BASE}/datasets/${datasetId}/export`, {
    method: "POST",
  });
  return unwrap<DatasetOut>(res);
}
