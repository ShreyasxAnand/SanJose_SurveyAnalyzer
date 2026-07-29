import { useEffect, useState } from "react";
import { exportDataset, listDatasets, selectColumns, uploadDataset } from "./api";
import type { DatasetOut, UploadResponse } from "./types";
import Ask from "./Ask";

type Step =
  | { name: "upload" }
  | { name: "select-columns"; upload: UploadResponse }
  | { name: "done"; dataset: DatasetOut };

type View = "ingest" | "ask";

export default function App() {
  const [view, setView] = useState<View>("ingest");
  const [step, setStep] = useState<Step>({ name: "upload" });
  const [error, setError] = useState<string | null>(null);

  return (
    <main style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>Survey Analyzer</h1>
      <nav style={{ marginBottom: "1.5rem" }}>
        {(["ingest", "ask"] as const).map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            style={{
              marginRight: "0.5rem",
              fontWeight: view === v ? "bold" : "normal",
              textDecoration: view === v ? "underline" : "none",
            }}
          >
            {v === "ingest" ? "Ingest" : "Ask"}
          </button>
        ))}
      </nav>
      {error && (
        <p style={{ color: "crimson" }}>
          {error}{" "}
          <button onClick={() => setError(null)}>dismiss</button>
        </p>
      )}

      {view === "ask" && <AskView onError={setError} />}

      {view === "ingest" && step.name === "upload" && (
        <UploadStep
          onUploaded={(upload) => setStep({ name: "select-columns", upload })}
          onError={setError}
        />
      )}

      {view === "ingest" && step.name === "select-columns" && (
        <ColumnSelectStep
          upload={step.upload}
          onDone={(dataset) => setStep({ name: "done", dataset })}
          onError={setError}
        />
      )}

      {view === "ingest" && step.name === "done" && (
        <DoneStep
          dataset={step.dataset}
          onDatasetChange={(dataset) => setStep({ name: "done", dataset })}
          onError={setError}
        />
      )}
    </main>
  );
}

function AskView({ onError }: { onError: (msg: string) => void }) {
  const [datasets, setDatasets] = useState<DatasetOut[] | null>(null);
  const [datasetId, setDatasetId] = useState<number | null>(null);

  useEffect(() => {
    listDatasets()
      .then((ds) => {
        setDatasets(ds);
        if (ds.length > 0) setDatasetId((id) => id ?? ds[0].id);
      })
      .catch((err) =>
        onError(err instanceof Error ? err.message : String(err)),
      );
  }, [onError]);

  if (datasets === null) return <p>Loading datasets…</p>;
  if (datasets.length === 0) {
    return <p>No datasets yet — ingest a survey file first.</p>;
  }

  return (
    <div>
      {datasets.length > 1 && (
        <p>
          Dataset:{" "}
          <select
            value={datasetId ?? ""}
            onChange={(e) => setDatasetId(Number(e.target.value))}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </p>
      )}
      {datasetId !== null && (
        // key resets in-flight ask state when the dataset changes
        <Ask key={datasetId} datasetId={datasetId} onError={onError} />
      )}
    </div>
  );
}

function UploadStep({
  onUploaded,
  onError,
}: {
  onUploaded: (u: UploadResponse) => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);

  async function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      const upload = await uploadDataset(file);
      onUploaded(upload);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <p>Upload a wide-format survey export (.csv, .xlsx, .xls).</p>
      <input type="file" accept=".csv,.xlsx,.xls" onChange={handleChange} disabled={busy} />
      {busy && <p>Uploading…</p>}
    </section>
  );
}

function ColumnSelectStep({
  upload,
  onDone,
  onError,
}: {
  upload: UploadResponse;
  onDone: (d: DatasetOut) => void;
  onError: (msg: string) => void;
}) {
  const [respondentIdColumn, setRespondentIdColumn] = useState<string>("");
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);

  function toggle(column: string) {
    setSelected((s) => ({ ...s, [column]: !s[column] }));
  }

  function isRealWording(column: string, label: string): boolean {
    const trimmed = label.trim();
    return trimmed !== "" && trimmed.toLowerCase() !== column.trim().toLowerCase();
  }

  async function handleSubmit() {
    const selectedColumns = upload.columns.filter((c) => selected[c.column]);

    if (selectedColumns.length === 0) {
      onError("Select at least one question column.");
      return;
    }

    const missingWording = selectedColumns.filter(
      (c) => !isRealWording(c.column, labels[c.column] ?? ""),
    );
    if (missingWording.length > 0) {
      onError(
        `Enter the actual question wording (not the raw column name) for: ${missingWording
          .map((c) => c.column)
          .join(", ")}`,
      );
      return;
    }

    const questions = selectedColumns.map((c) => ({
      column: c.column,
      label: labels[c.column].trim(),
    }));

    setBusy(true);
    try {
      const dataset = await selectColumns(
        upload.dataset_id,
        respondentIdColumn || null,
        questions,
        description.trim() || null,
      );
      onDone(dataset);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <p>
        {upload.original_filename} — {upload.row_count} rows
      </p>

      <fieldset style={{ marginBottom: "1rem" }}>
        <legend>Survey description</legend>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What is this survey and who answered it? Descriptive only — never what you hope to find."
          rows={3}
          style={{ width: "32rem", maxWidth: "100%" }}
        />
        <div style={{ fontSize: "0.85em", color: "#555" }}>
          Used as context in every analysis prompt. Editing it later re-versions
          taxonomy and labeling runs.
        </div>
      </fieldset>

      <fieldset style={{ marginBottom: "1rem" }}>
        <legend>Respondent ID column (optional)</legend>
        <select
          value={respondentIdColumn}
          onChange={(e) => setRespondentIdColumn(e.target.value)}
        >
          <option value="">— none —</option>
          {upload.columns.map((c) => (
            <option key={c.column} value={c.column}>
              {c.column}
            </option>
          ))}
        </select>
      </fieldset>

      <fieldset>
        <legend>Question columns to ingest</legend>
        {upload.columns.map((c) => (
          <div key={c.column} style={{ marginBottom: "0.5rem" }}>
            <label style={c.non_null_count === 0 ? { color: "#999" } : undefined}>
              <input
                type="checkbox"
                checked={!!selected[c.column]}
                onChange={() => toggle(c.column)}
                disabled={c.non_null_count === 0}
              />{" "}
              <strong>{c.column}</strong>{" "}
              {c.non_null_count === 0
                ? "(empty — cannot ingest)"
                : `(${c.non_null_count} non-empty)`}
            </label>
            {c.sample_values.length > 0 && (
              <div style={{ fontSize: "0.85em", color: "#555", marginLeft: "1.5rem" }}>
                e.g. "{c.sample_values[0]}"
              </div>
            )}
            {selected[c.column] && (
              <div style={{ marginLeft: "1.5rem", marginTop: "0.25rem" }}>
                <input
                  type="text"
                  value={labels[c.column] ?? ""}
                  onChange={(e) =>
                    setLabels((l) => ({ ...l, [c.column]: e.target.value }))
                  }
                  placeholder={`Question wording, e.g. based on "${c.column}"`}
                  style={{ width: "24rem", maxWidth: "100%" }}
                />
                {!isRealWording(c.column, labels[c.column] ?? "") && (
                  <div style={{ fontSize: "0.85em", color: "#b45309" }}>
                    Enter the actual question text — the raw column name alone isn't
                    allowed, since it's what the taxonomy prompt will see later.
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </fieldset>

      <button onClick={handleSubmit} disabled={busy}>
        {busy ? "Ingesting…" : "Reshape & persist"}
      </button>
    </section>
  );
}

function DoneStep({
  dataset,
  onDatasetChange,
  onError,
}: {
  dataset: DatasetOut;
  onDatasetChange: (d: DatasetOut) => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const { exports } = dataset;

  async function handleReExport() {
    setBusy(true);
    try {
      onDatasetChange(await exportDataset(dataset.id));
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <p>
        Dataset <strong>{dataset.name}</strong> ingested (status: {dataset.status}).
      </p>
      <ul>
        {dataset.questions.map((q) => (
          <li key={q.id}>
            {q.label} ({q.source_column}) — {q.response_count} responses
          </li>
        ))}
      </ul>

      {exports && (
        <div style={{ marginTop: "1.5rem", padding: "1rem", background: "#f4f4f4" }}>
          <p>
            Saved to disk — {exports.total_row_count} rows across{" "}
            {Object.keys(exports.per_question_counts).length} questions:
          </p>
          <ul style={{ fontFamily: "monospace", fontSize: "0.9em" }}>
            <li>{exports.parquet_path}</li>
            <li>{exports.csv_path}</li>
            <li>{exports.manifest_path}</li>
          </ul>
          <p>
            <a href={`/api${exports.csv_download_url}`}>Download CSV</a>
            {" · "}
            <a href={`/api${exports.parquet_download_url}`}>Download Parquet</a>
          </p>
          <button onClick={handleReExport} disabled={busy}>
            {busy ? "Re-exporting…" : "Re-export (merges latest labels)"}
          </button>
        </div>
      )}
    </section>
  );
}
