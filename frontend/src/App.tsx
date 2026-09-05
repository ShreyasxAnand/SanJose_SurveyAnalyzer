import { useEffect, useState } from "react";
import Pipeline from "./Pipeline";
import {
  appendToDataset,
  discardDataset,
  discardDatasetOnClose,
  exportDataset,
  getDataset,
  selectColumns,
  uploadDataset,
} from "./api";
import type { MetadataSelection } from "./api";
import type {
  AppendResponse,
  DatasetMatch,
  DatasetOut,
  DateRangesConfig,
  UploadResponse,
} from "./types";
import Ask from "./Ask";
import Catalog from "./Catalog";
import PasscodeGate from "./PasscodeGate";
import Settings from "./Settings";

/* The analyst chose upfront whether this file is a new dataset or an append
   to a known target; the server's duplicate check still runs either way as a
   safety net on the "new" path. */
type IngestMode =
  | { kind: "new" }
  | { kind: "append"; targetId: number; targetName: string };

type Step =
  | { name: "choose-file"; mode: IngestMode }
  | { name: "match-found"; upload: UploadResponse }
  | { name: "select-columns"; upload: UploadResponse }
  | { name: "append-confirm"; upload: UploadResponse; targetId: number; targetName: string }
  | { name: "done"; dataset: DatasetOut }
  | { name: "append-done"; result: AppendResponse };

type View =
  | { name: "catalog" }
  | { name: "ingest"; step: Step }
  | { name: "ask"; datasetId: number; datasetName: string }
  // Server settings. Carries no state of its own — the back button block
  // below already handles any non-catalog view, and provisionalUploadId
  // returns null for anything that is not an ingest step.
  | { name: "settings" };

/* Steps during which a provisional dataset exists server-side (created by
   upload, not yet consumed by commit/append/discard). */
function provisionalUploadId(view: View): number | null {
  if (view.name !== "ingest") return null;
  const step = view.step;
  if (
    step.name === "match-found" ||
    step.name === "select-columns" ||
    step.name === "append-confirm"
  ) {
    return step.upload.dataset_id;
  }
  return null;
}

export default function App() {
  const [view, setView] = useState<View>({ name: "catalog" });
  const [error, setError] = useState<string | null>(null);

  const setStep = (step: Step) => setView({ name: "ingest", step });

  /* While a provisional upload is on the server, closing the tab discards it
     so abandoned uploads don't linger on disk. The server 409s on anything
     already ingested, so this can only ever remove a provisional. */
  const provisionalId = provisionalUploadId(view);
  const step = view.name === "ingest" ? view.step : null;
  useEffect(() => {
    if (provisionalId === null) return;
    const handler = () => discardDatasetOnClose(provisionalId);
    window.addEventListener("pagehide", handler);
    return () => window.removeEventListener("pagehide", handler);
  }, [provisionalId]);

  async function handleBack() {
    /* Leaving mid-flow with a provisional on the server discards it — the
       button says so, so no confirm dialog is needed. */
    if (provisionalId !== null) {
      try {
        await discardDataset(provisionalId);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        return;
      }
    }
    setView({ name: "catalog" });
  }

  return (
    <main style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "sans-serif" }}>
      {/* Always mounted: any view can trip the admin gate, and the dialog it
          registers with api.ts is what replaces the old window.prompt. */}
      <PasscodeGate />
      <h1>Survey Analyzer</h1>
      {view.name !== "catalog" && (
        <p>
          <button onClick={handleBack}>
            {provisionalId !== null
              ? "← Discard upload & back to datasets"
              : "← All datasets"}
          </button>
          {view.name === "ask" && (
            <span style={{ marginLeft: "0.75rem", fontWeight: "bold" }}>
              {view.datasetName}
            </span>
          )}
        </p>
      )}
      {error && (
        <p style={{ color: "crimson" }}>
          {error}{" "}
          <button onClick={() => setError(null)}>dismiss</button>
        </p>
      )}

      {view.name === "catalog" && (
        <Catalog
          onOpenDataset={(d) =>
            setView({ name: "ask", datasetId: d.id, datasetName: d.name })
          }
          onStartNewUpload={() => setStep({ name: "choose-file", mode: { kind: "new" } })}
          onStartAppend={(target) =>
            setStep({
              name: "choose-file",
              mode: { kind: "append", targetId: target.id, targetName: target.name },
            })
          }
          onOpenSettings={() => setView({ name: "settings" })}
          onError={setError}
        />
      )}

      {view.name === "settings" && <Settings onError={setError} />}

      {view.name === "ask" && (
        // key resets in-flight ask state when the dataset changes
        <Ask key={view.datasetId} datasetId={view.datasetId} onError={setError} />
      )}

      {/* step is a const so TS narrowing survives into the JSX callbacks
          below (view.step property access doesn't). */}
      {step?.name === "choose-file" && (
        <ChooseFileStep
          mode={step.mode}
          onUploaded={(upload, mode) => {
            if (mode.kind === "append") {
              /* The target was chosen upfront; even a zero-overlap file goes
                 to the confirm screen — append recomputes dedup server-side. */
              setStep({
                name: "append-confirm",
                upload,
                targetId: mode.targetId,
                targetName: mode.targetName,
              });
            } else {
              setStep(
                upload.duplicate_check.outcome === "none" &&
                  upload.duplicate_check.column_matches.length === 0
                  ? { name: "select-columns", upload }
                  : { name: "match-found", upload },
              );
            }
          }}
          onError={setError}
        />
      )}

      {step?.name === "match-found" && (
        <MatchStep
          upload={step.upload}
          onAppendChosen={(upload, m) =>
            setStep({
              name: "append-confirm",
              upload,
              targetId: m.dataset_id,
              targetName: m.dataset_name,
            })
          }
          onIngestAsNew={() =>
            setStep({ name: "select-columns", upload: step.upload })
          }
          onDiscarded={() => setView({ name: "catalog" })}
          onError={setError}
        />
      )}

      {step?.name === "append-confirm" && (
        <AppendConfirmStep
          upload={step.upload}
          targetId={step.targetId}
          targetName={step.targetName}
          onAppendDone={(result) => setStep({ name: "append-done", result })}
          onIngestAsNew={() =>
            setStep({ name: "select-columns", upload: step.upload })
          }
          onDiscarded={() => setView({ name: "catalog" })}
          onError={setError}
        />
      )}

      {step?.name === "select-columns" && (
        <ColumnSelectStep
          upload={step.upload}
          onDone={(dataset) => setStep({ name: "done", dataset })}
          onError={setError}
        />
      )}

      {step?.name === "done" && (
        <DoneStep
          dataset={step.dataset}
          onDatasetChange={(dataset) => setStep({ name: "done", dataset })}
          onError={setError}
        />
      )}

      {step?.name === "append-done" && (
        <AppendDoneStep result={step.result} onError={setError} />
      )}
    </main>
  );
}

function MatchStep({
  upload,
  onAppendChosen,
  onIngestAsNew,
  onDiscarded,
  onError,
}: {
  upload: UploadResponse;
  onAppendChosen: (upload: UploadResponse, m: DatasetMatch) => void;
  onIngestAsNew: () => void;
  onDiscarded: () => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const check = upload.duplicate_check;
  const best = check.matches[0];

  async function handleDiscard() {
    setBusy(true);
    try {
      await discardDataset(upload.dataset_id);
      onDiscarded();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      {check.outcome === "none" ? (
        /* Column-level match: no row is identical (the column set differs,
           which changes every row hash) but most columns are byte-identical
           to a file already ingested. Appending is impossible; the choice is
           discard vs knowingly re-processing data the system already holds. */
        <>
          <p>
            <strong>{upload.original_filename}</strong> looks like data that is
            already ingested, with a different column set:
          </p>
          <ul>
            {check.column_matches.map((m) => (
              <li key={m.dataset_id} style={{ marginBottom: "0.75rem" }}>
                {m.matched_columns.length + m.renamed_columns.length} column
                {m.matched_columns.length + m.renamed_columns.length === 1
                  ? " is"
                  : "s are"}{" "}
                identical to <strong>{m.upload_filename}</strong> in{" "}
                <strong>{m.dataset_name}</strong> (dataset #{m.dataset_id},{" "}
                {m.upload_rows} rows).
                {m.missing_columns.length > 0 && (
                  <div>
                    Missing from this file: {m.missing_columns.join(", ")}
                  </div>
                )}
                {m.added_columns.length > 0 && (
                  <div>New in this file: {m.added_columns.join(", ")}</div>
                )}
                {m.renamed_columns.length > 0 && (
                  <div>
                    Renamed:{" "}
                    {m.renamed_columns
                      .map((r) => `${r.stored_name} → ${r.file_name}`)
                      .join(", ")}
                  </div>
                )}
                {m.changed_columns.length > 0 && (
                  <div>
                    Same name but different content:{" "}
                    {m.changed_columns.join(", ")}
                  </div>
                )}
              </li>
            ))}
          </ul>
          <p style={{ color: "#b45309" }}>
            Appending isn't possible (no whole row matches when the column set
            differs). Ingesting as new will run the full pipeline again on
            responses that were already processed.
          </p>
          <p>
            <button onClick={handleDiscard} disabled={busy}>
              Discard this upload
            </button>{" "}
            <button onClick={onIngestAsNew} disabled={busy}>
              Ingest as a separate new dataset anyway
            </button>
          </p>
        </>
      ) : check.outcome === "exact" ? (
        <>
          <p>
            All {best.file_rows} rows of <strong>{upload.original_filename}</strong>{" "}
            are already ingested as <strong>{best.dataset_name}</strong> (dataset
            #{best.dataset_id}). Appending would add nothing.
          </p>
          <p>
            <button onClick={handleDiscard} disabled={busy}>
              Discard this upload
            </button>{" "}
            <button onClick={onIngestAsNew} disabled={busy}>
              Ingest as a separate new dataset anyway
            </button>
          </p>
        </>
      ) : (
        <>
          <p>
            <strong>{upload.original_filename}</strong> overlaps existing data:
          </p>
          <ul>
            {check.matches.map((m) => (
              <li key={m.dataset_id} style={{ marginBottom: "0.75rem" }}>
                <strong>{m.dataset_name}</strong> (dataset #{m.dataset_id}) already
                holds {m.matched_rows} of this file's {m.file_rows} rows.
                Appending skips those {m.matched_rows} duplicates and adds the{" "}
                {m.file_rows - m.matched_rows} new rows — only the new rows get
                labeled, reusing the existing categories.
                {!m.columns_compatible && (
                  <div style={{ color: "#b45309", fontSize: "0.9em" }}>
                    Can't append: the file is missing this dataset's question
                    column{m.missing_columns.length === 1 ? "" : "s"}{" "}
                    {m.missing_columns.join(", ")}.
                  </div>
                )}
                <div style={{ marginTop: "0.25rem" }}>
                  <button
                    onClick={() => onAppendChosen(upload, m)}
                    disabled={busy || !m.columns_compatible}
                  >
                    Append to {m.dataset_name}…
                  </button>
                </div>
              </li>
            ))}
          </ul>
          <p>
            <button onClick={onIngestAsNew} disabled={busy}>
              Ingest as a separate new dataset
            </button>{" "}
            <button onClick={handleDiscard} disabled={busy}>
              Discard upload
            </button>
          </p>
        </>
      )}
    </section>
  );
}

function AppendConfirmStep({
  upload,
  targetId,
  targetName,
  onAppendDone,
  onIngestAsNew,
  onDiscarded,
  onError,
}: {
  upload: UploadResponse;
  targetId: number;
  targetName: string;
  onAppendDone: (r: AppendResponse) => void;
  onIngestAsNew: () => void;
  onDiscarded: () => void;
  onError: (msg: string) => void;
}) {
  const [target, setTarget] = useState<DatasetOut | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getDataset(targetId)
      .then(setTarget)
      .catch((err) => onError(err instanceof Error ? err.message : String(err)));
  }, [targetId, onError]);

  /* Fully knowable from data already in hand, so surface it before submit —
     the backend 400 remains the real gate. */
  const uploadColumns = new Set(upload.columns.map((c) => c.column));
  const missingColumns =
    target?.questions
      .map((q) => q.source_column)
      .filter((col) => !uploadColumns.has(col)) ?? [];

  /* Overlap numbers only if the upload-time duplicate check actually
     computed them for this target — never invented. */
  const match = upload.duplicate_check.matches.find(
    (m) => m.dataset_id === targetId,
  );

  async function handleAppend() {
    setBusy(true);
    try {
      onAppendDone(
        await appendToDataset(targetId, upload.dataset_id, note.trim() || null),
      );
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleDiscard() {
    setBusy(true);
    try {
      await discardDataset(upload.dataset_id);
      onDiscarded();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <p>
        Append <strong>{upload.original_filename}</strong> ({upload.row_count}{" "}
        rows) to <strong>{targetName}</strong>.
      </p>

      {target === null ? (
        <p>Checking column compatibility…</p>
      ) : missingColumns.length > 0 ? (
        <p style={{ color: "#b45309" }}>
          Can't append: the file is missing this dataset's question column
          {missingColumns.length === 1 ? "" : "s"} {missingColumns.join(", ")}.
        </p>
      ) : match ? (
        <p>
          <strong>{targetName}</strong> already holds {match.matched_rows} of
          this file's {match.file_rows} rows. Appending will skip those
          duplicates and add the {match.file_rows - match.matched_rows} new rows
          — only new rows get labeled, reusing the existing categories.
        </p>
      ) : (
        <p>
          No rows of this file matched {targetName} at upload time. The
          duplicate check runs again at append time; any rows already held will
          be skipped.
        </p>
      )}

      <fieldset style={{ marginBottom: "1rem" }}>
        <legend>Note for the dataset's history</legend>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder='What is this file and why is it being added? e.g. "2026 Q3 wave — late responses"'
          rows={2}
          style={{ width: "32rem", maxWidth: "100%" }}
        />
        <div style={{ fontSize: "0.85em", color: "#555" }}>
          Optional — stored with the upload date in this dataset's history.
        </div>
      </fieldset>

      <p>
        <button
          onClick={handleAppend}
          disabled={busy || target === null || missingColumns.length > 0}
        >
          {busy ? "Appending…" : `Append to ${targetName}`}
        </button>{" "}
        <button onClick={onIngestAsNew} disabled={busy}>
          Ingest as a separate new dataset instead
        </button>{" "}
        <button onClick={handleDiscard} disabled={busy}>
          Discard upload
        </button>
      </p>
    </section>
  );
}

function AppendDoneStep({
  result,
  onError,
}: {
  result: AppendResponse;
  onError: (msg: string) => void;
}) {
  const { dataset } = result;
  return (
    <section>
      <p>
        Appended <strong>{result.appended_rows}</strong> new row
        {result.appended_rows === 1 ? "" : "s"} to{" "}
        <strong>{dataset.name}</strong> — {result.skipped_duplicates} duplicate
        {result.skipped_duplicates === 1 ? "" : "s"} skipped (their labels
        already exist).
      </p>
      {Object.keys(result.new_responses_per_question).length > 0 && (
        <ul>
          {Object.entries(result.new_responses_per_question).map(([label, n]) => (
            <li key={label}>
              {label} — {n} new response{n === 1 ? "" : "s"}
            </li>
          ))}
        </ul>
      )}
      {result.warnings.length > 0 && (
        <ul style={{ color: "#b45309" }}>
          {result.warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}
      {result.appended_rows > 0 ? (
        /* Only the new rows still need labels; incremental mode plans exactly
           those, reusing the existing taxonomy. */
        <Pipeline datasetId={dataset.id} mode="incremental" onError={onError} />
      ) : (
        <p>Nothing new to process — the dataset is unchanged.</p>
      )}
    </section>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function ChooseFileStep({
  mode,
  onUploaded,
  onError,
}: {
  mode: IngestMode;
  onUploaded: (u: UploadResponse, mode: IngestMode) => void;
  onError: (msg: string) => void;
}) {
  /* Picking a file only stages it in the browser — nothing reaches the
     server (and nothing is saved to disk) until the analyst confirms.
     Closing the tab before confirming therefore leaves no trace. */
  const [pending, setPending] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [inputKey, setInputKey] = useState(0);

  function clearPending() {
    setPending(null);
    setInputKey((k) => k + 1); // remount the input so re-picking the same file fires onChange
  }

  async function handleConfirm() {
    if (!pending) return;
    setBusy(true);
    try {
      const upload = await uploadDataset(pending);
      clearPending();
      onUploaded(upload, mode);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <p>
        {mode.kind === "append" ? (
          <>
            This file will be <strong>appended to {mode.targetName}</strong>.
          </>
        ) : (
          <>
            This file will become a <strong>new dataset</strong>.
          </>
        )}
      </p>
      <p>Upload a wide-format survey export (.csv, .xlsx, .xls).</p>
      <input
        key={inputKey}
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={(e) => setPending(e.target.files?.[0] ?? null)}
        disabled={busy}
      />
      {pending && (
        <p>
          <strong>{pending.name}</strong> ({formatSize(pending.size)}) — nothing
          is uploaded or saved until you confirm.{" "}
          <button onClick={handleConfirm} disabled={busy}>
            Upload and check this file
          </button>{" "}
          <button onClick={clearPending} disabled={busy}>
            Cancel
          </button>
        </p>
      )}
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
  // Demographic columns: attributes of the RESPONDENT (District, Age band),
  // not open-ended text to analyse. Independent of `selected` — a column is
  // one or the other, and the API rejects picking it as both.
  const [demoSelected, setDemoSelected] = useState<Record<string, boolean>>({});
  const [demoLabels, setDemoLabels] = useState<Record<string, string>>({});
  // Response-date column: cells parsed to ISO at ingest; filters and charts
  // see derived period labels (quarters by default, or custom named ranges).
  const [dateColumn, setDateColumn] = useState<string>("");
  const [dateLabel, setDateLabel] = useState("");
  const [periodMode, setPeriodMode] = useState<
    "quarter" | "month" | "year" | "custom"
  >("quarter");
  const [customRanges, setCustomRanges] = useState<
    { label: string; start: string; end: string }[]
  >([{ label: "", start: "", end: "" }]);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [description, setDescription] = useState("");
  const [department, setDepartment] = useState("");
  const [notes, setNotes] = useState("");
  const [surveyStart, setSurveyStart] = useState("");
  const [surveyEnd, setSurveyEnd] = useState("");
  const [busy, setBusy] = useState(false);

  const datesBackwards = !!surveyStart && !!surveyEnd && surveyStart > surveyEnd;

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

    if (datesBackwards) {
      onError("The survey start date is after the end date.");
      return;
    }

    const questions = selectedColumns.map((c) => ({
      column: c.column,
      label: labels[c.column].trim(),
    }));

    // A demographic's label may equal its column name ("District" is already
    // the right display name), unlike a question, which needs real wording.
    const metadataColumns: MetadataSelection[] = upload.columns
      .filter(
        (c) =>
          demoSelected[c.column] &&
          !selected[c.column] &&
          c.column !== dateColumn,
      )
      .map((c) => ({
        column: c.column,
        label: (demoLabels[c.column] ?? "").trim() || c.column,
      }));

    let dateRanges: DateRangesConfig | null = null;
    if (dateColumn) {
      if (selected[dateColumn]) {
        onError("The date column is also selected as a question column — it can only be one.");
        return;
      }
      metadataColumns.push({
        column: dateColumn,
        label: dateLabel.trim() || "Period",
        value_type: "date",
      });
      if (periodMode === "custom") {
        const complete = customRanges.filter(
          (r) => r.label.trim() && r.start && r.end,
        );
        if (complete.length === 0) {
          onError("Add at least one complete period (label, from, to) or pick an automatic bucketing.");
          return;
        }
        const backwards = complete.find((r) => r.start > r.end);
        if (backwards) {
          onError(`Period "${backwards.label}" starts after it ends.`);
          return;
        }
        dateRanges = {
          mode: "ranges",
          ranges: complete.map((r) => ({
            label: r.label.trim(),
            start: r.start,
            end: r.end,
          })),
        };
      } else {
        dateRanges = { mode: "bucket", granularity: periodMode };
      }
    }

    setBusy(true);
    try {
      const dataset = await selectColumns(
        upload.dataset_id,
        respondentIdColumn || null,
        questions,
        {
          description: description.trim() || null,
          department: department.trim() || null,
          notes: notes.trim() || null,
          surveyStartDate: surveyStart || null,
          surveyEndDate: surveyEnd || null,
          dateRanges,
        },
        metadataColumns,
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
          Used as context in every analysis prompt and fixed once ingested —
          it is part of each run's identity.
        </div>
      </fieldset>

      <fieldset style={{ marginBottom: "1rem" }}>
        <legend>Dataset details (optional)</legend>
        <div style={{ marginBottom: "0.5rem" }}>
          <label>
            Department{" "}
            <input
              type="text"
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              placeholder="e.g. Parks & Recreation"
              style={{ width: "20rem", maxWidth: "100%" }}
            />
          </label>
        </div>
        <div style={{ marginBottom: "0.5rem" }}>
          Survey conducted from{" "}
          <input
            type="date"
            value={surveyStart}
            onChange={(e) => setSurveyStart(e.target.value)}
          />{" "}
          to{" "}
          <input
            type="date"
            value={surveyEnd}
            onChange={(e) => setSurveyEnd(e.target.value)}
          />
          {datesBackwards && (
            <div style={{ fontSize: "0.85em", color: "#b45309" }}>
              The start date is after the end date.
            </div>
          )}
        </div>
        <div style={{ marginBottom: "0.5rem" }}>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Anything a colleague should know about this dataset"
            rows={2}
            style={{ width: "32rem", maxWidth: "100%" }}
          />
        </div>
        <div style={{ fontSize: "0.85em", color: "#555" }}>
          Shown in the catalog and written to the export manifest. Unlike the
          description, these are never used in analysis prompts, and can be
          edited later from the catalog.
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

      <fieldset style={{ marginBottom: "1rem" }}>
        <legend>Response date column (optional)</legend>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.9em", color: "#555" }}>
          When each response was collected (e.g. a survey completion date).
          Enables filtering and charts by time period — quarterly waves,
          before/after comparisons. Dates are read in common formats
          (2023-09-19, 9/19/2023); the period labels can be renamed or re-cut
          later without re-ingesting.
        </p>
        <select
          value={dateColumn}
          onChange={(e) => setDateColumn(e.target.value)}
        >
          <option value="">— none —</option>
          {upload.columns.map((c) => (
            <option
              key={c.column}
              value={c.column}
              disabled={!!selected[c.column] || c.non_null_count === 0}
            >
              {c.column}
              {c.sample_values.length > 0 ? ` (e.g. "${c.sample_values[0]}")` : ""}
            </option>
          ))}
        </select>
        {dateColumn && (
          <div style={{ marginTop: "0.5rem" }}>
            <div style={{ marginBottom: "0.5rem" }}>
              <label>
                Filter name{" "}
                <input
                  type="text"
                  value={dateLabel}
                  onChange={(e) => setDateLabel(e.target.value)}
                  placeholder='Period'
                  style={{ width: "12rem" }}
                />
              </label>
            </div>
            <div style={{ marginBottom: "0.5rem" }}>
              Label periods{" "}
              {(["quarter", "month", "year", "custom"] as const).map((m) => (
                <label key={m} style={{ marginRight: "0.75rem" }}>
                  <input
                    type="radio"
                    name="period-mode"
                    checked={periodMode === m}
                    onChange={() => setPeriodMode(m)}
                  />{" "}
                  {m === "custom" ? "custom ranges" : `by ${m}`}
                </label>
              ))}
            </div>
            {periodMode === "custom" && (
              <div>
                {customRanges.map((r, i) => (
                  <div key={i} style={{ marginBottom: "0.35rem" }}>
                    <input
                      type="text"
                      value={r.label}
                      placeholder={`e.g. Wave ${i + 1}`}
                      onChange={(e) =>
                        setCustomRanges((rs) =>
                          rs.map((x, j) =>
                            j === i ? { ...x, label: e.target.value } : x,
                          ),
                        )
                      }
                      style={{ width: "10rem", marginRight: "0.5rem" }}
                    />
                    from{" "}
                    <input
                      type="date"
                      value={r.start}
                      onChange={(e) =>
                        setCustomRanges((rs) =>
                          rs.map((x, j) =>
                            j === i ? { ...x, start: e.target.value } : x,
                          ),
                        )
                      }
                    />{" "}
                    to{" "}
                    <input
                      type="date"
                      value={r.end}
                      onChange={(e) =>
                        setCustomRanges((rs) =>
                          rs.map((x, j) =>
                            j === i ? { ...x, end: e.target.value } : x,
                          ),
                        )
                      }
                    />{" "}
                    <button
                      type="button"
                      onClick={() =>
                        setCustomRanges((rs) => rs.filter((_, j) => j !== i))
                      }
                      disabled={customRanges.length === 1}
                    >
                      remove
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() =>
                    setCustomRanges((rs) => [
                      ...rs,
                      { label: "", start: "", end: "" },
                    ])
                  }
                >
                  + add period
                </button>
                <div style={{ fontSize: "0.85em", color: "#555", marginTop: "0.25rem" }}>
                  Responses dated outside every period show as “(unlabeled)”.
                </div>
              </div>
            )}
          </div>
        )}
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

      <fieldset>
        <legend>Demographic columns (optional)</legend>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.9em", color: "#555" }}>
          Attributes of the respondent — District, Age band, Own/Rent — used to
          filter answers later, never analysed as text. Best with{" "}
          <strong>general groupings rather than exact values</strong>: broad
          groups give each filter enough respondents for the counts to mean
          something.
        </p>
        {upload.columns.map((c) => {
          const isQuestion = !!selected[c.column];
          const isDate = c.column === dateColumn;
          const on = !!demoSelected[c.column] && !isQuestion && !isDate;
          return (
            <div key={c.column} style={{ marginBottom: "0.4rem" }}>
              <label
                style={
                  isQuestion || isDate || c.non_null_count === 0
                    ? { color: "#999" }
                    : undefined
                }
              >
                <input
                  type="checkbox"
                  checked={on}
                  disabled={isQuestion || isDate || c.non_null_count === 0}
                  onChange={() =>
                    setDemoSelected((s) => ({ ...s, [c.column]: !s[c.column] }))
                  }
                />{" "}
                <strong>{c.column}</strong>{" "}
                {c.non_null_count === 0
                  ? "(empty)"
                  : isQuestion
                    ? "(already a question column)"
                    : isDate
                      ? "(the date column)"
                      : `(${c.non_null_count} non-empty)`}
              </label>
              {on && (
                <div style={{ marginLeft: "1.5rem", marginTop: "0.25rem" }}>
                  <input
                    type="text"
                    value={demoLabels[c.column] ?? ""}
                    onChange={(e) =>
                      setDemoLabels((l) => ({ ...l, [c.column]: e.target.value }))
                    }
                    placeholder={`Display name (defaults to "${c.column}")`}
                    style={{ width: "18rem", maxWidth: "100%" }}
                  />
                </div>
              )}
            </div>
          );
        })}
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

      {/* Ingest alone leaves the dataset unaskable; this is the next step, so
          it belongs here rather than only behind a 409 on the Ask tab. */}
      <Pipeline datasetId={dataset.id} onError={onError} />
    </section>
  );
}
