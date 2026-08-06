import { useEffect, useRef, useState } from "react";
import { getDatasetHistory, listDatasets, patchDataset } from "./api";
import type { DatasetHistoryOut, DatasetOut } from "./types";
import "./catalog.css";

/* Google Drive-style dataset catalog: the home page. Cards are ingested
   datasets (provisionals never show); hovering a card reveals its
   description; the ⋯ menu edits catalog metadata and shows the upload
   history; clicking a card opens the Ask view for it. */

export default function Catalog({
  onOpenDataset,
  onStartNewUpload,
  onStartAppend,
  onError,
}: {
  onOpenDataset: (d: DatasetOut) => void;
  onStartNewUpload: () => void;
  onStartAppend: (target: DatasetOut) => void;
  onError: (msg: string) => void;
}) {
  const [datasets, setDatasets] = useState<DatasetOut[] | null>(null);
  const [chooserOpen, setChooserOpen] = useState(false);
  const [editFor, setEditFor] = useState<DatasetOut | null>(null);
  const [historyFor, setHistoryFor] = useState<DatasetOut | null>(null);

  useEffect(() => {
    listDatasets()
      .then(setDatasets)
      .catch((e) => onError(String(e)));
  }, [onError]);

  if (datasets === null) return <div className="cat"><p>Loading datasets…</p></div>;

  const ingested = datasets.filter((d) => d.status === "ingested");

  return (
    <div className="cat">
      <div className="cat-header">
        <h2>Datasets</h2>
        <button className="cat-upload-btn" onClick={() => setChooserOpen(true)}>
          Upload
        </button>
      </div>

      {ingested.length === 0 ? (
        <div className="cat-empty">
          <p>No datasets yet — upload a survey file to get started.</p>
        </div>
      ) : (
        <div className="cat-grid">
          {ingested.map((d) => (
            <DatasetCard
              key={d.id}
              dataset={d}
              onOpen={() => onOpenDataset(d)}
              onEdit={() => setEditFor(d)}
              onHistory={() => setHistoryFor(d)}
            />
          ))}
        </div>
      )}

      {chooserOpen && (
        <UploadChooserModal
          targets={ingested}
          onNew={() => {
            setChooserOpen(false);
            onStartNewUpload();
          }}
          onAppend={(target) => {
            setChooserOpen(false);
            onStartAppend(target);
          }}
          onClose={() => setChooserOpen(false)}
        />
      )}
      {editFor && (
        <MetadataModal
          dataset={editFor}
          onSaved={(updated) => {
            setDatasets((ds) =>
              ds ? ds.map((d) => (d.id === updated.id ? updated : d)) : ds,
            );
            setEditFor(null);
          }}
          onClose={() => setEditFor(null)}
          onError={onError}
        />
      )}
      {historyFor && (
        <HistoryModal
          dataset={historyFor}
          onClose={() => setHistoryFor(null)}
          onError={onError}
        />
      )}
    </div>
  );
}

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString();
}

function surveyRange(d: DatasetOut): string {
  if (d.survey_start_date && d.survey_end_date)
    return `${d.survey_start_date} – ${d.survey_end_date}`;
  return d.survey_start_date ?? d.survey_end_date ?? "";
}

function DatasetCard({
  dataset,
  onOpen,
  onEdit,
  onHistory,
}: {
  dataset: DatasetOut;
  onOpen: () => void;
  onEdit: () => void;
  onHistory: () => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node))
        setMenuOpen(false);
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [menuOpen]);

  const rows = dataset.exports?.total_row_count;
  const metaLine =
    rows != null
      ? `${rows.toLocaleString()} responses · ${dataset.questions.length} questions`
      : `${dataset.questions.length} questions`;
  const range = surveyRange(dataset);
  const detailLine = [dataset.department, range].filter(Boolean).join(" · ");

  return (
    <div
      className="cat-card"
      tabIndex={0}
      role="button"
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !menuOpen) onOpen();
      }}
    >
      <p className="cat-card-name">{dataset.name}</p>
      <p className="cat-card-meta">{metaLine}</p>
      {detailLine && <p className="cat-card-meta-2">{detailLine}</p>}
      <p className="cat-card-meta-2">Uploaded {formatDate(dataset.uploaded_at)}</p>

      <div className="cat-card-desc" aria-hidden="true">
        <p className="cat-card-desc-eyebrow">Description</p>
        <p className="cat-card-desc-text">
          {dataset.description ?? "No description — add one from the ⋯ menu."}
        </p>
      </div>

      <button
        className="cat-menu-btn"
        aria-label={`Options for ${dataset.name}`}
        onClick={(e) => {
          e.stopPropagation();
          setMenuOpen((o) => !o);
        }}
      >
        ⋯
      </button>
      {menuOpen && (
        <div className="cat-menu" ref={menuRef} onClick={(e) => e.stopPropagation()}>
          <button
            onClick={() => {
              setMenuOpen(false);
              onEdit();
            }}
          >
            Details &amp; edit
          </button>
          <button
            onClick={() => {
              setMenuOpen(false);
              onHistory();
            }}
          >
            View history
          </button>
        </div>
      )}
    </div>
  );
}

function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);

  return (
    <div
      className="cat-modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="cat-modal">
        <h3>{title}</h3>
        {children}
      </div>
    </div>
  );
}

function UploadChooserModal({
  targets,
  onNew,
  onAppend,
  onClose,
}: {
  targets: DatasetOut[];
  onNew: () => void;
  onAppend: (target: DatasetOut) => void;
  onClose: () => void;
}) {
  const [appendPicking, setAppendPicking] = useState(false);
  const [targetId, setTargetId] = useState<number | null>(null);

  return (
    <Modal title="Upload a survey file" onClose={onClose}>
      {!appendPicking ? (
        <>
          <button className="cat-choice" onClick={onNew}>
            <strong>New dataset</strong>
            <span>Ingest this file as a brand-new dataset.</span>
          </button>
          <button
            className="cat-choice"
            onClick={() => setAppendPicking(true)}
            disabled={targets.length === 0}
          >
            <strong>Append to an existing dataset</strong>
            <span>
              {targets.length === 0
                ? "No ingested datasets to append to yet."
                : "Add new responses to a dataset already here — duplicates are detected and skipped."}
            </span>
          </button>
          <div className="cat-modal-actions">
            <button className="cat-btn-secondary" onClick={onClose}>
              Cancel
            </button>
          </div>
        </>
      ) : (
        <>
          <p style={{ margin: "0 0 4px", fontSize: 13.5 }}>
            Which dataset should this file be appended to?
          </p>
          <div className="cat-target-list">
            {targets.map((t) => (
              <label key={t.id}>
                <input
                  type="radio"
                  name="append-target"
                  checked={targetId === t.id}
                  onChange={() => setTargetId(t.id)}
                />
                {t.name}
                {t.exports && (
                  <span className="cat-target-count">
                    {t.exports.total_row_count.toLocaleString()} responses
                  </span>
                )}
              </label>
            ))}
          </div>
          <div className="cat-modal-actions">
            <button
              className="cat-btn-secondary"
              onClick={() => setAppendPicking(false)}
            >
              Back
            </button>
            <button
              className="cat-btn-primary"
              disabled={targetId === null}
              onClick={() => {
                const target = targets.find((t) => t.id === targetId);
                if (target) onAppend(target);
              }}
            >
              Continue
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}

function MetadataModal({
  dataset,
  onSaved,
  onClose,
  onError,
}: {
  dataset: DatasetOut;
  onSaved: (updated: DatasetOut) => void;
  onClose: () => void;
  onError: (msg: string) => void;
}) {
  const [name, setName] = useState(dataset.name);
  const [department, setDepartment] = useState(dataset.department ?? "");
  const [notes, setNotes] = useState(dataset.notes ?? "");
  const [startDate, setStartDate] = useState(dataset.survey_start_date ?? "");
  const [endDate, setEndDate] = useState(dataset.survey_end_date ?? "");
  const [saving, setSaving] = useState(false);

  const datesBackwards = !!startDate && !!endDate && startDate > endDate;

  const handleSave = async () => {
    setSaving(true);
    try {
      /* Every editable field is sent: "" clears server-side, so a field the
         analyst blanked out actually clears instead of being preserved. */
      const updated = await patchDataset(dataset.id, {
        name,
        department,
        notes,
        survey_start_date: startDate,
        survey_end_date: endDate,
      });
      onSaved(updated);
    } catch (e) {
      onError(String(e));
      setSaving(false);
    }
  };

  return (
    <Modal title="Dataset details" onClose={onClose}>
      <label className="cat-field">
        <span className="cat-field-label">Name</span>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
      </label>

      <div className="cat-field">
        <span className="cat-field-label">Description</span>
        <div className="cat-readonly">
          {dataset.description ?? "No description was provided at ingest."}
        </div>
        <p className="cat-field-hint">
          Set at ingest. Not editable — it is part of every analysis prompt and
          each run's identity.
        </p>
      </div>

      <label className="cat-field">
        <span className="cat-field-label">Department</span>
        <input
          type="text"
          value={department}
          placeholder="Department this survey belongs to, e.g. Parks & Recreation"
          onChange={(e) => setDepartment(e.target.value)}
        />
      </label>

      <div className="cat-field">
        <span className="cat-field-label">Survey dates</span>
        <div className="cat-dates">
          <label>
            <span className="cat-field-hint">conducted from</span>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </label>
          <label>
            <span className="cat-field-hint">to</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </label>
        </div>
        {datesBackwards && (
          <p className="cat-field-hint" style={{ color: "var(--cat-amber-ink)" }}>
            The start date is after the end date.
          </p>
        )}
      </div>

      <label className="cat-field">
        <span className="cat-field-label">Notes</span>
        <textarea
          rows={3}
          value={notes}
          placeholder="Anything a colleague should know about this dataset"
          onChange={(e) => setNotes(e.target.value)}
        />
      </label>
      <p className="cat-field-hint">
        Name, department, dates, and notes are catalog metadata — shown here and
        written to the export manifest, never used in analysis prompts.
      </p>

      <div className="cat-modal-actions">
        <button className="cat-btn-secondary" onClick={onClose} disabled={saving}>
          Cancel
        </button>
        <button
          className="cat-btn-primary"
          onClick={handleSave}
          disabled={saving || !name.trim() || datesBackwards}
        >
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
    </Modal>
  );
}

function HistoryModal({
  dataset,
  onClose,
  onError,
}: {
  dataset: DatasetOut;
  onClose: () => void;
  onError: (msg: string) => void;
}) {
  const [history, setHistory] = useState<DatasetHistoryOut | null>(null);

  useEffect(() => {
    getDatasetHistory(dataset.id)
      .then(setHistory)
      .catch((e) => onError(String(e)));
  }, [dataset.id, onError]);

  return (
    <Modal title={`History — ${dataset.name}`} onClose={onClose}>
      {history === null ? (
        <p>Loading…</p>
      ) : (
        <ul className="cat-history">
          {history.entries.map((e) => (
            <li key={e.upload_id}>
              <div className="cat-history-head">
                <span className="cat-history-date">{formatDate(e.uploaded_at)}</span>
                <span>
                  <strong>{e.kind === "created" ? "Created" : "Appended"}</strong>
                  {" — "}
                  {e.kind === "created"
                    ? `${e.row_count.toLocaleString()} rows from ${e.filename}`
                    : `${e.new_row_count.toLocaleString()} new, ${e.duplicate_row_count.toLocaleString()} duplicates skipped (${e.filename})`}
                </span>
              </div>
              <p className="cat-history-note">{e.note ?? "—"}</p>
            </li>
          ))}
        </ul>
      )}
      <div className="cat-modal-actions">
        <button className="cat-btn-secondary" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}
