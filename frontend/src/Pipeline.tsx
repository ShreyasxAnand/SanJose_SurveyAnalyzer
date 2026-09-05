import { useEffect, useRef, useState } from "react";
import { pipelineEstimate, pipelineProcessed, pipelineRun, pipelineStatus } from "./api";
import type { PipelineEstimate, PipelineJob } from "./types";
import "./pipeline.css";

// The button that turns an ingested dataset into an askable one: induce ->
// label -> lexicon -> locations -> summary, as a background job on the server.
//
// Estimate first, then confirm. The estimate makes no model calls and writes
// nothing, so the analyst always approves a figure before any spend. Which
// figures are *planned* (real chunking and prompt sizes) and which are
// *projected* (extrapolated from a measured rate) is shown per row and must
// stay shown.

type Mode =
  | { name: "idle" }
  | { name: "estimating" }
  | { name: "review"; estimate: PipelineEstimate }
  | { name: "running"; job: PipelineJob }
  // preloaded = found already-finished on mount (started in another tab, from
  // the CLI, or before this screen existed) rather than started here — shown
  // with its finish time so it can't read as the result of the current action
  | { name: "finished"; job: PipelineJob; preloaded?: boolean };

const POLL_MS = 3000;
// Consecutive status-poll failures tolerated before giving up. The job runs on
// the server, so a transient fetch error says nothing about the run — retrying
// with linear backoff is right, and only a sustained outage is worth reporting.
const MAX_POLL_FAILURES = 5;

const STAGE_ICON: Record<string, string> = {
  pending: "○",
  running: "◐",
  done: "✓",
  failed: "✕",
  skipped: "–",
};

function usd(n: number): string {
  return n < 0.01 && n > 0 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}

/* A range formatted to one precision, chosen by the smaller end. Formatting
   each end independently produced "$0.0070–$0.01", which reads as a typo
   rather than as a range. */
function usdRange(low: number, high: number): string {
  const cents = low >= 0.01;
  const fmt = (n: number) => (cents ? `$${n.toFixed(2)}` : `$${n.toFixed(4)}`);
  return `${fmt(low)}–${fmt(high)}`;
}

/* What the input tokens behind a row were actually derived from. Kept
   distinct from `basis` (planned vs projected) because they answer different
   questions: basis says whether the prompts exist yet, this says whether they
   were measured. A row can be projected and still measured — incremental
   labeling counts real prompts but cannot see the pool step coming. */
const INPUT_BASIS_TITLE: Record<string, string> = {
  counted: "Every prompt for this stage was measured with Vertex's countTokens",
  sampled:
    "Prompts measured on a sample and scaled by the measured tokens-per-character ratio",
  heuristic:
    "countTokens was unavailable, so input is the old 4-characters-per-token guess",
  projected:
    "No prompts exist to measure yet — input comes from a measured per-response rate",
};

function fmtWhen(stamp: string): string {
  // Job timestamps use filesystem-safe hyphens in the time part
  // ("2026-08-03T23-22-52Z") — restore colons before parsing.
  const d = new Date(stamp.replace(/T(\d{2})-(\d{2})-(\d{2})/, "T$1:$2:$3"));
  return isNaN(d.getTime()) ? stamp : d.toLocaleString();
}

export default function Pipeline({
  datasetId,
  mode: runMode = "full",
  onProcessed,
  onError,
}: {
  datasetId: number | string;
  // "incremental" after an append: only never-labeled rows are processed, and
  // every estimate row is a projection (the uncovered pool is a run-time fact)
  mode?: "full" | "incremental";
  // fired once labels exist, so the parent can offer the Ask tab
  onProcessed?: () => void;
  onError: (msg: string) => void;
}) {
  const [mode, setMode] = useState<Mode>({ name: "idle" });
  const [processed, setProcessed] = useState<boolean | null>(null);
  // A finished-successfully job found on mount in incremental mode: it
  // predates the append that opened this screen, so it is context, not the
  // result of the current action — noted next to the button, never rendered
  // as a fresh "Done" panel.
  const [priorJob, setPriorJob] = useState<PipelineJob | null>(null);
  const timer = useRef<number | null>(null);
  const notified = useRef(false);
  // consecutive status-poll failures; reset on every successful poll
  const failures = useRef(0);

  // On mount: is this dataset already processed, and is a job in flight? Both
  // matter because a run started in another tab (or from the CLI) is still the
  // truth about this dataset.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [p, job] = await Promise.all([
          pipelineProcessed(datasetId),
          pipelineStatus(datasetId),
        ]);
        if (!alive) return;
        setProcessed(p.is_processed);
        if (job && job.status === "running") {
          setMode({ name: "running", job });
        } else if (job && job.status === "done" && runMode === "incremental") {
          // After an append, the latest finished job covered the rows that
          // existed BEFORE it — showing its "Done" panel here read as "the
          // rows you just appended are processed", which is false.
          setPriorJob(job);
        } else if (job) {
          setMode({ name: "finished", job, preloaded: true });
        }
      } catch {
        if (alive) setProcessed(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [datasetId, runMode]);

  // poll while a job is running
  useEffect(() => {
    if (mode.name !== "running") return;
    let alive = true;
    async function tick() {
      try {
        const job = await pipelineStatus(datasetId);
        if (!alive) return;
        if (!job) {
          // no job for this dataset in this process (a server restart) — keep
          // polling rather than freezing on a stale "Running…" panel
          timer.current = window.setTimeout(tick, POLL_MS);
          return;
        }
        failures.current = 0;
        if (job.status === "running") {
          setMode({ name: "running", job });
          timer.current = window.setTimeout(tick, POLL_MS);
        } else {
          setMode({ name: "finished", job });
          setProcessed(job.is_processed);
        }
      } catch (err) {
        if (!alive) return;
        // The job keeps running server-side, so a blip must not end polling —
        // that used to leave the panel stuck on "Running…" until a reload.
        // Back off, and only surface an error once we've genuinely lost it.
        failures.current += 1;
        if (failures.current >= MAX_POLL_FAILURES) {
          onError(
            `Lost contact with the pipeline job after ${MAX_POLL_FAILURES} attempts: ` +
              (err instanceof Error ? err.message : String(err)) +
              " — the run may still be going; reopen this screen to re-check.",
          );
          setMode({ name: "idle" });
          return;
        }
        timer.current = window.setTimeout(tick, POLL_MS * failures.current);
      }
    }
    timer.current = window.setTimeout(tick, POLL_MS);
    return () => {
      alive = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [mode, datasetId, onError]);

  useEffect(() => {
    if (processed && !notified.current) {
      notified.current = true;
      onProcessed?.();
    }
  }, [processed, onProcessed]);

  async function handleEstimate() {
    setMode({ name: "estimating" });
    try {
      const estimate = await pipelineEstimate(datasetId, runMode);
      setMode({ name: "review", estimate });
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
      setMode({ name: "idle" });
    }
  }

  async function handleRun() {
    try {
      const job = await pipelineRun(datasetId, 60, runMode);
      failures.current = 0;
      setMode({ name: "running", job });
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  }

  const job = mode.name === "running" || mode.name === "finished" ? mode.job : null;

  return (
    <div className="pl">
      <div className="pl-head">
        <h4>Process this dataset</h4>
        {processed !== null && (
          <span className={`pl-badge${processed ? " pl-badge-ok" : ""}`}>
            {processed ? "ready to ask" : "not processed yet"}
          </span>
        )}
      </div>
      <p className="pl-hint">
        Builds the taxonomy, labels every response, then derives the keyword and
        place layers. Until this runs, the Ask tab has nothing to answer from.
      </p>

      {mode.name === "idle" && (
        <>
          {priorJob && (
            <p className="pl-hint">
              The last pipeline run finished{" "}
              {fmtWhen(priorJob.finished_utc)} (spent{" "}
              {usd(priorJob.cost_usd)}) — before this append, so it does not
              cover the rows just added. "Process new rows only" plans exactly
              those.
            </p>
          )}
          <button onClick={handleEstimate}>
            {runMode === "incremental"
              ? "Process new rows only"
              : processed
                ? "Re-process (adds a new run)"
                : "Process this dataset"}
          </button>
        </>
      )}
      {mode.name === "estimating" && <button disabled>Planning…</button>}

      {mode.name === "review" && (
        <div className="pl-plan">
          <div className="pl-plan-head">
            <strong>
              {mode.estimate.mode === "incremental"
                ? `Plan — ${mode.estimate.n_new_responses} new responses (of ${mode.estimate.n_responses} total)`
                : `Plan — ${mode.estimate.n_responses} responses across ${mode.estimate.n_questions} question${mode.estimate.n_questions === 1 ? "" : "s"}`}
            </strong>
            <span className="pl-free">no API calls made yet</span>
          </div>
          <table className="pl-table">
            <tbody>
              {mode.estimate.items.map((it, i) => (
                <tr key={`${it.stage}-${it.question_id}-${i}`}>
                  <td className="pl-stage">{it.stage}</td>
                  <td className="pl-q">{it.question_id ? `q${it.question_id}` : ""}</td>
                  <td className="pl-detail">{it.detail}</td>
                  <td>
                    <span
                      className={`pl-basis pl-basis-${it.basis}`}
                      title={
                        it.basis === "planned"
                          ? "From the real chunking and prompt sizes this run will use"
                          : "Extrapolated from a measured rate — the real dry-run needs an artifact that doesn't exist yet"
                      }
                    >
                      {it.basis}
                    </span>
                    <span
                      className="pl-input-basis"
                      title={INPUT_BASIS_TITLE[it.input_basis] ?? it.input_basis}
                    >
                      {it.input_basis}
                    </span>
                  </td>
                  <td className="pl-cost">{usd(it.est_cost_usd)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td colSpan={4}>
                  estimated total
                  <span className="pl-sub">
                    {" "}
                    · {mode.estimate.est_input_tokens.toLocaleString()} in /{" "}
                    {mode.estimate.est_output_tokens.toLocaleString()} out tokens
                  </span>
                </td>
                <td className="pl-cost">
                  {mode.estimate.priced ? (
                    <>
                      <strong>{usd(mode.estimate.est_total_usd)}</strong>
                      <span className="pl-band">
                        {usdRange(
                          mode.estimate.est_low_usd,
                          mode.estimate.est_high_usd,
                        )}
                      </span>
                    </>
                  ) : (
                    <strong className="pl-unpriced">no rate set</strong>
                  )}
                </td>
              </tr>
            </tfoot>
          </table>
          {!mode.estimate.priced && (
            <p className="pl-warn">
              ⚠ No token price is configured for{" "}
              <code>{mode.estimate.model_id}</code>, so the token counts above
              are real but the dollars are unknown. Add a rate on the Settings
              screen to price this run.
            </p>
          )}
          <p className="pl-hint">
            An estimate, not a quote. Prompt tokens were counted with Vertex's
            countTokens ({mode.estimate.count_calls} free calls, no model runs);
            output tokens cannot be counted in advance by anyone, so those come
            from rates measured over{" "}
            {mode.estimate.calibration_source === "measured"
              ? "this install's own completed runs"
              : "built-in defaults — recalibrate on the Settings screen to fit them to your data"}
            . The range is the 10th–90th percentile spread of those runs; the
            real figure comes from each stage's own manifest as it finishes.
          </p>
          {mode.estimate.mode === "incremental" && (
            <p className="pl-hint">
              Incremental: already-labelled responses are untouched; only new
              rows are labelled, against the existing taxonomy. Rows it can't
              place may add new categories automatically — every figure
              above is a projection.
            </p>
          )}
          {mode.estimate.mode !== "incremental" &&
            mode.estimate.questions_with_existing_taxonomy.length > 0 && (
            <p className="pl-warn">
              ⚠ Question
              {mode.estimate.questions_with_existing_taxonomy.length === 1 ? " " : "s "}
              {mode.estimate.questions_with_existing_taxonomy
                .map((q) => `q${q}`)
                .join(", ")}{" "}
              already {mode.estimate.questions_with_existing_taxonomy.length === 1
                ? "has a taxonomy"
                : "have taxonomies"}
              . Running again adds a new versioned run and re-spends — it does
              not skip them. Old runs are never overwritten.
            </p>
          )}
          <div className="pl-actions">
            <button onClick={handleRun}>
              {/* the TOP of the band, not the headline: "up to" the midpoint
                  is not an upper bound, and this is the last thing read
                  before money is spent */}
              Confirm and run — around {usd(mode.estimate.est_total_usd)}
              {mode.estimate.priced &&
                mode.estimate.est_high_usd > mode.estimate.est_total_usd &&
                `, up to ${usd(mode.estimate.est_high_usd)}`}
            </button>
            <button onClick={() => setMode({ name: "idle" })}>Cancel</button>
          </div>
        </div>
      )}

      {job && (
        <div className="pl-job">
          <div className="pl-job-head">
            <strong>
              {job.status === "running"
                ? "Running…"
                : job.status === "done"
                  ? "Done"
                  : "Failed"}
            </strong>
            {mode.name === "finished" && mode.preloaded && (
              <span className="pl-sub">
                earlier run — finished {fmtWhen(job.finished_utc)}
              </span>
            )}
            <span className="pl-job-cost">
              spent {usd(job.cost_usd)}
              <span className="pl-sub"> (from each stage's manifest)</span>
            </span>
          </div>
          <ul className="pl-stages">
            {job.stages.map((s) => (
              <li key={s.key} className={`pl-st pl-st-${s.status}`}>
                <span className="pl-st-icon">{STAGE_ICON[s.status] ?? "○"}</span>
                <span className="pl-st-label">{s.label}</span>
                <span className="pl-st-meta">
                  {s.cost_usd != null && <span>{usd(s.cost_usd)}</span>}
                  {s.seconds != null && <span>{s.seconds}s</span>}
                </span>
                {s.error && (
                  <span className="pl-st-err">
                    {s.error}
                    {s.detail ? ` — ${s.detail}` : ""}
                  </span>
                )}
              </li>
            ))}
          </ul>
          {job.status === "running" && (
            <p className="pl-hint">
              Runs on the server, so you can leave this screen. Don't restart
              the backend while it's going — a reload kills the run (induction
              checkpoints its expensive phase, so re-running skips what it
              already paid for).
            </p>
          )}
          {job.status === "failed" && (
            <p className="pl-warn">
              ⚠ {job.error || "A stage failed."} Remaining stages were skipped.
              Full output: <code>data/jobs/{job.dataset_id}/{job.job_id}/log.txt</code>
            </p>
          )}
          {job.status === "done" && (
            <p className="pl-ok">
              The Ask tab can answer questions about this dataset now.
            </p>
          )}
          {job.status !== "running" && (
            <div className="pl-actions">
              <button onClick={() => setMode({ name: "idle" })}>Close</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
