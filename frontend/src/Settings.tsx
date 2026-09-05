import { useCallback, useEffect, useState } from "react";
import {
  clearStoredPasscode,
  getConfig,
  putConfig,
  recalibrateCosts,
  setStoredPasscode,
} from "./api";
import type { CalibrationInfo, ConfigPatch, ModelPrice, ServerConfig } from "./types";
import "./settings.css";

/* Server settings: the admin passcode, which Gemini model each stage runs on,
   what tokens cost, and where the cost estimate's rates came from. Everything
   here is stored in config.json on the server, which the screen names so the
   file stays findable — and editable — without the app.

   The screen is admin-gated because GET /api/config is. When no passcode is
   configured the gate is off and this opens normally, which is what lets
   someone set the first one. */

function perToken(perMTok: number): string {
  // What people ask for when they say "cost per token". Shown next to the
  // per-million figure rather than instead of it: per-token is unreadable at
  // 0.0000003, and per-million is what the pricing page prints.
  if (perMTok === 0) return "free";
  return `$${(perMTok / 1_000_000).toExponential(2)} / token`;
}

function fmtWhen(stamp: string): string {
  if (!stamp) return "";
  const d = new Date(stamp.replace(/T(\d{2})-(\d{2})-(\d{2})/, "T$1:$2:$3"));
  return isNaN(d.getTime()) ? stamp : d.toLocaleString();
}

export default function Settings({ onError }: { onError: (msg: string) => void }) {
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [locked, setLocked] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // draft state — the form is not applied until Save
  const [defaultModel, setDefaultModel] = useState("");
  const [synthModel, setSynthModel] = useState("");
  const [prices, setPrices] = useState<ModelPrice[]>([]);
  const [passcode, setPasscode] = useState("");
  const [passcodeAgain, setPasscodeAgain] = useState("");
  const [clearing, setClearing] = useState(false);
  const [vertexProject, setVertexProject] = useState("");
  const [vertexLocation, setVertexLocation] = useState("");

  const load = useCallback(async () => {
    try {
      const next = await getConfig();
      setConfig(next);
      setLocked(false);
      setDefaultModel(next.default_model);
      setSynthModel(next.synth_model);
      /* The CONFIGURED value, not the effective one — an empty box means
         "let ADC decide", and prefilling it with the resolved project would
         turn the next Save into a silent pin. */
      setVertexProject(next.vertex_project ?? "");
      setVertexLocation(next.vertex_location ?? "");
      setPrices(next.prices);
    } catch (e) {
      /* A 401 here is not an error to shout about — it is the gate doing its
         job, and the user has a passcode box right below. adminFetch already
         prompted once and failed, so show the lock rather than a red banner. */
      if (String(e).includes("401")) setLocked(true);
      else onError(String(e));
    }
  }, [onError]);

  useEffect(() => {
    void load();
  }, [load]);

  if (locked) {
    return (
      <div className="cat set">
        <div className="cat-header">
          <h2>Settings</h2>
        </div>
        <div className="set-locked">
          <p>
            <strong>This screen needs the admin passcode.</strong>
          </p>
          <p className="cat-field-hint">
            If nobody knows it, it can be read or cleared directly in the
            server's <code>config.json</code>.
          </p>
          <label className="cat-field">
            <span className="cat-field-label">Admin passcode</span>
            <input
              type="password"
              value={passcode}
              onChange={(e) => setPasscode(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && passcode.trim()) {
                  setStoredPasscode(passcode);
                  setPasscode("");
                  void load();
                }
              }}
            />
          </label>
          <div className="cat-modal-actions">
            <button
              className="cat-btn-primary"
              disabled={!passcode.trim()}
              onClick={() => {
                setStoredPasscode(passcode);
                setPasscode("");
                void load();
              }}
            >
              Unlock
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (config === null) {
    return (
      <div className="cat set">
        <p>Loading settings…</p>
      </div>
    );
  }

  const passcodeMismatch =
    passcode.trim() !== "" && passcode.trim() !== passcodeAgain.trim();
  const envLocksPasscode = config.admin_passcode_source === "env";

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    const patch: ConfigPatch = {
      default_model: defaultModel.trim(),
      synth_model: synthModel.trim(),
      vertex_project: vertexProject.trim(),
      vertex_location: vertexLocation.trim(),
      prices: prices.map((p) => ({
        model_id: p.model_id,
        input_per_mtok: p.input_per_mtok,
        output_per_mtok: p.output_per_mtok,
      })),
    };
    if (clearing) patch.admin_passcode = "";
    else if (passcode.trim()) patch.admin_passcode = passcode.trim();

    try {
      const next = await putConfig(patch);
      /* Keep this tab working across a passcode change. Without this the
         save succeeds and the very next request 401s, which reads as the
         save having failed. */
      if (clearing) clearStoredPasscode();
      else if (passcode.trim()) setStoredPasscode(passcode.trim());
      setConfig(next);
      setPrices(next.prices);
      setVertexProject(next.vertex_project ?? "");
      setVertexLocation(next.vertex_location ?? "");
      setPasscode("");
      setPasscodeAgain("");
      setClearing(false);
      setSaved(true);
    } catch (e) {
      onError(String(e));
    }
    setSaving(false);
  };

  return (
    <div className="cat set">
      <div className="cat-header">
        <h2>Settings</h2>
        <span className="set-path" title={config.config_path}>
          {config.config_exists ? "config.json" : "config.json (not created yet)"}
        </span>
      </div>

      {config.env_overrides.length > 0 && (
        <p className="set-warn">
          ⚠ {config.env_overrides.join(", ")}{" "}
          {config.env_overrides.length === 1 ? "is" : "are"} set as environment
          variable{config.env_overrides.length === 1 ? "" : "s"} on the server
          and outrank this file. Saving here will not change{" "}
          {config.env_overrides.length === 1 ? "it" : "them"} until{" "}
          {config.env_overrides.length === 1 ? "it is" : "they are"} unset.
        </p>
      )}

      {/* --- passcode --- */}
      <section className="set-section">
        <h3>Admin passcode</h3>
        <p className="cat-field-hint">
          Gates uploading, appending, editing, processing, and deleting.
          Reading and asking questions stay open to anyone who can reach the
          app. With no passcode set, nothing is gated — the right setting for a
          single-user desktop install, the wrong one on a shared network.
        </p>
        <p className="set-status">
          {config.admin_passcode_set ? (
            <>
              <span className="set-dot set-dot-on" /> A passcode is set
              {config.admin_passcode_source === "env" && " (from the environment)"}
              {config.admin_passcode_source === "config" && " (from config.json)"}
            </>
          ) : (
            <>
              <span className="set-dot set-dot-off" /> No passcode — every
              action is open
            </>
          )}
        </p>

        {envLocksPasscode ? (
          <p className="cat-field-hint">
            The passcode comes from the ADMIN_PASSCODE environment variable, so
            it cannot be changed from here. Unset it on the server to manage
            the passcode in config.json instead.
          </p>
        ) : clearing ? (
          <p className="set-danger-note">
            The passcode will be removed when you save. Every action becomes
            open to anyone who can reach this app.{" "}
            <button className="set-linkish" onClick={() => setClearing(false)}>
              Keep it
            </button>
          </p>
        ) : (
          <>
            <div className="set-row">
              <label className="cat-field">
                <span className="cat-field-label">
                  {config.admin_passcode_set ? "New passcode" : "Set a passcode"}
                </span>
                <input
                  type="password"
                  value={passcode}
                  placeholder="leave blank to keep the current one"
                  onChange={(e) => setPasscode(e.target.value)}
                />
              </label>
              <label className="cat-field">
                <span className="cat-field-label">Repeat it</span>
                <input
                  type="password"
                  value={passcodeAgain}
                  onChange={(e) => setPasscodeAgain(e.target.value)}
                />
              </label>
            </div>
            {passcodeMismatch && (
              <p className="set-danger-note">The two passcodes do not match.</p>
            )}
            {config.admin_passcode_set && (
              <button className="set-linkish" onClick={() => setClearing(true)}>
                Remove the passcode instead
              </button>
            )}
          </>
        )}
      </section>

      {/* --- vertex --- */}
      <section className="set-section">
        <h3>Google Cloud</h3>
        <p className="cat-field-hint">
          Which Vertex AI project and region the model calls bill to. Both are
          optional: leave them blank and the project comes from whatever{" "}
          <code>gcloud auth application-default login</code> recorded on this
          machine, and the region is the global endpoint. Set them only to
          override that — for a second project, or a region you need for data
          residency.
        </p>
        <div className="set-row">
          <label className="cat-field">
            <span className="cat-field-label">Project id</span>
            <input
              type="text"
              value={vertexProject}
              placeholder={config.vertex_project_effective ?? "not set anywhere"}
              onChange={(e) => setVertexProject(e.target.value)}
            />
            <span className="cat-field-hint">
              {config.vertex_project
                ? "Configured here."
                : config.vertex_project_effective
                  ? `Blank — using ${config.vertex_project_effective}, from your gcloud credentials.`
                  : "Blank, and no project could be inferred from your credentials — model calls will fail until this is set or you run gcloud auth application-default login."}
            </span>
          </label>
          <label className="cat-field">
            <span className="cat-field-label">Region</span>
            <input
              type="text"
              value={vertexLocation}
              placeholder={config.vertex_location_effective}
              onChange={(e) => setVertexLocation(e.target.value)}
            />
            <span className="cat-field-hint">
              {config.vertex_location
                ? "Configured here."
                : `Blank — using ${config.vertex_location_effective}, the default.`}
            </span>
          </label>
        </div>
      </section>

      {/* --- models --- */}
      <section className="set-section">
        <h3>Models</h3>
        <p className="cat-field-hint">
          Any model id the Vertex AI endpoint accepts. Add a price for it below
          or its runs report as unpriced — the token counts stay right, the
          dollars go missing.
        </p>
        <div className="set-row">
          <label className="cat-field">
            <span className="cat-field-label">Workhorse</span>
            <input
              type="text"
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
            />
            <span className="cat-field-hint">
              Induction, labeling, sub-themes, routing — everything whose call
              count grows with the corpus. Hundreds to thousands of calls per
              run, so this is the choice that decides what a dataset costs.
            </span>
          </label>
          <label className="cat-field">
            <span className="cat-field-label">Answer writing</span>
            <input
              type="text"
              value={synthModel}
              onChange={(e) => setSynthModel(e.target.value)}
            />
            <span className="cat-field-hint">
              One call per question asked, so a pricier model is affordable
              here. Note that thinking tokens bill at the output rate and can
              be several times the visible answer.
            </span>
          </label>
        </div>
      </section>

      {/* --- prices --- */}
      <section className="set-section">
        <h3>Token prices</h3>
        <p className="cat-field-hint">
          USD per 1,000,000 tokens, as published on the Vertex AI pricing page.
          The output rate covers thinking tokens too — they are billed, just
          invisible — so there is no separate rate to set for them.
        </p>
        <table className="set-prices">
          <thead>
            <tr>
              <th>Model</th>
              <th>Input / 1M</th>
              <th>Output / 1M</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {prices.map((row, i) => (
              <tr key={row.model_id}>
                <td>
                  <input
                    type="text"
                    value={row.model_id}
                    aria-label="Model id"
                    onChange={(e) =>
                      setPrices((rows) =>
                        rows.map((r, j) =>
                          j === i ? { ...r, model_id: e.target.value } : r,
                        ),
                      )
                    }
                  />
                </td>
                {(["input_per_mtok", "output_per_mtok"] as const).map((field) => (
                  <td key={field}>
                    <input
                      type="number"
                      min={0}
                      step="0.01"
                      value={row[field]}
                      aria-label={field}
                      onChange={(e) =>
                        setPrices((rows) =>
                          rows.map((r, j) =>
                            j === i
                              ? { ...r, [field]: Number(e.target.value) || 0 }
                              : r,
                          ),
                        )
                      }
                    />
                    <span className="set-per-token">{perToken(row[field])}</span>
                  </td>
                ))}
                <td>
                  <button
                    className="set-linkish"
                    aria-label={`Remove the price for ${row.model_id}`}
                    onClick={() =>
                      setPrices((rows) => rows.filter((_, j) => j !== i))
                    }
                  >
                    remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          className="set-linkish"
          onClick={() =>
            setPrices((rows) => [
              ...rows,
              { model_id: "", input_per_mtok: 0, output_per_mtok: 0, overridden: true },
            ])
          }
        >
          + add a model
        </button>
      </section>

      {/* --- calibration --- */}
      <CalibrationPanel
        calibration={config.calibration}
        onError={onError}
        onCalibrated={(next) =>
          setConfig((c) => (c ? { ...c, calibration: next } : c))
        }
      />

      <div className="set-actions">
        <button
          className="cat-btn-primary"
          onClick={handleSave}
          disabled={
            saving ||
            passcodeMismatch ||
            !defaultModel.trim() ||
            !synthModel.trim() ||
            prices.some((p) => !p.model_id.trim())
          }
        >
          {saving ? "Saving…" : "Save settings"}
        </button>
        {saved && <span className="set-saved">Saved.</span>}
      </div>
      <p className="cat-field-hint">
        Written to <code>{config.config_path}</code>. Changes take effect on
        the next request — no restart.
      </p>
    </div>
  );
}

function CalibrationPanel({
  calibration,
  onCalibrated,
  onError,
}: {
  calibration: CalibrationInfo;
  onCalibrated: (next: CalibrationInfo) => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const measured = calibration.source === "measured";
  const runs = Object.entries(calibration.sample ?? {});
  const totalRuns = runs.reduce((sum, [, n]) => sum + n, 0);

  return (
    <section className="set-section">
      <h3>Cost estimate accuracy</h3>
      <p className="cat-field-hint">
        Before a run, prompt tokens are counted with Vertex's countTokens —
        free, and not a model call. Output tokens cannot be counted in advance
        by anyone, so those rates are measured from runs that already finished
        on this machine.
      </p>
      <p className="set-status">
        {measured ? (
          <>
            <span className="set-dot set-dot-on" /> Measured from {totalRuns} of
            this install's own runs
            {calibration.measured_utc &&
              ` on ${fmtWhen(calibration.measured_utc)}`}
          </>
        ) : (
          <>
            <span className="set-dot set-dot-off" /> Using built-in rates — this
            install has not measured its own yet
          </>
        )}
      </p>
      <dl className="set-rates">
        <div>
          <dt>Labeling, per response</dt>
          <dd>
            {calibration.label_input_tokens_per_response} in /{" "}
            {calibration.label_output_tokens_per_response} out tokens
          </dd>
        </div>
        <div>
          <dt>Induction output, per chunk</dt>
          <dd>{calibration.induce_output_tokens_per_chunk} tokens</dd>
        </div>
        <div>
          <dt>Candidates proposed, per response</dt>
          <dd>{calibration.candidates_per_response}</dd>
        </div>
        <div>
          <dt>Spread of the estimate</dt>
          <dd>
            {Math.round(calibration.spread_low * 100)}–
            {Math.round(calibration.spread_high * 100)}% of the headline figure
          </dd>
        </div>
      </dl>
      {runs.length > 0 && (
        <p className="cat-field-hint">
          Sample:{" "}
          {runs
            .map(([key, n]) => `${n} ${key.replace(/_runs$/, "")}`)
            .join(", ")}
          . Runs smaller than 200 responses are excluded — the smoke-test
          datasets produce rates several times off the real ones.
        </p>
      )}
      <button
        className="cat-btn-secondary"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          try {
            onCalibrated(await recalibrateCosts());
          } catch (e) {
            onError(String(e));
          }
          setBusy(false);
        }}
      >
        {busy ? "Measuring…" : "Recalibrate from completed runs"}
      </button>
      <span className="cat-field-hint set-inline-hint">
        Reads run manifests already on disk. Free, offline, and makes no model
        calls.
      </span>
    </section>
  );
}
