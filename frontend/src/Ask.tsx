import { useEffect, useMemo, useRef, useState } from "react";
import { askAnswer, askRoute } from "./api";
import type {
  AskAnswerResponse,
  AskCandidate,
  AskRouteResponse,
  AskSource,
} from "./types";
import "./ask.css";

// Stateless two-step flow: this component holds the routing proposal in
// React state while the analyst edits checkboxes, then sends back only the
// choices. Refreshing the page loses an in-flight question — accepted
// trade-off of keeping the server stateless.

type Phase =
  | { name: "question" }
  | { name: "routing"; question: string }
  | { name: "review"; question: string; proposal: AskRouteResponse }
  | { name: "answering"; question: string; proposal: AskRouteResponse }
  | {
      name: "answer";
      question: string;
      proposal: AskRouteResponse;
      result: AskAnswerResponse;
    };

// The labeling pass marks every response "specific" or "general"; this is the
// analyst-facing wording for that axis. Counts come from the proposal, so an
// option whose value the data never carries is hidden rather than offered.
const ACTIONABILITY_CHOICES = [
  {
    value: "",
    label: "All responses",
    help: "no filter on response type",
  },
  {
    value: "specific",
    label: "Concrete suggestions only",
    help: "responses proposing a specific, implementable action",
  },
  {
    value: "general",
    label: "General concerns only",
    help: "broad wishes and complaints, not specific proposals",
  },
];

const RELEVANCE_COLOR: Record<string, string> = {
  high: "#166534",
  medium: "#92400e",
  low: "#6b7280",
};

export default function Ask({
  datasetId,
  onError,
}: {
  datasetId: number;
  onError: (msg: string) => void;
}) {
  const [phase, setPhase] = useState<Phase>({ name: "question" });

  async function handleAsk(question: string) {
    setPhase({ name: "routing", question });
    try {
      const proposal = await askRoute(datasetId, question);
      setPhase({ name: "review", question, proposal });
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
      setPhase({ name: "question" });
    }
  }

  async function handleConfirm(
    question: string,
    proposal: AskRouteResponse,
    selectedIds: Set<string>,
    concepts: Set<string>,
    places: Set<string>,
    groupByLocation: boolean,
    actionability: string,
    eventsOnly: boolean,
  ) {
    setPhase({ name: "answering", question, proposal });
    try {
      const result = await askAnswer(datasetId, {
        question,
        route: proposal.route,
        reason: proposal.reason,
        selected: proposal.candidates
          .filter((c) => selectedIds.has(c.label_id))
          .map((c) => ({
            label_id: c.label_id,
            relevance: c.relevance,
            rationale: c.rationale,
          })),
        lexicon_concepts: [...concepts],
        group_by: groupByLocation ? "location" : "category",
        location_filter: [...places],
        actionability_filter: actionability,
        event_filter: eventsOnly ? "reported" : "",
        proposed_label_ids: proposal.candidates.map((c) => c.label_id),
      });
      setPhase({ name: "answer", question, proposal, result });
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
      setPhase({ name: "review", question, proposal });
    }
  }

  return (
    <section>
      {(phase.name === "question" || phase.name === "routing") && (
        <QuestionForm
          busy={phase.name === "routing"}
          onAsk={handleAsk}
        />
      )}

      {(phase.name === "review" || phase.name === "answering") && (
        <ReviewPanel
          question={phase.question}
          proposal={phase.proposal}
          busy={phase.name === "answering"}
          onConfirm={(
            ids,
            concepts,
            places,
            groupByLocation,
            actionability,
            eventsOnly,
          ) =>
            handleConfirm(
              phase.question,
              phase.proposal,
              ids,
              concepts,
              places,
              groupByLocation,
              actionability,
              eventsOnly,
            )
          }
          onBack={() => setPhase({ name: "question" })}
        />
      )}

      {phase.name === "answer" && (
        <AnswerView
          question={phase.question}
          proposal={phase.proposal}
          result={phase.result}
          onReset={() => setPhase({ name: "question" })}
        />
      )}
    </section>
  );
}

function QuestionForm({
  busy,
  onAsk,
}: {
  busy: boolean;
  onAsk: (q: string) => void;
}) {
  const [question, setQuestion] = useState("");

  return (
    <div>
      <p>
        Ask a question about what respondents said — e.g.{" "}
        <em>"when residents mention affordability, what specific costs are they
        referring to?"</em>
      </p>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={3}
        style={{ width: "100%", fontFamily: "inherit", fontSize: "1em" }}
        placeholder="Your question…"
        disabled={busy}
      />
      <div style={{ marginTop: "0.5rem" }}>
        <button onClick={() => onAsk(question)} disabled={busy || !question.trim()}>
          {busy ? "Finding relevant categories…" : "Ask"}
        </button>
      </div>
    </div>
  );
}

function ReviewPanel({
  question,
  proposal,
  busy,
  onConfirm,
  onBack,
}: {
  question: string;
  proposal: AskRouteResponse;
  busy: boolean;
  onConfirm: (
    ids: Set<string>,
    concepts: Set<string>,
    places: Set<string>,
    groupByLocation: boolean,
    actionability: string,
    eventsOnly: boolean,
  ) => void;
  onBack: () => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(proposal.candidates.map((c) => c.label_id)),
  );
  const [concepts, setConcepts] = useState<Set<string>>(
    () => new Set(proposal.lexicon_concepts),
  );
  const [places, setPlaces] = useState<Set<string>>(
    () => new Set(proposal.location_filter),
  );
  const [groupByLocation, setGroupByLocation] = useState(
    proposal.group_by === "location",
  );
  const [actionability, setActionability] = useState(
    proposal.actionability_filter,
  );
  const [eventsOnly, setEventsOnly] = useState(
    proposal.event_filter === "reported",
  );

  const byQuestion = useMemo(() => {
    const groups = new Map<string, AskCandidate[]>();
    for (const c of proposal.candidates) {
      const list = groups.get(c.question_id) ?? [];
      list.push(c);
      groups.set(c.question_id, list);
    }
    return groups;
  }, [proposal.candidates]);

  function toggle(id: string) {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleConcept(name: string) {
    setConcepts((s) => {
      const next = new Set(s);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function togglePlace(name: string) {
    setPlaces((s) => {
      const next = new Set(s);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  if (!proposal.answerable) {
    return (
      <div>
        <p>
          <strong>“{question}”</strong>
        </p>
        <div style={{ padding: "1rem", background: "#fef2f2", border: "1px solid #fecaca" }}>
          <strong>This data can't answer that question.</strong>
          <p style={{ margin: "0.5rem 0 0" }}>
            {proposal.reason || "No relevant categories were found."} Nothing was
            matched rather than forcing the nearest plausible category.
          </p>
        </div>
        <button onClick={onBack} style={{ marginTop: "1rem" }}>
          Ask a different question
        </button>
      </div>
    );
  }

  return (
    <div>
      <p>
        <strong>“{question}”</strong>
      </p>
      <p>
        Answer strategy: <strong>{proposal.route}</strong> — {proposal.reason}
      </p>
      {proposal.warnings.map((w) => (
        <p key={w} style={{ color: "#b45309", fontSize: "0.9em" }}>
          ⚠ {w}
        </p>
      ))}

      <fieldset>
        <legend>
          Categories the answer will search — untick anything that doesn't
          belong
        </legend>
        {[...byQuestion.entries()].map(([qid, candidates]) => (
          <div key={qid} style={{ marginBottom: "0.75rem" }}>
            <div style={{ fontSize: "0.85em", color: "#555", marginBottom: "0.25rem" }}>
              Survey question {qid}: “{candidates[0].question_text}”
            </div>
            {candidates.map((c) => (
              <div key={c.label_id} style={{ marginLeft: "1rem", marginBottom: "0.4rem" }}>
                <label>
                  <input
                    type="checkbox"
                    checked={selected.has(c.label_id)}
                    onChange={() => toggle(c.label_id)}
                    disabled={busy}
                  />{" "}
                  <strong>{c.name}</strong>
                  {c.parent_name && (
                    <span style={{ color: "#555" }}> · {c.parent_name}</span>
                  )}{" "}
                  <span style={{ fontFamily: "monospace" }}>n={c.count}</span>{" "}
                  <span
                    style={{
                      color: RELEVANCE_COLOR[c.relevance] ?? "#6b7280",
                      fontSize: "0.85em",
                    }}
                  >
                    {c.relevance}
                  </span>
                </label>
                <div style={{ fontSize: "0.85em", color: "#555", marginLeft: "1.5rem" }}>
                  {c.rationale}
                </div>
              </div>
            ))}
          </div>
        ))}
      </fieldset>

      {proposal.available_lexicon_concepts.length > 0 && (
        <fieldset style={{ marginTop: "1rem" }}>
          <legend>
            Keyword concepts — exact-match mention counts added to the answer
          </legend>
          {proposal.available_lexicon_concepts.map((name) => (
            <label key={name} style={{ display: "inline-block", marginRight: "1rem" }}>
              <input
                type="checkbox"
                checked={concepts.has(name)}
                onChange={() => toggleConcept(name)}
                disabled={busy}
              />{" "}
              {name}
            </label>
          ))}
        </fieldset>
      )}

      {proposal.available_locations.length > 0 && (
        <fieldset style={{ marginTop: "1rem" }}>
          <legend>
            Places — restrict the evidence to responses mentioning a place, or
            organize the whole answer by place
          </legend>
          <label style={{ display: "block", marginBottom: "0.5rem" }}>
            <input
              type="checkbox"
              checked={groupByLocation}
              onChange={() => setGroupByLocation((v) => !v)}
              disabled={busy}
            />{" "}
            <strong>Organize the answer by place</strong> (for "where…"
            questions — counts and quotes grouped per place)
          </label>
          {places.size > 0 && (
            <p style={{ margin: "0 0 0.5rem", fontSize: "0.85em", color: "#555" }}>
              Only responses mentioning a ticked place will be used as evidence.
            </p>
          )}
          {proposal.available_locations.map((l) => (
            <label
              key={l.name}
              style={{ display: "inline-block", marginRight: "1rem" }}
            >
              <input
                type="checkbox"
                checked={places.has(l.name)}
                onChange={() => togglePlace(l.name)}
                disabled={busy}
              />{" "}
              {l.name}{" "}
              <span style={{ color: "#555", fontFamily: "monospace" }}>
                n={l.count}
              </span>
            </label>
          ))}
        </fieldset>
      )}

      {Object.keys(proposal.available_actionability).length > 0 && (
        <fieldset style={{ marginTop: "1rem" }}>
          <legend>
            Response type — answer from concrete suggestions only, or from
            everything
          </legend>
          {ACTIONABILITY_CHOICES.map((choice) => {
            const n = proposal.available_actionability[choice.value];
            if (choice.value && n == null) return null;
            return (
              <label key={choice.value || "any"} style={{ display: "block", marginBottom: "0.35rem" }}>
                <input
                  type="radio"
                  name="actionability"
                  checked={actionability === choice.value}
                  onChange={() => setActionability(choice.value)}
                  disabled={busy}
                />{" "}
                <strong>{choice.label}</strong>
                {n != null && (
                  <span style={{ color: "#555", fontFamily: "monospace" }}>
                    {" "}
                    n={n}
                  </span>
                )}
                <span style={{ color: "#555" }}> — {choice.help}</span>
              </label>
            );
          })}
          {actionability !== "" && (
            <p style={{ margin: "0.5rem 0 0", fontSize: "0.85em", color: "#555" }}>
              Counts and quotes will cover only this subset, and the answer will
              say so. Responses the labeling pass never marked either way are
              excluded — they are missing data, not evidence of absence.
            </p>
          )}
        </fieldset>
      )}

      {(proposal.available_events.reported ?? 0) > 0 && (
        <fieldset style={{ marginTop: "1rem" }}>
          <legend>
            First-hand experience — answer only from incidents respondents say
            happened to them
          </legend>
          <label style={{ display: "block" }}>
            <input
              type="checkbox"
              checked={eventsOnly}
              onChange={() => setEventsOnly((v) => !v)}
              disabled={busy}
            />{" "}
            <strong>Only responses describing something that happened</strong>{" "}
            <span style={{ color: "#555", fontFamily: "monospace" }}>
              n={proposal.available_events.reported}
            </span>
            <span style={{ color: "#555" }}>
              {" "}
              — of {proposal.available_events.coded} responses checked
            </span>
          </label>
          <p style={{ margin: "0.4rem 0 0", fontSize: "0.85em", color: "#555" }}>
            A specific incident ("my car window got smashed"), not an opinion or
            an ongoing condition. There is no option for the opposite: a
            response that doesn't describe an incident is not evidence that
            nothing happened to that person.
          </p>
        </fieldset>
      )}

      <fieldset style={{ marginTop: "1rem", color: "#999" }}>
        <legend style={{ color: "#999" }}>Respondent filters</legend>
        <p style={{ margin: 0, fontSize: "0.9em" }}>
          Filtering by demographics (district, etc.) needs demographic columns
          ingested first — coming with metadata column support.
        </p>
      </fieldset>

      <div style={{ marginTop: "1rem" }}>
        <button
          onClick={() =>
            onConfirm(
              selected,
              concepts,
              places,
              groupByLocation,
              actionability,
              eventsOnly,
            )
          }
          disabled={busy || selected.size === 0}
        >
          {busy
            ? "Writing answer…"
            : `Answer from ${selected.size} categor${selected.size === 1 ? "y" : "ies"}`}
        </button>{" "}
        <button onClick={onBack} disabled={busy}>
          Back
        </button>
        {selected.size === 0 && (
          <span style={{ marginLeft: "0.75rem", color: "#b45309", fontSize: "0.9em" }}>
            Select at least one category.
          </span>
        )}
      </div>
    </div>
  );
}

const ROUTE_DISPLAY: Record<string, string> = {
  retrieval: "Retrieval — read what respondents say",
  aggregate: "Aggregate — computed counts",
  comparative: "Comparative — contrast groups",
  hybrid: "Hybrid — counts first, then reasons",
};

const PLACES_SHOWN = 10;

// Mirrors router.filter_phrase — how the active evidence restrictions read in
// the counts panel, so the sidebar can't describe a narrower set than the
// answer was written from.
function filterPhrase(result: AskAnswerResponse): string {
  const parts: string[] = [];
  if (result.location_filter.length > 0) {
    parts.push(`that mention ${result.location_filter.join(", ")}`);
  }
  if (result.actionability_filter === "specific") {
    parts.push("that propose a concrete action");
  } else if (result.actionability_filter === "general") {
    parts.push("raising a general concern");
  }
  if (result.event_filter === "reported") {
    parts.push("recounting a first-hand incident");
  }
  return parts.join(" and ");
}

function AnswerView({
  question,
  proposal,
  result,
  onReset,
}: {
  question: string;
  proposal: AskRouteResponse;
  result: AskAnswerResponse;
  onReset: () => void;
}) {
  const nameOf = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of proposal.candidates) m.set(c.label_id, c.name);
    return m;
  }, [proposal.candidates]);

  const sourceByN = useMemo(
    () => new Map(result.sources.map((s) => [s.n, s])),
    [result.sources],
  );

  // grouped by place for "where" answers, by category otherwise
  const byLocation =
    result.location_counts.length > 0 && result.sources.some((s) => s.location);
  const sourceGroups = useMemo(() => {
    const groups = new Map<string, AskSource[]>();
    for (const s of result.sources) {
      const key = byLocation
        ? (s.location ?? "no place named")
        : (nameOf.get(s.label_id) ?? s.label_id);
      const list = groups.get(key) ?? [];
      list.push(s);
      groups.set(key, list);
    }
    return [...groups.entries()];
  }, [result.sources, byLocation, nameOf]);

  const articleRef = useRef<HTMLDivElement>(null);
  const sourcesRef = useRef<HTMLDetailsElement>(null);
  const [pop, setPop] = useState<{ n: number; top: number; left: number } | null>(null);
  const [flashN, setFlashN] = useState<number | null>(null);
  const [allPlaces, setAllPlaces] = useState(false);
  const [copied, setCopied] = useState(false);

  // the confirm button sits at the bottom of a long review screen — land
  // the analyst on the takeaway, not mid-answer
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      const el = e.target as HTMLElement;
      if (el.closest(".ask-popover") || el.closest(".ask-cite")) return;
      setPop(null);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPop(null);
    }
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  function handleCite(n: number, el: HTMLElement) {
    if (pop?.n === n) {
      setPop(null);
      return;
    }
    const wrap = articleRef.current;
    if (!wrap) return;
    const w = wrap.getBoundingClientRect();
    const b = el.getBoundingClientRect();
    const width = Math.min(420, w.width - 8);
    const left = Math.max(0, Math.min(b.left - w.left, w.width - width));
    setPop({ n, top: b.bottom - w.top + 8, left });
  }

  function jumpToSource(n: number) {
    setPop(null);
    if (sourcesRef.current) sourcesRef.current.open = true;
    setFlashN(null);
    requestAnimationFrame(() => {
      document
        .getElementById(`ask-src-${n}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
      setFlashN(n);
    });
  }

  async function copyAnswer() {
    const doc = [
      `# ${question}`,
      "",
      result.answer_markdown,
      "",
      "---",
      "**Sources** (response_key — verbatim):",
      "",
      ...result.sources.map((s) => `- [${s.n}] \`${s.response_key}\`: "${s.text}"`),
    ].join("\n");
    await navigator.clipboard.writeText(doc);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  const { stats } = result;
  const countRows = Object.entries(result.counts).sort(([, a], [, b]) => b - a);
  const countMax = Math.max(...countRows.map(([, n]) => n), 1);
  const placeRows = allPlaces
    ? result.location_counts
    : result.location_counts.slice(0, PLACES_SHOWN);
  const placeMax = Math.max(...result.location_counts.map((l) => l.count), 1);
  const groupMax = Math.max(
    ...result.group_counts.map((g) => g.count_unique_responses),
    1,
  );
  const nSampled = Object.keys(result.sampling_notes).length;
  const popSource = pop ? sourceByN.get(pop.n) : undefined;

  return (
    <div className="ask-answer">
      <p className="ask-eyebrow">Answer</p>
      <h2 className="ask-question">“{question}”</h2>
      <div className="ask-chips">
        <span className="ask-chip ask-chip-route">
          <span className="ask-dot" />
          {ROUTE_DISPLAY[proposal.route] ?? proposal.route}
          {byLocation ? " · grouped by place" : ""}
        </span>
        {result.location_filter.length > 0 && (
          <span className="ask-chip ask-chip-route">
            Only responses mentioning: {result.location_filter.join(", ")}
          </span>
        )}
        {result.actionability_filter !== "" && (
          <span className="ask-chip ask-chip-route">
            {result.actionability_filter === "specific"
              ? "Only concrete suggestions"
              : "Only general concerns"}
          </span>
        )}
        {result.event_filter === "reported" && (
          <span className="ask-chip ask-chip-route">
            Only first-hand incidents
          </span>
        )}
        {result.deselected.length > 0 && (
          <span className="ask-chip">
            {result.deselected.length} proposed categor
            {result.deselected.length === 1 ? "y" : "ies"} deselected
          </span>
        )}
        {result.invalid_citations > 0 && (
          <span className="ask-chip ask-chip-warn">
            ⚠ {result.invalid_citations} unresolved citation
            {result.invalid_citations === 1 ? "" : "s"}
          </span>
        )}
      </div>

      <dl className="ask-method">
        <div>
          <dt>Categories</dt>
          <dd>
            {stats.categories_searched}{" "}
            <span className="ask-sub">of {stats.categories_total}</span>
          </dd>
        </div>
        <div>
          <dt>Responses covered</dt>
          <dd>{stats.unique_responses}</dd>
        </div>
        {result.actionability_denominator && (
          <div>
            <dt>
              {result.actionability_filter === "specific"
                ? "Gave a suggestion"
                : "General concerns"}
            </dt>
            <dd>
              {result.actionability_denominator.matching}{" "}
              <span className="ask-sub">
                of {result.actionability_denominator.in_scope}
              </span>
            </dd>
          </div>
        )}
        {result.event_denominator && (
          <div>
            <dt>Described an incident</dt>
            <dd>
              {result.event_denominator.matching}{" "}
              <span className="ask-sub">
                of {result.event_denominator.in_scope}
              </span>
            </dd>
          </div>
        )}
        {result.location_denominator && (
          <div>
            <dt>Named a place</dt>
            <dd>
              {result.location_denominator.naming_any}{" "}
              <span className="ask-sub">
                of {result.location_denominator.in_scope}
              </span>
            </dd>
          </div>
        )}
        <div>
          <dt>Verbatims shown</dt>
          <dd>
            {stats.quotes_shown}
            {nSampled > 0 && <span className="ask-sub"> sampled</span>}
          </dd>
        </div>
        <div>
          <dt>Cited in answer</dt>
          <dd>{stats.quotes_cited}</dd>
        </div>
        <div>
          <dt>Unresolved citations</dt>
          <dd className={result.invalid_citations > 0 ? "ask-warn" : ""}>
            {result.invalid_citations}
          </dd>
        </div>
      </dl>
      {result.event_denominator && (
        <p className="ask-caveat">
          Self-reported first-hand accounts, not verified incidents. The other{" "}
          {result.event_denominator.in_scope -
            result.event_denominator.matching}{" "}
          in-scope responses did not describe an incident — which is not the
          same as nothing having happened to those respondents.
        </p>
      )}
      <details className="ask-method-full">
        <summary>How this answer was computed</summary>
        <p>{result.process_note}</p>
      </details>

      <div className="ask-grid">
        <div className="ask-article" ref={articleRef}>
          <AnswerMarkdown
            text={result.answer_markdown}
            sourceNs={sourceByN}
            activeN={pop?.n ?? null}
            onCite={handleCite}
          />

          <details className="ask-sources" ref={sourcesRef}>
            <summary>
              Sources{" "}
              <span className="ask-cnt">
                — {result.sources.length} cited response
                {result.sources.length === 1 ? "" : "s"}, each traceable to its
                survey row
              </span>
            </summary>
            {sourceGroups.map(([title, items]) => (
              <div key={title} className="ask-src-group">
                <h6>{title}</h6>
                {items.map((s) => (
                  <div
                    key={s.n}
                    id={`ask-src-${s.n}`}
                    className={`ask-src-item${flashN === s.n ? " ask-src-flash" : ""}`}
                  >
                    <div className="ask-meta">
                      <span className="ask-n">[{s.n}]</span> {s.response_key}
                    </div>
                    <blockquote>“{s.text}”</blockquote>
                  </div>
                ))}
              </div>
            ))}
          </details>

          {pop && popSource && (
            <div
              className="ask-popover"
              role="dialog"
              aria-label="Cited response"
              style={{ top: pop.top, left: pop.left }}
            >
              <div className="ask-popover-head">
                <span>
                  [{pop.n}] · {popSource.response_key}
                </span>
                <span className="ask-popover-tag">
                  {byLocation && popSource.location
                    ? popSource.location
                    : (nameOf.get(popSource.label_id) ?? popSource.label_id)}
                </span>
              </div>
              <blockquote>“{popSource.text}”</blockquote>
              <button
                className="ask-popover-jump"
                onClick={() => jumpToSource(pop.n)}
              >
                Jump to source ↓
              </button>
            </div>
          )}
        </div>

        <aside>
          <div className="ask-panel">
            <h5>Categories searched</h5>
            <p className="ask-hint">
              {filterPhrase(result)
                ? `Responses in each category ${filterPhrase(result)} — “of N” is the category's full size.`
                : "Real counts from the coded data, never estimated."}
            </p>
            {countRows.map(([lid, n]) => (
              <div
                key={lid}
                className="ask-bar-row"
                title={
                  result.sampling_notes[lid]
                    ? `${nameOf.get(lid) ?? lid} — answer quoted a sample (${result.sampling_notes[lid]})`
                    : (nameOf.get(lid) ?? lid)
                }
              >
                <div className="ask-bar-label">
                  <span className="ask-nm">{nameOf.get(lid) ?? lid}</span>
                  <span className="ask-ct">
                    {n}
                    {result.counts_unfiltered[lid] != null && (
                      <span className="ask-sub"> of {result.counts_unfiltered[lid]}</span>
                    )}
                  </span>
                </div>
                <div className="ask-bar-track">
                  <div
                    className="ask-bar-fill"
                    style={{ width: `${Math.max((n / countMax) * 100, 2)}%` }}
                  />
                </div>
              </div>
            ))}
            {nSampled > 0 && (
              <div className="ask-sampling-note">
                Quotes shown to the model were sampled for {nSampled}{" "}
                {byLocation ? "place" : "categor"}
                {nSampled === 1 ? (byLocation ? "" : "y") : byLocation ? "s" : "ies"}
                ; counts always cover the full data.
              </div>
            )}
          </div>

          {result.location_counts.length > 0 && (
            <div className="ask-panel">
              <h5>Where — places named</h5>
              {result.location_denominator && (
                <p className="ask-hint">
                  {result.location_denominator.naming_any} of{" "}
                  {result.location_denominator.in_scope} in-scope responses
                  named a place — only those are localizable.
                </p>
              )}
              {placeRows.map((l) => (
                <div
                  key={l.name}
                  className="ask-bar-row"
                  title={`${l.count} in-scope responses mention “${l.name}”`}
                >
                  <div className="ask-bar-label">
                    <span className="ask-nm">
                      {l.name}
                      {l.kind === "named" && (
                        <span className="ask-tag-named">named</span>
                      )}
                    </span>
                    <span className="ask-ct">{l.count}</span>
                  </div>
                  <div className="ask-bar-track">
                    <div
                      className="ask-bar-fill"
                      style={{ width: `${Math.max((l.count / placeMax) * 100, 2)}%` }}
                    />
                  </div>
                </div>
              ))}
              {result.location_counts.length > PLACES_SHOWN && (
                <button
                  className="ask-more-link"
                  onClick={() => setAllPlaces((v) => !v)}
                >
                  {allPlaces
                    ? "Show fewer places"
                    : `Show ${result.location_counts.length - PLACES_SHOWN} more places…`}
                </button>
              )}
            </div>
          )}

          {result.group_counts.length > 0 && (
            <div className="ask-panel">
              <h5>Compared groups</h5>
              {result.group_counts.map((g) => (
                <div key={g.name} className="ask-bar-row">
                  <div className="ask-bar-label">
                    <span className="ask-nm">{g.name}</span>
                    <span className="ask-ct">{g.count_unique_responses}</span>
                  </div>
                  <div className="ask-bar-track">
                    <div
                      className="ask-bar-fill"
                      style={{
                        width: `${Math.max((g.count_unique_responses / groupMax) * 100, 2)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          {result.lexicon_counts.length > 0 && (
            <div className="ask-panel">
              <h5>Keyword mentions (exact match)</h5>
              <ul className="ask-kw-list">
                {result.lexicon_counts.map((lc) => (
                  <li
                    key={lc.concept}
                    title={Object.entries(lc.mentions_by_question)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([q, n]) => `q${q}: ${n}`)
                      .join(", ")}
                  >
                    <span>{lc.concept}</span>
                    <span className="ask-ct">{lc.mentions_total}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>
      </div>

      <div className="ask-runbar">
        <span>
          Saved as run <code>{result.run_id}</code> under <code>data/answers/</code>
        </span>
        <span className="ask-actions">
          <button onClick={copyAnswer}>{copied ? "Copied ✓" : "Copy answer"}</button>
          <button onClick={onReset}>Ask another question</button>
        </span>
      </div>
    </div>
  );
}

// Minimal renderer for the small markdown subset the synthesis prompt
// produces: "### " subheadings, paragraphs, "- " bullets, "> " notes, and
// **bold**. The first paragraph is the takeaway and gets lede styling.
// Citations like [12] become buttons opening the source popover. Kept
// dependency-free on purpose.
function AnswerMarkdown({
  text,
  sourceNs,
  activeN,
  onCite,
}: {
  text: string;
  sourceNs: Map<number, AskSource>;
  activeN: number | null;
  onCite: (n: number, el: HTMLElement) => void;
}) {
  // split into homogeneous segments: a heading, a run of bullets, a run of
  // "> " note lines, or a run of plain lines each become their own segment,
  // even when the model omits blank lines between them (it often writes a
  // bold finding with the bullets directly underneath in the same block)
  const lineKind = (l: string) =>
    /^#{1,4}\s/.test(l) ? "heading"
    : l.startsWith("- ") ? "bullet"
    : l.startsWith("> ") ? "note"
    : "plain";
  const segments = text
    .split(/\n{2,}/)
    .flatMap((block) => {
      const out: string[] = [];
      let current: string[] = [];
      let kind: string | null = null;
      for (const raw of block.split("\n")) {
        const line = raw.trim();
        if (!line) continue;
        const k = lineKind(line);
        if (k === "heading" || k !== kind) {
          if (current.length) out.push(current.join("\n"));
          current = [];
          kind = k;
        }
        current.push(line);
        if (k === "heading") {
          out.push(current.join("\n"));
          current = [];
          kind = null;
        }
      }
      if (current.length) out.push(current.join("\n"));
      return out;
    })
    .filter((s) => s.trim());

  const inline = (t: string) => (
    <InlineMd text={t} sourceNs={sourceNs} activeN={activeN} onCite={onCite} />
  );

  let sawParagraph = false;
  return (
    <div>
      {segments.map((block, i) => {
        const heading = block.match(/^(#{1,4})\s+(.*)$/);
        if (heading) return <h4 key={i}>{inline(heading[2])}</h4>;
        if (block.trim() === "---") return <hr key={i} />;
        const lines = block.split("\n");
        if (lines[0].startsWith("> ")) {
          return (
            <blockquote key={i} className="ask-note">
              {inline(lines.map((l) => l.replace(/^> /, "")).join(" "))}
            </blockquote>
          );
        }
        if (lines[0].startsWith("- ")) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^- /, ""))}</li>
              ))}
            </ul>
          );
        }
        const isLede = !sawParagraph;
        sawParagraph = true;
        return (
          <p key={i} className={isLede ? "ask-lede" : undefined}>
            {inline(block)}
          </p>
        );
      })}
    </div>
  );
}

function InlineMd({
  text,
  sourceNs,
  activeN,
  onCite,
}: {
  text: string;
  sourceNs: Map<number, AskSource>;
  activeN: number | null;
  onCite: (n: number, el: HTMLElement) => void;
}) {
  // split on **bold** and [n] citations, keeping the delimiters
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d{1,4}\])/g);
  return (
    <>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        const cite = part.match(/^\[(\d{1,4})\]$/);
        if (cite) {
          const n = Number(cite[1]);
          if (!sourceNs.has(n)) return <sup key={i}>{part}</sup>;
          return (
            <button
              key={i}
              className={`ask-cite${activeN === n ? " ask-cite-active" : ""}`}
              title="Show the cited response"
              onClick={(e) => onCite(n, e.currentTarget)}
            >
              {n}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
