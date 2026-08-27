import { useEffect, useMemo, useRef, useState } from "react";
import { askAnswer, askDemographics, askQuestions, askRoute } from "./api";
import Pipeline from "./Pipeline";
import type {
  AskAnswerResponse,
  AskDemographic,
  AskParentGroup,
  AskQuestionOut,
  AskRouteResponse,
  AskSelectedCandidate,
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
  // the dataset was ingested but never processed — a 409 from /ask/route. Not
  // an error to dump on the analyst: it is a missing step, with the button
  // that performs it.
  | { name: "unprocessed"; question: string; detail: string }
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

// Day/night classification of verbatim time mentions. Like the event filter,
// deliberately one-directional per value: a response naming no time of day
// says nothing about when its experience happened, so there is no such option.
const TIME_CHOICES = [
  { value: "", label: "Any time", help: "no filter on time of day" },
  {
    value: "night",
    label: "Nighttime mentions only",
    help: 'responses explicitly saying "at night", "after dark", …',
  },
  {
    value: "day",
    label: "Daytime mentions only",
    help: 'responses explicitly saying "during the day", "morning", …',
  },
];

const RELEVANCE_COLOR: Record<string, string> = {
  high: "#166534",
  medium: "#92400e",
  low: "#6b7280",
};

// Words that carry no topic meaning, so two category names differing only by
// these are the same topic. Kept deliberately short: over-eager stripping would
// fuse categories that genuinely differ, and a wrong grouping is worse than an
// ungrouped list.
const TOPIC_STOPWORDS = new Set([
  "a", "an", "and", "at", "by", "for", "general", "in", "of", "on", "or",
  "other", "the", "to", "with",
  "concern", "concerns", "issue", "issues",
]);

function topicTokens(name: string): Set<string> {
  return new Set(
    name
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w && !TOPIC_STOPWORDS.has(w)),
  );
}

function tokenOverlap(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let shared = 0;
  for (const w of a) if (b.has(w)) shared += 1;
  return shared / (a.size + b.size - shared);
}

/** How alike two category names must be to be shown as one topic. Measured
 *  against dataset 2's 297 categories: 0.6 groups 15 topics (31 categories) and
 *  every one is a real restatement — "Police Staffing, Funding, Presence, and
 *  Response Times" with "Police Staffing, Presence, and Response Times". 0.5
 *  starts fusing "Park and Public Space Cleanliness" with "Park and Public
 *  Space Safety", which are different things. Requiring an exact word-set match
 *  catches only 3. Grouping is purely presentational — it never merges label
 *  ids or counts, and every variant keeps its own checkbox — but a wrong
 *  grouping still misleads, so it errs toward leaving names apart. */
const TOPIC_SIMILARITY = 0.6;

export default function Ask({
  datasetId,
  onError,
}: {
  datasetId: number;
  onError: (msg: string) => void;
}) {
  const [phase, setPhase] = useState<Phase>({ name: "question" });
  // the dataset's survey questions, for the scope selector; null while loading
  const [questions, setQuestions] = useState<AskQuestionOut[] | null>(null);
  // question_scope is a LIST end to end (schemas.py, ask_service.propose), so
  // multi-select needed no backend change — the UI was the only thing
  // restricting it to one. [] = all questions.
  const [scope, setScope] = useState<string[]>([]);
  // The dataset's demographic fields (null while loading, [] when it has
  // none) and the analyst's respondent filter: field -> ticked values.
  // Same posture as question scope — picked here, applied deterministically
  // server-side, never proposed by the router.
  const [demographics, setDemographics] = useState<AskDemographic[] | null>(
    null,
  );
  const [demoFilter, setDemoFilter] = useState<Record<string, string[]>>({});
  // respondents matching the whole current filter; null when nothing ticked
  const [nMatching, setNMatching] = useState<number | null>(null);
  // guards the faceted refetch: rapid ticking can land responses out of
  // order, and stale counts would contradict the checkboxes on screen
  const demoSeq = useRef(0);
  // Debug mode surfaces the category-review step between routing and the
  // answer. Off by default: the router's proposal is accepted as-is and the
  // answer appears in one step ("Adjust and re-answer" on the answer screen
  // still reaches the review). Persisted so the choice survives reloads.
  const [debug, setDebug] = useState(
    () => localStorage.getItem("ask_debug") === "1",
  );

  function toggleDebug() {
    setDebug((v) => {
      localStorage.setItem("ask_debug", v ? "0" : "1");
      return !v;
    });
  }

  useEffect(() => {
    let cancelled = false;
    setDemoFilter({});
    askQuestions(datasetId)
      .then((qs) => {
        if (!cancelled) setQuestions(qs);
      })
      .catch(() => {
        // 409 (unprocessed) or transient failure — the selector just stays
        // hidden; asking still works and routes to the unprocessed screen
        if (!cancelled) setQuestions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // Faceted dropdown counts: every tick refetches, so each field's numbers
  // reflect the OTHER fields' current selection (its own list stays
  // unrestricted — an OR selection must be extendable). Runs on mount too,
  // fetching the unfiltered counts.
  useEffect(() => {
    const seq = ++demoSeq.current;
    const active = Object.fromEntries(
      Object.entries(demoFilter).filter(([, vals]) => vals.length > 0),
    );
    askDemographics(datasetId, active)
      .then((r) => {
        if (demoSeq.current !== seq) return; // a newer tick superseded this
        setDemographics(r.fields);
        setNMatching(r.n_matching_respondents);
      })
      .catch(() => {
        // 409 (unprocessed) or transient failure — first load degrades to
        // the disabled control; later failures keep the previous counts
        if (demoSeq.current === seq) setDemographics((d) => d ?? []);
      });
  }, [datasetId, demoFilter]);

  async function handleAsk(question: string) {
    setPhase({ name: "routing", question });
    // fields with nothing ticked are no filter at all
    const activeDemo = Object.fromEntries(
      Object.entries(demoFilter).filter(([, vals]) => vals.length > 0),
    );
    try {
      const proposal = await askRoute(datasetId, question, scope, activeDemo);
      if (!debug && proposal.answerable) {
        // accept the router's proposal as-is and answer in one step; the
        // unanswerable screen still shows (there is nothing to auto-accept)
        void handleConfirm(
          question,
          proposal,
          proposal.candidates.map((c) => ({
            label_id: c.label_id,
            relevance: c.relevance,
            rationale: c.rationale,
          })),
          new Set(proposal.lexicon_concepts),
          new Set(proposal.location_filter),
          proposal.group_by === "location",
          proposal.actionability_filter,
          proposal.event_filter === "reported",
          proposal.time_filter,
        );
        return;
      }
      setPhase({ name: "review", question, proposal });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // 409 means "the pipeline hasn't run for this dataset" — a state the
      // analyst can fix in place, so route to the runner instead of an error
      if (msg.startsWith("409")) {
        setPhase({ name: "unprocessed", question, detail: msg });
        return;
      }
      onError(msg);
      setPhase({ name: "question" });
    }
  }

  async function handleConfirm(
    question: string,
    proposal: AskRouteResponse,
    // built from the whole taxonomy, not just the proposal — the review
    // screen can now add categories the router never suggested
    selected: AskSelectedCandidate[],
    concepts: Set<string>,
    places: Set<string>,
    groupByLocation: boolean,
    actionability: string,
    eventsOnly: boolean,
    timeOfDay: string,
  ) {
    setPhase({ name: "answering", question, proposal });
    try {
      const result = await askAnswer(datasetId, {
        question,
        route: proposal.route,
        reason: proposal.reason,
        selected,
        lexicon_concepts: [...concepts],
        group_by: groupByLocation ? "location" : "category",
        location_filter: [...places],
        actionability_filter: actionability,
        event_filter: eventsOnly ? "reported" : "",
        time_filter: timeOfDay,
        question_scope: proposal.question_scope,
        // the validated echo from the proposal, not local state — the answer
        // must apply exactly the filter the routing step was asked under
        demographic_filter: proposal.demographic_filter ?? {},
        proposed_label_ids: proposal.candidates.map((c) => c.label_id),
        aggregate_target: proposal.aggregate_target ?? "",
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
          questions={questions ?? []}
          scope={scope}
          onScope={setScope}
          demographics={demographics ?? []}
          demoFilter={demoFilter}
          onDemoFilter={setDemoFilter}
          nMatching={nMatching}
          debug={debug}
          onToggleDebug={toggleDebug}
        />
      )}

      {phase.name === "unprocessed" && (
        <div>
          <p>
            <strong>“{phase.question}”</strong>
          </p>
          <div className="ask-unprocessed">
            <strong>This dataset hasn't been processed yet.</strong>
            <p>
              It has been ingested, but there is no taxonomy and nothing is
              labeled, so there is no coded data to answer from. Run the
              pipeline below, then ask again.
            </p>
          </div>
          <Pipeline
            datasetId={datasetId}
            onError={onError}
            onProcessed={() => handleAsk(phase.question)}
          />
          <div style={{ marginTop: "1rem" }}>
            <button onClick={() => handleAsk(phase.question)}>
              Try this question again
            </button>{" "}
            <button onClick={() => setPhase({ name: "question" })}>
              Ask a different question
            </button>
          </div>
        </div>
      )}

      {phase.name === "answering" && !debug && (
        <div className="ask-progress">
          <p className="ask-eyebrow">Answering</p>
          <h2 className="ask-progress-q">“{phase.question}”</h2>
          <p className="ask-progress-note">
            {phase.proposal.candidates.length} categor
            {phase.proposal.candidates.length === 1 ? "y" : "ies"} matched ·
            computing counts and writing the answer…
          </p>
          <div className="ask-progress-bar" />
        </div>
      )}

      {(phase.name === "review" || (phase.name === "answering" && debug)) && (
        <ReviewPanel
          question={phase.question}
          proposal={phase.proposal}
          busy={phase.name === "answering"}
          onConfirm={(
            selected,
            concepts,
            places,
            groupByLocation,
            actionability,
            eventsOnly,
            timeOfDay,
          ) =>
            handleConfirm(
              phase.question,
              phase.proposal,
              selected,
              concepts,
              places,
              groupByLocation,
              actionability,
              eventsOnly,
              timeOfDay,
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
          onRefine={() =>
            setPhase({
              name: "review",
              question: phase.question,
              proposal: phase.proposal,
            })
          }
        />
      )}
    </section>
  );
}

const EXAMPLE_QUESTIONS = [
  "When residents mention affordability, what specific costs are they referring to?",
  "What makes residents feel unsafe at night?",
  "What concrete changes do residents propose for downtown?",
];

function QuestionForm({
  busy,
  onAsk,
  questions,
  scope,
  onScope,
  demographics,
  demoFilter,
  onDemoFilter,
  nMatching,
  debug,
  onToggleDebug,
}: {
  busy: boolean;
  onAsk: (q: string) => void;
  questions: AskQuestionOut[];
  scope: string[];
  onScope: (s: string[]) => void;
  demographics: AskDemographic[];
  demoFilter: Record<string, string[]>;
  onDemoFilter: (f: Record<string, string[]>) => void;
  nMatching: number | null;
  debug: boolean;
  onToggleDebug: () => void;
}) {
  const [question, setQuestion] = useState("");
  const canAsk = !busy && question.trim().length > 0;

  return (
    <div className="ask-home">
      <div className="ask-home-card">
        <textarea
          className="ask-home-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && canAsk) {
              e.preventDefault();
              onAsk(question);
            }
          }}
          rows={3}
          placeholder="e.g. what are the most common cleanliness complaints?"
          disabled={busy}
          autoFocus
        />
        <div className="ask-home-row">
          <div className="ask-home-filters">
            {questions.length > 1 && (
              <CheckboxDropdown
                label="Answer from"
                allLabel="all survey questions"
                disabled={busy}
                selected={scope}
                onChange={onScope}
                options={questions.map((q) => ({
                  value: q.question_id,
                  label: `“${q.question_text}”`,
                  meta: q.n_responses.toLocaleString(),
                }))}
              />
            )}
            {/* Respondent filters — one dropdown per demographic field the
                dataset carries (marked at ingest). Values within a field are
                OR, fields are AND; the server validates and applies the
                filter deterministically. A dataset without demographics keeps
                the disabled control so the seam stays visible. */}
            {demographics.length === 0 ? (
              <CheckboxDropdown
                label="Respondents"
                allLabel="everyone"
                options={[]}
                selected={[]}
                onChange={() => {}}
                disabled
                disabledNote="no demographic columns in this dataset"
              />
            ) : (
              demographics.map((d) => (
                <CheckboxDropdown
                  key={d.field}
                  label={d.field}
                  allLabel="all respondents"
                  disabled={busy}
                  selected={demoFilter[d.field] ?? []}
                  onChange={(vals) =>
                    onDemoFilter({ ...demoFilter, [d.field]: vals })
                  }
                  options={d.values.map((v) => ({
                    value: v.value,
                    label: v.value,
                    meta: v.n_respondents.toLocaleString(),
                  }))}
                />
              ))
            )}
          </div>
          <button
            className="ask-home-go"
            onClick={() => onAsk(question)}
            disabled={!canAsk}
          >
            {busy ? "Finding relevant categories…" : "Ask"}
          </button>
        </div>
        {nMatching != null && (
          <p
            className="ask-fs-hint"
            style={{ margin: "0.5rem 0 0" }}
            title="Respondents matching every ticked value (values within a field are either/or, fields combine). Respondents with no recorded value for a ticked field are missing data and never match."
          >
            {nMatching.toLocaleString()} respondent
            {nMatching === 1 ? "" : "s"} match
            {nMatching === 1 ? "es" : ""} the current filter
          </p>
        )}
      </div>
      <div className="ask-home-examples">
        <span className="ask-home-try">Try:</span>
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            className="ask-home-example"
            onClick={() => setQuestion(q)}
            disabled={busy}
          >
            {q}
          </button>
        ))}
      </div>
      <label className="ask-home-debug">
        <input type="checkbox" checked={debug} onChange={onToggleDebug} />{" "}
        Review the matched categories before answering (debug) — otherwise the
        router's selection is used as-is, and you can still adjust it from the
        answer screen
      </label>
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
    selected: AskSelectedCandidate[],
    concepts: Set<string>,
    places: Set<string>,
    groupByLocation: boolean,
    actionability: string,
    eventsOnly: boolean,
    timeOfDay: string,
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
  const [timeOfDay, setTimeOfDay] = useState(proposal.time_filter);

  // The full taxonomy as the server ordered it (parents holding a proposed
  // category first). If a server predating available_categories answers, fall
  // back to synthesizing groups from the proposal — a blank category list
  // would be a worse failure than a shorter one.
  const groups: AskParentGroup[] = useMemo(() => {
    if (proposal.available_categories.length > 0) {
      return proposal.available_categories;
    }
    const byKey = new Map<string, AskParentGroup>();
    for (const c of proposal.candidates) {
      const parent = c.parent_name ?? "Ungrouped";
      const key = `${c.question_id}::${parent}`;
      const g =
        byKey.get(key) ??
        {
          question_id: c.question_id,
          question_text: c.question_text,
          parent_name: parent,
          count_unique_responses: 0, // unknown without the tree; not shown
          n_proposed: 0,
          children: [],
        };
      g.children.push({
        label_id: c.label_id,
        name: c.name,
        count: c.count,
        description: "",
        proposed: true,
        relevance: c.relevance,
        rationale: c.rationale,
      });
      g.n_proposed += 1;
      byKey.set(key, g);
    }
    return [...byKey.values()];
  }, [proposal.available_categories, proposal.candidates]);

  const hasTree = proposal.available_categories.length > 0;

  // question_id -> its parent groups, preserving the server's parent order
  /* The analyst's scope restricts what the ROUTER was allowed to propose, but
     the tree deliberately still shows everything — step 2 stays permissive so
     a deliberate out-of-scope addition isn't rejected. That only works if the
     screen is honest about it: in-scope questions come first, and anything
     outside the scope is labelled rather than silently mixed in. */
  const scopeSet = useMemo(
    () => new Set(proposal.question_scope),
    [proposal.question_scope],
  );

  const byQuestion = useMemo(() => {
    const out = new Map<string, AskParentGroup[]>();
    for (const g of groups) {
      const list = out.get(g.question_id) ?? [];
      list.push(g);
      out.set(g.question_id, list);
    }
    if (scopeSet.size === 0) return out;
    return new Map(
      [...out.entries()].sort(
        (a, b) => Number(!scopeSet.has(a[0])) - Number(!scopeSet.has(b[0])),
      ),
    );
  }, [groups, scopeSet]);

  // label_id -> the router's relevance/rationale, so a ticked category carries
  // them into the manifest and an added one is honestly blank
  const childById = useMemo(() => {
    const m = new Map<string, { relevance: string; rationale: string }>();
    for (const g of groups) {
      for (const c of g.children) {
        m.set(c.label_id, { relevance: c.relevance, rationale: c.rationale });
      }
    }
    return m;
  }, [groups]);

  const nCategories = childById.size;

  // Search. 39 parent dropdowns over 297 categories is unusable by scrolling:
  // an analyst thinks "graffiti", not "question 7 > cleanliness and
  // infrastructure". Matching happens over the child name, its description and
  // its parent's name, across every question at once.
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();

  // A flat result list, deliberately not the tree: rendering matches inside
  // <details> would mean forcing them open, and a computed `open` prop is the
  // one thing this screen must not do (see the openParents note above). Flat
  // also means the matches are visible without a click, which is the point.
  const matches = useMemo(() => {
    if (!q) return [];
    const out = [];
    for (const g of groups) {
      const parentHit = g.parent_name.toLowerCase().includes(q);
      for (const c of g.children) {
        const nameHit = c.name.toLowerCase().includes(q);
        if (nameHit || parentHit || (c.description ?? "").toLowerCase().includes(q)) {
          out.push({
            ...c,
            question_id: g.question_id,
            parent_name: g.parent_name,
            nameHit,
          });
        }
      }
    }
    // Name matches lead. Searching "graffiti" otherwise opens with "Trash,
    // Litter, and Street Cleanliness" — a correct hit on its description, but
    // it reads as a bad match. Description hits still follow, since dropping
    // them would silently narrow recall.
    return out.sort(
      (a, b) =>
        Number(b.nameHit) - Number(a.nameHit) ||
        Number(b.proposed) - Number(a.proposed) ||
        b.count - a.count,
    );
  }, [groups, q]);

  // Each question induced its own taxonomy, so the same topic exists once per
  // question ("Graffiti and vandalism" q7 / "Graffiti and Vandalism" q9 /
  // "Vandalism and Graffiti" q6). Listed as siblings they read as duplicated
  // data. They are not: they are three different survey questions, and a
  // response only ever belongs to one of them. Grouping them under one topic
  // row says that, where three near-identical rows imply the opposite.
  const matchTopics = useMemo(() => {
    // `matches` is already ranked, so the anchor of each cluster is its most
    // prominent variant and gets to name the topic — inventing a name would
    // put a category on screen that no taxonomy contains.
    const toks = matches.map((m) => topicTokens(m.name));
    const taken = matches.map(() => false);
    const out = [];
    for (let i = 0; i < matches.length; i += 1) {
      if (taken[i]) continue;
      taken[i] = true;
      const items = [matches[i]];
      for (let j = i + 1; j < matches.length; j += 1) {
        if (!taken[j] && tokenOverlap(toks[i], toks[j]) >= TOPIC_SIMILARITY) {
          taken[j] = true;
          items.push(matches[j]);
        }
      }
      out.push({
        key: items.map((m) => m.label_id).join("+"),
        name: items[0].name,
        items,
        // a sum over *different questions* — distinct response rows, so a real
        // total. NOT distinct respondents: one person answers several
        // questions, which is why this is never worded as people.
        total: items.reduce((s, m) => s + m.count, 0),
        proposed: items.some((m) => m.proposed),
        nQuestions: new Set(items.map((m) => m.question_id)).size,
      });
    }
    return out;
  }, [matches]);

  const nGrouped = matches.length - matchTopics.length;

  // Nothing may be both selected and invisible. A search that hides ticked
  // categories has to say so, or the analyst submits a selection they can't see.
  const hiddenSelected = useMemo(() => {
    if (!q) return 0;
    const shown = new Set(matches.map((m) => m.label_id));
    return [...selected].filter((id) => !shown.has(id)).length;
  }, [q, matches, selected]);

  function setMany(ids: string[], on: boolean) {
    setSelected((s) => {
      const next = new Set(s);
      for (const id of ids) {
        if (on) next.add(id);
        else next.delete(id);
      }
      return next;
    });
  }

  // <details> open state has to live in React: passing `open` as a plain prop
  // would snap a parent shut on the next re-render (every checkbox tick).
  // Parents holding a proposal start open — the proposal stays the thing the
  // analyst reads first; everything else is one click away.
  const [openParents, setOpenParents] = useState<Set<string>>(
    () =>
      new Set(
        groups
          .filter((g) => g.n_proposed > 0)
          .map((g) => `${g.question_id}::${g.parent_name}`),
      ),
  );
  // Keyword concepts and the place list stay collapsed until asked for; what
  // is selected is named in the summary line, so nothing is both active and
  // invisible.
  const [conceptsOpen, setConceptsOpen] = useState(false);
  const [placesOpen, setPlacesOpen] = useState(false);

  function toggleParent(key: string, open: boolean) {
    setOpenParents((s) => {
      const next = new Set(s);
      if (open) next.add(key);
      else next.delete(key);
      return next;
    });
  }

  function submit() {
    onConfirm(
      [...selected].map((id) => ({
        label_id: id,
        relevance: childById.get(id)?.relevance || "medium",
        rationale: childById.get(id)?.rationale || "",
      })),
      concepts,
      places,
      groupByLocation,
      actionability,
      eventsOnly,
      timeOfDay,
    );
  }

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
    <div className="ask-review">
      <p>
        <strong>“{question}”</strong>
      </p>
      <p>
        Answer strategy: <strong>{proposal.route}</strong> — {proposal.reason}
      </p>
      {Object.keys(proposal.demographic_filter ?? {}).length > 0 && (
        <p className="ask-fs-hint">
          Respondent filter (set on the ask form):{" "}
          <strong>
            {Object.entries(proposal.demographic_filter)
              .map(([f, vals]) => `${f} = ${vals.join(" or ")}`)
              .join("; ")}
          </strong>
          . Counts and quotes will cover only matching respondents, and the
          answer will state that denominator.
        </p>
      )}
      {proposal.warnings.map((w) => (
        <p key={w} style={{ color: "#b45309", fontSize: "0.9em" }}>
          ⚠ {w}
        </p>
      ))}

      <fieldset className="ask-fs">
        <legend>
          Categories the answer will search —{" "}
          <strong>
            {selected.size} of {nCategories}
          </strong>{" "}
          selected
        </legend>
        <p className="ask-fs-hint">
          The {proposal.candidates.length} the router proposed are ticked, and
          their parent categories are open. Every other category is here too —
          search for one, open a parent to add one, or untick anything that
          doesn't belong.
        </p>

        <div className="ask-search">
          <input
            type="search"
            className="ask-search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Search all ${nCategories} categories — e.g. graffiti, parking, rent`}
            aria-label="Search categories"
          />
          {q && (
            <button
              type="button"
              className="ask-search-clear"
              onClick={() => setQuery("")}
            >
              Clear
            </button>
          )}
        </div>

        {q ? (
          <div className="ask-search-results">
            <p className="ask-fs-hint">
              {matches.length === 0
                ? `No category matches “${query}”.`
                : `${matchTopics.length} topic${matchTopics.length === 1 ? "" : "s"} match “${query}”.`}
              {nGrouped > 0 &&
                " Each survey question was coded separately and words the same topic differently, so those are shown as one row — different questions, not repeated responses."}
              {hiddenSelected > 0 &&
                ` ${hiddenSelected} selected categor${hiddenSelected === 1 ? "y is" : "ies are"} hidden by this search — still included in the answer.`}
            </p>
            {matchTopics.map((t) => {
              // one question only: nothing to consolidate, render it plainly
              if (t.items.length === 1) {
                const m = t.items[0];
                return (
                  <label
                    key={m.label_id}
                    className={`ask-child${m.proposed ? " ask-child-proposed" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(m.label_id)}
                      onChange={() => toggle(m.label_id)}
                      disabled={busy}
                    />
                    <span className="ask-child-body">
                      <span className="ask-child-head">
                        <strong>{m.name}</strong>
                        <span className="ask-child-n">n={m.count}</span>
                        {m.proposed && (
                          <span
                            className="ask-rel"
                            style={{
                              color: RELEVANCE_COLOR[m.relevance] ?? "#6b7280",
                            }}
                          >
                            {m.relevance}
                          </span>
                        )}
                      </span>
                      <span className="ask-child-crumb">
                        question {m.question_id} · {m.parent_name}
                      </span>
                      {(m.proposed ? m.rationale : m.description) && (
                        <span className="ask-child-sub">
                          {m.proposed ? m.rationale : m.description}
                        </span>
                      )}
                    </span>
                  </label>
                );
              }
              const ids = t.items.map((i) => i.label_id);
              const nSel = ids.filter((id) => selected.has(id)).length;
              return (
                <div
                  key={t.key}
                  className={`ask-topic${t.proposed ? " ask-child-proposed" : ""}`}
                >
                  <label className="ask-topic-head">
                    <input
                      type="checkbox"
                      checked={nSel === ids.length}
                      ref={(el) => {
                        // partial selection reads as neither on nor off
                        if (el) el.indeterminate = nSel > 0 && nSel < ids.length;
                      }}
                      onChange={() => setMany(ids, nSel < ids.length)}
                      disabled={busy}
                    />
                    <span className="ask-child-head">
                      <strong>{t.name}</strong>
                      <span className="ask-child-n">n={t.total}</span>
                      <span className="ask-topic-note">
                        {t.nQuestions > 1
                          ? `worded ${t.items.length} ways across ${t.nQuestions} survey questions`
                          : `${t.items.length} near-identical categories in one question`}
                      </span>
                    </span>
                  </label>
                  <div className="ask-topic-items">
                    {t.items.map((m) => (
                      <label key={m.label_id} className="ask-child">
                        <input
                          type="checkbox"
                          checked={selected.has(m.label_id)}
                          onChange={() => toggle(m.label_id)}
                          disabled={busy}
                        />
                        <span className="ask-child-body">
                          <span className="ask-child-head">
                            <span className="ask-child-crumb">
                              question {m.question_id} · {m.parent_name}
                            </span>
                            <span className="ask-child-n">n={m.count}</span>
                            {m.proposed && (
                              <span
                                className="ask-rel"
                                style={{
                                  color:
                                    RELEVANCE_COLOR[m.relevance] ?? "#6b7280",
                                }}
                              >
                                {m.relevance}
                              </span>
                            )}
                          </span>
                          {m.name !== t.name && (
                            <span className="ask-child-sub">
                              worded here as “{m.name}”
                            </span>
                          )}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          [...byQuestion.entries()].map(([qid, parents]) => (
          <div key={qid} className="ask-qblock">
            <div className="ask-qhead">
              Survey question {qid}: “{parents[0].question_text}”
              {scopeSet.size > 0 && !scopeSet.has(qid) && (
                <span className="ask-qhead-outside">
                  outside your scope — nothing here was proposed, but ticking it
                  still adds it to the answer
                </span>
              )}
            </div>
            {parents.map((g) => {
              const key = `${qid}::${g.parent_name}`;
              const nSel = g.children.filter((c) =>
                selected.has(c.label_id),
              ).length;
              return (
                <details
                  key={key}
                  className="ask-parent"
                  open={openParents.has(key)}
                  onToggle={(e) =>
                    toggleParent(key, (e.currentTarget as HTMLDetailsElement).open)
                  }
                >
                  <summary>
                    <span className="ask-parent-name">{g.parent_name}</span>
                    <span className="ask-parent-meta">
                      {nSel > 0 && (
                        <span className="ask-sel-badge">{nSel} selected</span>
                      )}
                      <span>
                        {g.children.length} categor
                        {g.children.length === 1 ? "y" : "ies"}
                      </span>
                      {hasTree && (
                        <span
                          className="ask-parent-n"
                          title="Responses labelled with at least one category under this parent — a union, not the sum of the child counts"
                        >
                          n={g.count_unique_responses}
                        </span>
                      )}
                    </span>
                  </summary>
                  <div className="ask-children">
                    {g.children.map((c) => (
                      <label
                        key={c.label_id}
                        className={`ask-child${c.proposed ? " ask-child-proposed" : ""}`}
                      >
                        <input
                          type="checkbox"
                          checked={selected.has(c.label_id)}
                          onChange={() => toggle(c.label_id)}
                          disabled={busy}
                        />
                        <span className="ask-child-body">
                          <span className="ask-child-head">
                            <strong>{c.name}</strong>
                            <span className="ask-child-n">n={c.count}</span>
                            {c.proposed && (
                              <span
                                className="ask-rel"
                                style={{
                                  color:
                                    RELEVANCE_COLOR[c.relevance] ?? "#6b7280",
                                }}
                              >
                                {c.relevance}
                              </span>
                            )}
                          </span>
                          {(c.proposed ? c.rationale : c.description) && (
                            <span className="ask-child-sub">
                              {c.proposed ? c.rationale : c.description}
                            </span>
                          )}
                        </span>
                      </label>
                    ))}
                  </div>
                </details>
              );
            })}
          </div>
          ))
        )}
      </fieldset>

      {proposal.available_lexicon_concepts.length > 0 && (
        <fieldset className="ask-fs">
          <legend>Keyword concepts — optional</legend>
          <details
            className="ask-drop"
            open={conceptsOpen}
            onToggle={(e) =>
              setConceptsOpen((e.currentTarget as HTMLDetailsElement).open)
            }
          >
            <summary>
              {concepts.size > 0 ? (
                <>
                  <span className="ask-sel-badge">{concepts.size} selected</span>
                  <span className="ask-drop-names">
                    {[...concepts].join(", ")}
                  </span>
                </>
              ) : (
                <span className="ask-drop-names">
                  No keyword concepts — {proposal.available_lexicon_concepts.length}{" "}
                  available
                </span>
              )}
            </summary>
            <p className="ask-fs-hint">
              A deterministic exact-match sweep, separate from the categories:
              ticking one adds its mention count to the answer. Recall is a
              floor — paraphrase that avoids the terms is not matched.
            </p>
            <div className="ask-chipbox">
              {proposal.available_lexicon_concepts.map((name) => (
                <label key={name} className="ask-inline-check">
                  <input
                    type="checkbox"
                    checked={concepts.has(name)}
                    onChange={() => toggleConcept(name)}
                    disabled={busy}
                  />{" "}
                  {name}
                </label>
              ))}
            </div>
          </details>
        </fieldset>
      )}

      {proposal.available_locations.length > 0 && (
        <fieldset className="ask-fs">
          <legend>Places</legend>
          <div className="ask-toggle-row">
            <button
              type="button"
              className={`ask-toggle${groupByLocation ? " ask-toggle-on" : ""}`}
              aria-pressed={groupByLocation}
              onClick={() => setGroupByLocation((v) => !v)}
              disabled={busy}
            >
              {groupByLocation ? "✓ Sorting by place" : "Sort by place"}
            </button>
            <span className="ask-fs-hint">
              For “where…” questions — counts and quotes grouped per place.
              Only responses that volunteered a place are localizable, and the
              answer discloses that denominator.
            </span>
          </div>
          <details
            className="ask-drop"
            open={placesOpen}
            onToggle={(e) =>
              setPlacesOpen((e.currentTarget as HTMLDetailsElement).open)
            }
          >
            <summary>
              {places.size > 0 ? (
                <>
                  <span className="ask-sel-badge">
                    {places.size} place{places.size === 1 ? "" : "s"}
                  </span>
                  <span className="ask-drop-names">{[...places].join(", ")}</span>
                </>
              ) : (
                <span className="ask-drop-names">
                  No place filter — {proposal.available_locations.length} places
                  available
                </span>
              )}
            </summary>
            {places.size > 0 && (
              <p className="ask-fs-hint">
                Only responses mentioning a ticked place will be used as
                evidence.
              </p>
            )}
            {(
              [
                ["named", "Named places"],
                ["type", "Kinds of place"],
              ] as const
            ).map(([kind, label]) => {
              const items = proposal.available_locations.filter(
                (l) => l.kind === kind,
              );
              if (items.length === 0) return null;
              return (
                <div key={kind} className="ask-place-group">
                  <div className="ask-place-head">{label}</div>
                  <div className="ask-chipbox">
                    {items.map((l) => (
                      <label key={l.name} className="ask-inline-check">
                        <input
                          type="checkbox"
                          checked={places.has(l.name)}
                          onChange={() => togglePlace(l.name)}
                          disabled={busy}
                        />{" "}
                        {l.name}{" "}
                        <span className="ask-child-n">n={l.count}</span>
                      </label>
                    ))}
                  </div>
                </div>
              );
            })}
          </details>
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

      {((proposal.available_time.day ?? 0) > 0 ||
        (proposal.available_time.night ?? 0) > 0) && (
        <fieldset style={{ marginTop: "1rem" }}>
          <legend>
            Time of day — answer only from responses that name one
          </legend>
          {TIME_CHOICES.map((choice) => {
            const n = proposal.available_time[choice.value];
            if (choice.value && !n) return null;
            return (
              <label
                key={choice.value || "any"}
                style={{ display: "block", marginBottom: "0.35rem" }}
              >
                <input
                  type="radio"
                  name="timeOfDay"
                  checked={timeOfDay === choice.value}
                  onChange={() => setTimeOfDay(choice.value)}
                  disabled={busy}
                />{" "}
                <strong>{choice.label}</strong>
                {choice.value !== "" && n != null && (
                  <span style={{ color: "#555", fontFamily: "monospace" }}>
                    {" "}
                    n={n}
                  </span>
                )}
                <span style={{ color: "#555" }}> — {choice.help}</span>
              </label>
            );
          })}
          <p style={{ margin: "0.4rem 0 0", fontSize: "0.85em", color: "#555" }}>
            Classified from verbatim time mentions. Responses naming no time of
            day are excluded when a filter is on — naming none says nothing
            about when their experience happened, and the answer will say so.
          </p>
        </fieldset>
      )}

      {/* Respondent (demographic) filters are set on the ask form, like the
          question scope — the active filter is echoed at the top of this
          screen. To change it, go back and re-ask. */}

      <div style={{ marginTop: "1rem" }}>
        {/* a tally route needs no categories — selecting some only narrows
            the tally to their responses */}
        <button
          onClick={submit}
          disabled={
            busy ||
            (selected.size === 0 && proposal.route !== "aggregate_direct")
          }
        >
          {busy
            ? proposal.route === "aggregate_direct"
              ? "Computing tally…"
              : "Writing answer…"
            : proposal.route === "aggregate_direct"
              ? selected.size === 0
                ? "Compute tally over every response in scope"
                : `Compute tally over ${selected.size} categor${selected.size === 1 ? "y" : "ies"}`
              : `Answer from ${selected.size} categor${selected.size === 1 ? "y" : "ies"}`}
        </button>{" "}
        <button onClick={onBack} disabled={busy}>
          Back
        </button>
        {selected.size === 0 && proposal.route !== "aggregate_direct" && (
          <span style={{ marginLeft: "0.75rem", color: "#b45309", fontSize: "0.9em" }}>
            Select at least one category.
          </span>
        )}
        {proposal.route === "aggregate_direct" && (
          <span style={{ marginLeft: "0.75rem", color: "#666", fontSize: "0.9em" }}>
            Counted directly from the coded data — no AI writes this answer.
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
  aggregate_direct: "Direct tally — counted, not written by AI",
};

/* A <select> that holds checkboxes: pick several survey questions or several
   demographic values instead of one.
   Native <select multiple> is unusable for this — it needs ctrl-click, gives
   no count, and cannot show a per-option response total. Closes on outside
   click or Escape; the button reports the selection so the state is legible
   without opening it. */
function CheckboxDropdown({
  label,
  allLabel,
  options,
  selected,
  onChange,
  disabled,
  disabledNote,
}: {
  label: string;
  allLabel: string;
  options: { value: string; label: string; meta?: string }[];
  selected: string[];
  onChange: (next: string[]) => void;
  disabled?: boolean;
  disabledNote?: string;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const byValue = new Map(options.map((o) => [o.value, o]));
  const chosen = selected.filter((v) => byValue.has(v));
  const summary =
    chosen.length === 0
      ? allLabel
      : chosen.length === 1
        ? (byValue.get(chosen[0])!.label)
        : `${chosen.length} of ${options.length} selected`;

  return (
    <div className="ask-dd" ref={wrap}>
      <span className="ask-dd-label">{label}</span>
      <button
        type="button"
        className="ask-dd-btn"
        disabled={disabled}
        title={disabled ? disabledNote : summary}
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="ask-dd-summary">{disabled ? disabledNote : summary}</span>
        <span className="ask-dd-caret" aria-hidden="true">▾</span>
      </button>
      {open && !disabled && (
        <div className="ask-dd-menu" role="listbox">
          <button
            type="button"
            className="ask-dd-all"
            onClick={() => onChange([])}
          >
            {allLabel}
            {chosen.length === 0 && <span className="ask-dd-tick">✓</span>}
          </button>
          <div className="ask-dd-sep" />
          {options.map((o) => {
            const on = chosen.includes(o.value);
            return (
              <label key={o.value} className="ask-dd-opt">
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() =>
                    onChange(
                      on
                        ? chosen.filter((v) => v !== o.value)
                        : [...chosen, o.value],
                    )
                  }
                />
                <span className="ask-dd-opt-text">{o.label}</span>
                {o.meta && <span className="ask-dd-opt-meta">{o.meta}</span>}
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

const PLACES_SHOWN = 10;

// Sources per group before "show more". A filtered ask can cite 90+ responses
// (the night-safety ask cited 91 of 98 shown), and the panel rendered every
// one at full length — quotes stopped being truncated on 2026-08-13 — so the
// list read as a wall rather than as examples. Five is enough to see what a
// group sounds like; the rest are one click away and still all present.
const SOURCES_SHOWN = 5;

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
  if (result.time_filter === "night" || result.time_filter === "day") {
    parts.push(`explicitly mentioning ${result.time_filter}time`);
  }
  for (const [f, vals] of Object.entries(result.demographic_filter ?? {})) {
    parts.push(`from respondents with ${f} ${vals.join(" or ")}`);
  }
  return parts.join(" and ");
}

function AnswerView({
  question,
  proposal,
  result,
  onReset,
  onRefine,
}: {
  question: string;
  proposal: AskRouteResponse;
  result: AskAnswerResponse;
  onReset: () => void;
  onRefine: () => void;
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

  // which group holds each citation — a "jump to source" for a source hidden
  // behind "show more" has to open that group first, or it scrolls to nothing
  const groupTitleOf = useMemo(() => {
    const m = new Map<number, string>();
    for (const [title, items] of sourceGroups) {
      for (const s of items) m.set(s.n, title);
    }
    return m;
  }, [sourceGroups]);

  const articleRef = useRef<HTMLDivElement>(null);
  const sourcesRef = useRef<HTMLDetailsElement>(null);

  const [pop, setPop] = useState<{ n: number; top: number; left: number } | null>(null);
  const [flashN, setFlashN] = useState<number | null>(null);
  const [allPlaces, setAllPlaces] = useState(false);
  const [copied, setCopied] = useState(false);
  const [openGroups, setOpenGroups] = useState<ReadonlySet<string>>(new Set());
  // quote cards are line-clamped so one long response can't set a whole row's
  // height; clicking one opens it in place
  const [openQuotes, setOpenQuotes] = useState<ReadonlySet<number>>(new Set());

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

  function toggleGroup(title: string) {
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (!next.delete(title)) next.add(title);
      return next;
    });
  }

  function jumpToSource(n: number) {
    setPop(null);
    if (sourcesRef.current) sourcesRef.current.open = true;
    setFlashN(null);
    // the target may be past its group's "show more" cut — open that group,
    // then wait for the render to commit before scrolling
    const title = groupTitleOf.get(n);
    if (title) setOpenGroups((prev) => new Set(prev).add(title));
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        document
          .getElementById(`ask-src-${n}`)
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
        setFlashN(n);
      }),
    );
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

  // Per-section breakdown charts: match a model-written "### " heading back
  // to the category (or sub-theme) it narrates, by token overlap, so the
  // heading can offer the FULL sub-theme distribution as a chart — the prose
  // deliberately covers only the top sub-themes.
  const chartFor = useMemo(() => {
    const tok = (s: string) =>
      new Set(
        (s.toLowerCase().match(/[a-z]+/g) ?? []).filter(
          (w) => !CHART_STOPWORDS.has(w),
        ),
      );
    const entries: { tokens: Set<string>; data: SectionChartData }[] = [];
    for (const [lid, b] of Object.entries(result.sub_breakdowns ?? {})) {
      if (!b.sub_counts.length) continue;
      const catName = nameOf.get(lid) ?? lid;
      const rows = [
        ...b.sub_counts.map((sc) => ({
          name: sc.name,
          count: sc.count,
          id: sc.sub_label_id,
        })),
        ...(b.generic > 0
          ? [{ name: "No specific sub-theme named", count: b.generic, generic: true }]
          : []),
      ];
      const base = { category: catName, rows, coded: b.coded };
      entries.push({ tokens: tok(catName), data: base });
      for (const sc of b.sub_counts) {
        entries.push({
          tokens: tok(sc.name),
          data: { ...base, highlight: sc.sub_label_id },
        });
      }
    }
    return (heading: string): SectionChartData | null => {
      const h = tok(heading);
      if (!h.size) return null;
      let best: SectionChartData | null = null;
      let bestScore = 0.34; // below this, the heading isn't about that unit
      for (const e of entries) {
        if (!e.tokens.size) continue;
        const inter = [...h].filter((w) => e.tokens.has(w)).length;
        const score = inter / (h.size + e.tokens.size - inter);
        if (score > bestScore) {
          bestScore = score;
          best = e.data;
        }
      }
      return best;
    };
  }, [result, nameOf]);

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
        {(result.time_filter === "night" || result.time_filter === "day") && (
          <span className="ask-chip ask-chip-route">
            Only {result.time_filter}time mentions
          </span>
        )}
        {Object.keys(result.demographic_filter ?? {}).length > 0 && (
          <span className="ask-chip ask-chip-route">
            Respondents:{" "}
            {Object.entries(result.demographic_filter)
              .map(([f, vals]) => `${f} ${vals.join(" or ")}`)
              .join(" · ")}
          </span>
        )}
        {result.demographic_thin && result.demographic_denominator && (
          <span className="ask-chip ask-chip-warn">
            ⚠ Thin group — only {result.demographic_denominator.matching}{" "}
            matching response
            {result.demographic_denominator.matching === 1 ? "" : "s"}
          </span>
        )}
        {proposal.question_scope.length > 0 && (
          <span className="ask-chip ask-chip-route">
            Scoped to question {proposal.question_scope.join(", ")}
          </span>
        )}
        {result.deselected.length > 0 && (
          <span className="ask-chip">
            {result.deselected.length} proposed categor
            {result.deselected.length === 1 ? "y" : "ies"} deselected
          </span>
        )}
        {stats.scope_coverage != null && (
          <span
            className={`ask-chip${stats.scope_coverage < 0.6 ? " ask-chip-warn" : ""}`}
            title={`The searched categories cover ${stats.unique_responses} of the ${stats.scope_total} coded responses in scope`}
          >
            Covers {Math.round(stats.scope_coverage * 100)}% of responses in
            scope
          </span>
        )}
        {stats.small_base && (
          <span className="ask-chip ask-chip-warn">
            ⚠ Small base — only {stats.unique_responses} responses
          </span>
        )}
        {result.cached && (
          <span
            className="ask-chip"
            title="This exact question was already answered against the current data — showing the stored answer. It refreshes automatically when the data changes."
          >
            ↺ Saved answer — identical every time until the data changes
          </span>
        )}
        {(result.verification?.violations.length ?? 0) > 0 && (
          <span
            className="ask-chip ask-chip-warn"
            title={result.verification!.violations
              .map((v) => `${v.value}: ${v.detail}`)
              .join("\n")}
          >
            ⚠ {result.verification!.violations.length} statement
            {result.verification!.violations.length === 1 ? "" : "s"} could not
            be verified against the data
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
          <dd>
            {stats.unique_responses}
            {stats.scope_coverage != null && (
              <span className="ask-sub">
                {" "}
                of {stats.scope_total} in scope
              </span>
            )}
          </dd>
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
        {result.time_denominator && (
          <div>
            <dt>Mentioned {result.time_filter}time</dt>
            <dd>
              {result.time_denominator.matching}{" "}
              <span className="ask-sub">
                of {result.time_denominator.in_scope}
              </span>
            </dd>
          </div>
        )}
        {result.demographic_denominator && (
          <div>
            <dt>Matched the respondent filter</dt>
            <dd>
              {result.demographic_denominator.matching}{" "}
              <span className="ask-sub">
                of {result.demographic_denominator.in_scope}
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
      {result.demographic_thin && result.demographic_denominator && (
        <p className="ask-caveat">
          Thin group: only {result.demographic_denominator.matching} of{" "}
          {result.demographic_denominator.in_scope} in-scope responses match
          the respondent filter. The answer is those few voices, not the
          group — and respondents with no recorded value for a filtered field
          are missing data, not part of either side.
        </p>
      )}
      {stats.small_base && (
        <p className="ask-caveat">
          Small base: this answer rests on only {stats.unique_responses}{" "}
          responses{stats.scope_total > 0 && (
            <> of the {stats.scope_total} coded responses in scope</>
          )}
          . Read it as a description of that small group, not of respondents
          overall.
        </p>
      )}
      <details className="ask-method-full">
        <summary>How this answer was computed</summary>
        <p>{result.process_note}</p>
      </details>

      <div className="ask-main">
        <div className="ask-article" ref={articleRef}>
          <AnswerMarkdown
            text={result.answer_markdown}
            sourceNs={sourceByN}
            activeN={pop?.n ?? null}
            onCite={handleCite}
            chartFor={chartFor}
          />

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

      </div>

      {/* Sources sit OUTSIDE the reading column: 91 cited responses stacked
          one-per-row down a 600px column ran for thousands of pixels with the
          rest of the page empty beside it. As cards they flow left-to-right
          and wrap, so the width does the work instead of the scrollbar. */}
      <details className="ask-sources" ref={sourcesRef}>
        <summary>
          Sources{" "}
          <span className="ask-cnt">
            — {result.sources.length} cited response
            {result.sources.length === 1 ? "" : "s"}, each traceable to its
            survey row
          </span>
        </summary>
        {sourceGroups.map(([title, items]) => {
          const expanded = openGroups.has(title);
          const shown = expanded ? items : items.slice(0, SOURCES_SHOWN);
          const hidden = items.length - shown.length;
          return (
            <div key={title} className="ask-src-group">
              <h6>
                {title} <span className="ask-src-n">{items.length}</span>
              </h6>
              <div className="ask-src-items">
                {shown.map((s) => (
                  <div
                    key={s.n}
                    id={`ask-src-${s.n}`}
                    className={
                      "ask-src-item"
                      + (flashN === s.n ? " ask-src-flash" : "")
                      + (openQuotes.has(s.n) ? " ask-src-open" : "")
                    }
                  >
                    <div className="ask-meta">
                      <span className="ask-n">[{s.n}]</span> {s.response_key}
                    </div>
                    <blockquote
                      onClick={() =>
                        setOpenQuotes((prev) => {
                          const next = new Set(prev);
                          if (!next.delete(s.n)) next.add(s.n);
                          return next;
                        })
                      }
                    >
                      “{s.text}”
                    </blockquote>
                  </div>
                ))}
              </div>
              {(hidden > 0 || expanded) && (
                <button
                  className="ask-more-link"
                  onClick={() => toggleGroup(title)}
                >
                  {expanded
                    ? "Show fewer"
                    : `Show ${hidden} more response${hidden === 1 ? "" : "s"}…`}
                </button>
              )}
            </div>
          );
        })}
      </details>

      {/* Evidence boxes. These used to be a 300px rail beside the answer,
          which put the prose in a ~66ch gutter and made the biggest panel
          (Categories searched, with a sub-theme breakdown per category) the
          narrowest thing on the page. The answer now runs full width in the
          middle and the supporting counts sit below it as boxes. */}
      <div className="ask-boxes">
          {(result.uncovered_categories?.length ?? 0) > 0 && (
            <div className="ask-panel">
              <h5>Not covered by this answer</h5>
              <p className="ask-hint">
                The searched categories cover{" "}
                {stats.scope_coverage != null
                  ? `${Math.round(stats.scope_coverage * 100)}%`
                  : "a minority"}{" "}
                of responses in scope. The largest categories left out:
              </p>
              {result.uncovered_categories.map((u) => (
                <div key={u.label_id} className="ask-bar-row">
                  <div className="ask-bar-label">
                    <span className="ask-nm">{u.name}</span>
                    <span className="ask-ct">{u.count}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

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
          <div className="ask-panel ask-panel-wide">
            <h5>Categories searched</h5>
            <p className="ask-hint">
              {filterPhrase(result)
                ? `Responses in each category ${filterPhrase(result)} — “of N” is the category's full size.`
                : "Real counts from the coded data, never estimated."}
            </p>
            <div className="ask-cat-cols">
              {countRows.map(([lid, n]) => {
                const breakdown = result.sub_breakdowns?.[lid];
                return (
                  <div key={lid} className="ask-cat-item">
                    <div
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
                    {breakdown && breakdown.sub_counts.length > 0 && (
                      /* the full-coverage composition the answer's sections are
                         built from — every member of the category was coded, so
                         these are counts, not a sample */
                      <div className="ask-subthemes">
                        {breakdown.sub_counts.map((sc) => (
                          <div key={sc.sub_label_id} className="ask-subtheme-row">
                            <span className="ask-subtheme-ct">{sc.count}</span>
                            <span className="ask-subtheme-nm">{sc.name}</span>
                          </div>
                        ))}
                        {breakdown.generic > 0 && (
                          <div className="ask-subtheme-row ask-subtheme-generic">
                            <span className="ask-subtheme-ct">
                              {breakdown.generic}
                            </span>
                            <span className="ask-subtheme-nm">
                              raise it only generically — no specific sub-theme
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {nSampled > 0 && (
              <div className="ask-sampling-note">
                {/* a note now means "these quotes are not all of them" from
                    any cause — sampling, or a response already quoted under
                    another place — so this no longer says "sampled" */}
                {nSampled}{" "}
                {byLocation ? "place" : "categor"}
                {nSampled === 1 ? (byLocation ? "" : "y") : byLocation ? "s" : "ies"}{" "}
                showed the model only some of their responses; counts always
                cover the full data.
              </div>
            )}
          </div>
      </div>

      <div className="ask-runbar">
        <span>
          Saved as run <code>{result.run_id}</code> under <code>data/answers/</code>
        </span>
        <span className="ask-actions ask-no-print">
          <button onClick={copyAnswer}>{copied ? "Copied ✓" : "Copy answer"}</button>
          <button onClick={onRefine}>Adjust categories &amp; re-answer</button>
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
// words too common in category/sub-theme names to signal a match
const CHART_STOPWORDS = new Set([
  "and", "the", "of", "to", "in", "for", "a", "on", "with", "general",
  "specific", "other", "more",
]);

interface SectionChartData {
  category: string;
  rows: { name: string; count: number; id?: string; generic?: boolean }[];
  coded: number;
  highlight?: string;
}

function SectionChart({ data }: { data: SectionChartData }) {
  const max = Math.max(...data.rows.map((r) => r.count), 1);
  return (
    <div className="ask-sec-chart">
      <div className="ask-sec-chart-head">
        Full breakdown of “{data.category}” — {data.coded.toLocaleString()}{" "}
        coded responses; one response can raise several sub-themes
      </div>
      {data.rows.map((r) => (
        <div
          key={r.id ?? r.name}
          className={
            "ask-chart-row" +
            (r.generic ? " ask-chart-generic" : "") +
            (data.highlight && r.id === data.highlight ? " ask-chart-hl" : "")
          }
        >
          <span className="ask-chart-ct">{r.count.toLocaleString()}</span>
          <div className="ask-chart-body">
            <span className="ask-chart-nm">{r.name}</span>
            <div className="ask-chart-track">
              <div
                className="ask-chart-fill"
                style={{ width: `${Math.max((r.count / max) * 100, 2)}%` }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function AnswerMarkdown({
  text,
  sourceNs,
  activeN,
  onCite,
  chartFor,
}: {
  text: string;
  sourceNs: Map<number, AskSource>;
  activeN: number | null;
  onCite: (n: number, el: HTMLElement) => void;
  chartFor?: (heading: string) => SectionChartData | null;
}) {
  const [openCharts, setOpenCharts] = useState<Set<number>>(new Set());
  // split into homogeneous segments: a heading, a run of bullets, a run of
  // "> " note lines, or a run of plain lines each become their own segment,
  // even when the model omits blank lines between them (it often writes a
  // bold finding with the bullets directly underneath in the same block)
  const lineKind = (l: string) =>
    /^#{1,4}\s/.test(l) ? "heading"
    : l.startsWith("- ") ? "bullet"
    : l.startsWith("> ") ? "note"
    : l.startsWith("|") ? "table"
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

  const renderSegment = (block: string, i: number) => {
        const heading = block.match(/^(#{1,4})\s+(.*)$/);
        if (heading) {
          const chart = chartFor?.(heading[2]) ?? null;
          if (!chart) return <h4 key={i}>{inline(heading[2])}</h4>;
          const open = openCharts.has(i);
          return (
            <div key={i}>
              <h4>
                {inline(heading[2])}{" "}
                <button
                  className={"ask-chart-btn" + (open ? " ask-chart-btn-on" : "")}
                  title="Full sub-theme breakdown for this section"
                  aria-expanded={open}
                  onClick={() =>
                    setOpenCharts((prev) => {
                      const next = new Set(prev);
                      if (next.has(i)) next.delete(i);
                      else next.add(i);
                      return next;
                    })
                  }
                >
                  ▤ breakdown
                </button>
              </h4>
              {open && <SectionChart data={chart} />}
            </div>
          );
        }
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
        if (lines[0].startsWith("|")) {
          // pipe table — tally answers (route "aggregate_direct") arrive as
          // one. Row 2 is the |---|---| separator; drop it, first row heads.
          const rows = lines
            .filter((l) => !/^\|[\s|:-]+\|$/.test(l))
            .map((l) =>
              l.replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()),
            );
          if (rows.length > 1) {
            const [head, ...body] = rows;
            return (
              <div key={i} className="ask-md-table-wrap">
                <table className="ask-md-table">
                  <thead>
                    <tr>
                      {head.map((c, j) => (
                        <th key={j}>{inline(c)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {body.map((r, j) => (
                      <tr key={j}>
                        {r.map((c, k) => (
                          <td key={k}>{inline(c)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }
        }
        const isLede = !sawParagraph;
        sawParagraph = true;
        return (
          <p key={i} className={isLede ? "ask-lede" : undefined}>
            {inline(block)}
          </p>
        );
  };

  // Group the flat segment list into sections: a "### " heading and every
  // block under it until the next heading. The intro (before the first
  // heading) keeps the full reading measure; the sections then flow
  // left-to-right and wrap, rather than stacking down one narrow column.
  const intro: number[] = [];
  const sections: { head: number; body: number[] }[] = [];
  segments.forEach((block, i) => {
    if (/^#{1,4}\s/.test(block)) sections.push({ head: i, body: [] });
    else if (sections.length) sections[sections.length - 1].body.push(i);
    else intro.push(i);
  });

  return (
    <div>
      {intro.length > 0 && (
        <div className="ask-intro">
          {intro.map((i) => renderSegment(segments[i], i))}
        </div>
      )}
      {sections.length > 0 && (
        <div className="ask-sections">
          {sections.map((s) => (
            <section key={s.head} className="ask-section">
              {renderSegment(segments[s.head], s.head)}
              {s.body.map((i) => renderSegment(segments[i], i))}
            </section>
          ))}
        </div>
      )}
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
  // split on **bold** and [n]/[n, m] citations, keeping the delimiters —
  // one bracket may carry several citations (the model sometimes writes
  // [1, 3] for [1][3]; the backend resolves both, so both must render)
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d{1,4}(?:\s*,\s*\d{1,4})*\])/g);
  return (
    <>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        const cite = part.match(/^\[(\d{1,4}(?:\s*,\s*\d{1,4})*)\]$/);
        if (cite) {
          const ns = cite[1].split(",").map((s) => Number(s.trim()));
          return (
            <span key={i}>
              {ns.map((n, j) =>
                sourceNs.has(n) ? (
                  <button
                    key={j}
                    className={`ask-cite${activeN === n ? " ask-cite-active" : ""}`}
                    title="Show the cited response"
                    onClick={(e) => onCite(n, e.currentTarget)}
                  >
                    {n}
                  </button>
                ) : (
                  <sup key={j}>[{n}]</sup>
                )
              )}
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
