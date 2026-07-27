"""Phase 2: taxonomy induction (map -> merge -> reviewable candidate taxonomy).

Pipeline, per question:
  1. Load the long-format parquet, filter sentinel non-answers (recorded,
     never deleted from source).
  2. Seeded shuffle, split into disjoint chunks — every response passes
     through exactly one chunk prompt.
  3. MAP: one model call per chunk proposes flat, granular candidate
     categories. Evidence is cited by response *number in the prompt*; code
     resolves numbers back to response_key + verbatim text, so no quote can
     be hallucinated (invalid citations are dropped and counted).
  4. MERGE stage A (deterministic): candidates whose normalized names match
     are merged automatically. Recorded as mechanism="exact_name".
  5. SORT (one LLM call): every remaining candidate is filed under a broad
     parent theme. Sorting only — nothing is renamed, combined, or dropped.
  6. DEDUP (one small LLM call per theme): within a single theme, candidates
     naming the same idea are grouped. Recorded as mechanism="llm_dedup". A
     candidate that only restates its theme is absorbed into the parent
     (mechanism="absorbed_into_parent", provenance preserved on the parent);
     that is the only way a candidate stops being a label. Anything the model
     forgets to mention survives untouched.
  6b. CROSS-THEME (one LLM call): per-theme dedup cannot see a duplicate that
     was sorted into two different themes. This pass looks only for that, over
     the short list of survivors. Recorded as mechanism="llm_cross_theme". A
     group whose members share a theme is rejected — that theme's dedup call
     already ruled on them with more context than this pass has.

     Sort-then-dedup rather than one global merge call: reconciling every
     candidate against every other in one pass is a hard problem that a model
     can "solve" by merging nothing, and cheaper models reliably do. After
     sorting, duplicates can only be siblings, so each dedup call is ~10
     names on a single topic — an easy problem, and the calls parallelize.
  7. Write candidate_taxonomy.json (the single hand-editable review
     artifact) + manifest.json (full audit record) to a fresh versioned run
     directory. Nothing upstream is modified.

Output is a two-level tree: `parents` carry no evidence of their own, and
every label carries `parent_id` (null if the reviewer should assign one).
Phase 3 labeling keys off child label_ids, never parent ids.

chunk_support = number of distinct chunks that independently proposed the
concept. It is a replication signal, NOT an estimate of corpus prevalence.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import ModelClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TAXONOMY_DIR = DATA_DIR / "taxonomy"

SCHEMA_VERSION = 2          # v2 = two-level tree (parents + labels.parent_id)
MAX_RESPONSE_CHARS = 500          # truncation cap inside prompts (counted in manifest)
MAX_EXAMPLES_PER_LABEL = 6
MAX_EVIDENCE_PER_CANDIDATE = 10
LARGE_MERGE_GROUP = 4             # groups this size or bigger get needs_review

# Pure null tokens only. "none"/"nothing" are deliberately NOT here: for a
# question like "what makes you feel unsafe" they are a real answer (nothing
# does) and the taxonomy should surface that as a category, not lose it.
SENTINEL_NON_ANSWERS = {
    "n/a", "na", "n.a", "n.a.", "idk", "i don't know", "i dont know",
    "dont know", "don't know", "no comment", "nil", "nada", "x", "xx",
    "xxx", "?", "??", "???", "-", "--", ".", "..", "...", "unsure",
    "not sure",
}

TEXT_COLUMN_CANDIDATES = ["raw_text", "response_text", "text"]
QUESTION_TEXT_COLUMN_CANDIDATES = ["question_text", "question_label", "label"]

# TEMPORARY. Phase 1 will collect this at upload and store it on the Dataset;
# this constant only exists so the plumbing can be exercised before that UI
# lands. Purely descriptive by design: it says what the survey is and who
# answered it, never what the analyst hopes to find. Stating an area of
# interest here would bias induction toward confirming it, which is exactly
# what "grounded ONLY in these responses" is meant to prevent.
DEFAULT_DATASET_DESCRIPTION = (
    "This survey is the Community Focus Area survey for San José. It asks "
    "residents a variety of questions about San José and how it can be "
    "improved in various ways."
)

# ---------------------------------------------------------------------------
# Prompts (hashed into the manifest; edit = new prompt version)
# ---------------------------------------------------------------------------


def context_block(description: str) -> str:
    """Render the dataset description as a prompt preamble. Empty description
    produces an empty string, so runs without one are byte-identical to how
    the prompts looked before this existed."""
    d = (description or "").strip()
    return f"Survey context:\n{d}\n\n" if d else ""


MAP_SYSTEM = """\
{dataset_context}You are inducing a candidate coding taxonomy for one open-ended survey question.

Survey question shown to respondents:
"{question_text}"

You will receive a numbered list of verbatim responses. Propose a FLAT list of
candidate categories grounded ONLY in these responses.

Rules:
- Flat list. No hierarchy, no parent/child, no grouping headers.
- Stay GRANULAR where the data is granular: distinct specific issues get
  distinct categories (e.g. car break-ins, armed robbery, and gang activity
  are three categories, never one generic "crime"). A generic category is
  allowed only for responses that are themselves generic.
- Interpret every response in the context of THIS question's wording.
- If some respondents say the premise does not apply to them (e.g. nothing
  makes them feel unsafe), that is a real category — include it.
- A response may be evidence for multiple categories.
- Do not shrink the list to look tidy. 10-30 categories is typical; follow
  the data.
- Never estimate counts, frequencies, or percentages anywhere in the output.
- "evidence": up to {max_evidence} response numbers copied exactly from the
  list, citing responses that clearly belong to the category.
- "description": one or two sentences of operating instructions for a later
  labeling model — literal and testable, no rhetoric.
- "include": 2-4 short criteria stating what belongs.
- "exclude": 1-3 boundary statements distinguishing this category from the
  categories it is most likely to be confused with.

Return ONLY valid JSON, exactly this shape:
{{"categories": [{{"name": "...", "description": "...", "include": ["..."],
"exclude": ["..."], "evidence": [1, 2]}}]}}
"""

MAP_USER = """Responses ({n} total):
{numbered_responses}
"""

ASSIGN_SYSTEM = """\
{dataset_context}You are organizing candidate categories for one open-ended survey question.

Survey question shown to respondents:
"{question_text}"

You will receive candidate categories (id, name, description). Sort ALL of
them into a small set of broad parent themes. That is the only task right now.

Rules:
- No id may appear under more than one parent.
- Use 5-8 parents. Prefer broad, reusable theme names that would also make
  sense for a different survey question — for example: property crime,
  violent crime, policing and justice, homelessness, transportation,
  cleanliness and infrastructure, cost of living, city governance.
- Sort a candidate only where it genuinely belongs. If a candidate fits none
  of your themes, LEAVE IT OUT of every parent — it gets flagged for a human
  instead. Do NOT invent a catch-all theme, and do NOT widen a theme's
  meaning to swallow leftovers: a theme whose members have nothing to do with
  each other is worse than no theme at all.
- Do NOT rename, rewrite, combine, or delete any candidate here. Sorting only.
- Never estimate counts or frequencies.

Return ONLY valid JSON, exactly this shape:
{{"parents": [{{"name": "...", "description": "...",
"members": ["c00_03", "c01_07"]}}]}}
"""

ASSIGN_USER = """Candidate categories ({n} total):
{candidate_lines}
"""

DEDUP_SYSTEM = """\
{dataset_context}Several people each read a different sample of responses to the same survey
question, and each wrote their own category names without seeing anyone
else's list. Those lists have been sorted into themes. You are looking at ONE
theme.

Survey question shown to respondents:
"{question_text}"

Theme: {parent_name}

Because the readers worked separately, the SAME idea usually appears several
times under different wording. Group the ones that are the same idea.

Rules:
- Every id goes into exactly one group. Do not leave any out.
- Ids describing the same underlying idea belong in the same group, even when
  the wording differs a lot. This is the common case — expect most groups to
  have more than one member.
- Keep genuinely different specifics apart. Stealing a whole car and breaking
  into a parked car are different ideas, so they stay in different groups. A
  group with a single member is right when that candidate is genuinely
  distinct from every other one here.
- Name each group with the clearest name among its members, or a better one.
- If a candidate is not a specific idea at all, and only restates the theme
  "{parent_name}" as a whole, put its id in "too_broad" instead of a group.
- Never estimate counts or frequencies.

Return ONLY valid JSON, exactly this shape:
{{"groups": [{{"ids": ["c00_03", "c01_07"], "name": "..."}}],
"too_broad": ["c02_01"]}}
"""

DEDUP_USER = """Categories under "{parent_name}" ({n} total):
{candidate_lines}
"""

# Note the deliberately inverted default here. Within a theme, duplicates are
# the common case, so that prompt pushes toward merging. Across themes they are
# rare — almost every pair is genuinely distinct — so this one pushes toward
# leaving things alone and says outright that an empty answer is correct.
CROSS_SYSTEM = """\
{dataset_context}These categories have already been sorted into themes and deduplicated inside
each theme. One kind of duplicate can survive that: the same idea filed under
two different themes, which no earlier step was able to see.

Survey question shown to respondents:
"{question_text}"

You will receive every surviving category with the theme it sits under. Find
only groups that name the SAME idea across DIFFERENT themes.

Rules:
- Only group ids that sit under DIFFERENT themes. Ids sharing a theme were
  already checked — leave them alone.
- Only group ids that name the same idea. Being related, or both being about
  crime, is not enough.
- Most ids belong in no group at all. Returning an empty list is a correct and
  expected answer. Do not hunt for something to merge.
- Never estimate counts or frequencies.

Return ONLY valid JSON, exactly this shape:
{{"groups": [{{"ids": ["c00_03", "c01_07"], "name": "..."}}]}}
"""

CROSS_USER = """Surviving categories ({n} total):
{candidate_lines}
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResponseRow:
    response_key: str
    text: str


@dataclass
class Candidate:
    cid: str                      # e.g. "c00_03" = chunk 0, category 3
    chunk_index: int
    name: str
    description: str
    include: list[str]
    exclude: list[str]
    evidence: list[ResponseRow]   # resolved, verified rows


@dataclass
class ProvisionalLabel:
    """A candidate or an exact-name merge of candidates, pre-LLM-merge."""
    pid: str
    name: str
    description: str
    include: list[str]
    exclude: list[str]
    members: list[Candidate]

    @property
    def chunks(self) -> list[int]:
        return sorted({m.chunk_index for m in self.members})


@dataclass
class ProvisionalParent:
    """A broad theme grouping specific child labels. Carries no evidence of
    its own — all evidence lives on the children, so a parent can never be
    the thing a quote traces back to. `absorbed` holds candidates from a
    broad label that was folded into this parent rather than kept as a child
    of itself; kept for audit, never silently discarded."""
    name: str
    description: str
    child_pids: list[str]
    rationale: str
    absorbed: list[Candidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading & filtering
# ---------------------------------------------------------------------------


def discover_parquet(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")
        return p
    hits: list[Path] = []
    for pattern in ("exports/*/reshaped.parquet", "exports/*/responses.parquet"):
        hits.extend((DATA_DIR).glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No parquet found under {DATA_DIR / 'exports'}. Pass --parquet explicitly."
        )
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(hits) > 1:
        print(f"NOTE: {len(hits)} export parquets found; using most recent: {hits[0]}")
    return hits[0]


def _pick_column(columns: list[str], candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"No {what} column found. Tried {candidates}; parquet has {columns}")


def load_question(parquet_path: Path, question: str) -> tuple[list[ResponseRow], dict, list[dict]]:
    """Return (usable rows, meta, filtered non-answers). Inspects the actual
    schema instead of trusting any listing."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    columns = list(df.columns)
    if "question_id" not in columns:
        raise KeyError(f"Parquet has no question_id column; columns: {columns}")
    text_col = _pick_column(columns, TEXT_COLUMN_CANDIDATES, "response text")

    qmask = df["question_id"].astype(str) == str(question)
    sub = df[qmask]
    if sub.empty:
        available = (
            df["question_id"].astype(str).value_counts().to_dict()
        )
        raise SystemExit(
            f"Question {question!r} not in parquet. Available (id: n): {available}"
        )

    # question wording — load-bearing for the prompt
    question_text = str(question)
    for c in QUESTION_TEXT_COLUMN_CANDIDATES:
        if c in columns:
            vals = sub[c].dropna().unique()
            if len(vals):
                question_text = str(vals[0])
                break

    dataset_id = ""
    if "dataset_id" in columns:
        ids = sub["dataset_id"].dropna().unique()
        if len(ids):
            dataset_id = str(ids[0])
    if not dataset_id:
        dataset_id = parquet_path.parent.name  # exports/{dataset_id}/file.parquet

    rows: list[ResponseRow] = []
    filtered: list[dict] = []
    n_empty = 0
    for pos, (idx, r) in enumerate(sub.iterrows()):
        text = r[text_col]
        text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
        if "response_key" in columns and r["response_key"] and str(r["response_key"]) != "nan":
            key = str(r["response_key"])
        else:
            sri = r["source_row_index"] if "source_row_index" in columns else idx
            key = f"{dataset_id}:{question}:{sri}"
        stripped = text.strip()
        if not stripped:
            n_empty += 1
            continue
        if _is_sentinel(stripped):
            filtered.append({"response_key": key, "text": stripped})
            continue
        rows.append(ResponseRow(response_key=key, text=stripped))

    meta = {
        "parquet_path": str(parquet_path),
        "parquet_columns": columns,
        "text_column_used": text_col,
        "dataset_id": dataset_id,
        "question_id": str(question),
        "question_text": question_text,
        "rows_for_question": int(len(sub)),
        "rows_empty": n_empty,
        "rows_sentinel_filtered": len(filtered),
        "rows_used": len(rows),
    }
    return rows, meta, filtered


def _is_sentinel(text: str) -> bool:
    s = text.strip().lower()
    s = s.strip(" \t.!?")
    return not s or s in SENTINEL_NON_ANSWERS


def list_questions(parquet_path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    out = []
    for qid, grp in df.groupby("question_id"):
        wording = ""
        for c in QUESTION_TEXT_COLUMN_CANDIDATES:
            if c in df.columns:
                vals = grp[c].dropna().unique()
                if len(vals):
                    wording = str(vals[0])
                    break
        out.append({"question_id": str(qid), "question_text": wording, "n_rows": int(len(grp))})
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def make_chunks(rows: list[ResponseRow], chunk_size: int, seed: int) -> list[list[ResponseRow]]:
    """Seeded shuffle then near-equal disjoint slices. Every row lands in
    exactly one chunk — no sampling."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_chunks = max(1, math.ceil(len(rows) / chunk_size))
    base, extra = divmod(len(rows), n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append([rows[j] for j in order[start : start + size]])
        start += size
    return chunks


# ---------------------------------------------------------------------------
# JSON robustness
# ---------------------------------------------------------------------------


def extract_json(raw: str) -> dict:
    """Parse model output as JSON, tolerating code fences / stray prose."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in model output: {s[:300]!r}")
        return json.loads(s[start : end + 1])


def _str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------


def build_map_prompts(
    question_text: str, chunk: list[ResponseRow], dataset_description: str = ""
) -> tuple[str, str, int]:
    """Returns (system, user, n_truncated)."""
    lines, n_truncated = [], 0
    for i, row in enumerate(chunk, start=1):
        t = row.text.replace("\n", " ").strip()
        if len(t) > MAX_RESPONSE_CHARS:
            t = t[:MAX_RESPONSE_CHARS] + "…"
            n_truncated += 1
        lines.append(f"{i}. {t}")
    system = MAP_SYSTEM.format(
        dataset_context=context_block(dataset_description),
        question_text=question_text,
        max_evidence=MAX_EVIDENCE_PER_CANDIDATE,
    )
    user = MAP_USER.format(n=len(chunk), numbered_responses="\n".join(lines))
    return system, user, n_truncated


def parse_map_output(raw: str, chunk_index: int, chunk: list[ResponseRow]) -> tuple[list[Candidate], int]:
    """Validate & resolve one chunk's proposals. Returns (candidates,
    n_invalid_citations)."""
    obj = extract_json(raw)
    cats = obj.get("categories")
    if not isinstance(cats, list):
        raise ValueError(f"chunk {chunk_index}: output missing 'categories' list")
    candidates, invalid = [], 0
    for j, c in enumerate(cats):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        evidence: list[ResponseRow] = []
        seen: set[str] = set()
        for e in c.get("evidence", []) or []:
            try:
                num = int(e)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if 1 <= num <= len(chunk):
                row = chunk[num - 1]
                if row.response_key not in seen:
                    seen.add(row.response_key)
                    evidence.append(row)
            else:
                invalid += 1
        candidates.append(
            Candidate(
                cid=f"c{chunk_index:02d}_{j:02d}",
                chunk_index=chunk_index,
                name=name,
                description=str(c.get("description", "")).strip(),
                include=_str_list(c.get("include")),
                exclude=_str_list(c.get("exclude")),
                evidence=evidence[:MAX_EVIDENCE_PER_CANDIDATE],
            )
        )
    return candidates, invalid


# ---------------------------------------------------------------------------
# MERGE stage A: deterministic exact-name merge
# ---------------------------------------------------------------------------


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", name.lower()).strip()


def auto_merge(candidates: list[Candidate]) -> list[ProvisionalLabel]:
    by_norm: dict[str, list[Candidate]] = {}
    order: list[str] = []
    for c in candidates:
        key = _norm_name(c.name)
        if key not in by_norm:
            by_norm[key] = []
            order.append(key)
        by_norm[key].append(c)
    labels = []
    for i, key in enumerate(order):
        members = by_norm[key]
        # keep the longest description among identically-named proposals
        best = max(members, key=lambda m: len(m.description))
        labels.append(
            ProvisionalLabel(
                pid=members[0].cid,
                name=members[0].name,
                description=best.description,
                include=_dedupe([x for m in members for x in m.include]),
                exclude=_dedupe([x for m in members for x in m.exclude]),
                members=members,
            )
        )
    return labels


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# MERGE stage B: one LLM pass, rationale required, combine-only
# ---------------------------------------------------------------------------


def _candidate_lines(labels: list[ProvisionalLabel]) -> str:
    return "\n".join(
        f"- id={lab.pid} | {lab.name} :: {lab.description.replace(chr(10), ' ')}"
        for lab in labels
    )


def build_assign_prompts(
    question_text: str, labels: list[ProvisionalLabel], dataset_description: str = ""
) -> tuple[str, str]:
    return (
        ASSIGN_SYSTEM.format(
            dataset_context=context_block(dataset_description), question_text=question_text
        ),
        ASSIGN_USER.format(n=len(labels), candidate_lines=_candidate_lines(labels)),
    )


def build_dedup_prompts(
    question_text: str,
    parent_name: str,
    labels: list[ProvisionalLabel],
    dataset_description: str = "",
) -> tuple[str, str]:
    return (
        DEDUP_SYSTEM.format(
            dataset_context=context_block(dataset_description),
            question_text=question_text,
            parent_name=parent_name,
        ),
        DEDUP_USER.format(
            parent_name=parent_name, n=len(labels), candidate_lines=_candidate_lines(labels)
        ),
    )


def build_cross_prompts(
    question_text: str,
    labels: list[ProvisionalLabel],
    theme_of: dict[str, str],
    dataset_description: str = "",
) -> tuple[str, str]:
    lines = "\n".join(
        f"- id={lab.pid} | theme={theme_of.get(lab.pid) or '(unsorted)'} | {lab.name} "
        f":: {lab.description.replace(chr(10), ' ')}"
        for lab in labels
    )
    return (
        CROSS_SYSTEM.format(
            dataset_context=context_block(dataset_description), question_text=question_text
        ),
        CROSS_USER.format(n=len(labels), candidate_lines=lines),
    )


def apply_cross_merges(
    raw: str,
    labels: list[ProvisionalLabel],
    parents: list[ProvisionalParent],
    theme_of: dict[str, str],
) -> tuple[list[ProvisionalLabel], list[dict], list[str]]:
    """Merge duplicates that ended up under different themes. Mutates each
    parent's child_pids to match. Returns (labels, merge_log, warnings).

    A group whose ids all share one theme is rejected: that theme's dedup call
    already ruled on them, and letting this pass re-litigate it would silently
    override a more informed decision made with the full theme in view."""
    obj = extract_json(raw)
    by_pid = {lab.pid: lab for lab in labels}
    used: set[str] = set()
    merge_log: list[dict] = []
    warnings: list[str] = []
    absorbed_into: dict[str, str] = {}      # dropped pid -> surviving pid

    for g in obj.get("groups") or []:
        if not isinstance(g, dict):
            continue
        ids: list[str] = []
        for raw_id in _str_list(g.get("ids")):
            if raw_id not in by_pid:
                warnings.append(f"cross-theme group: unknown id {raw_id!r} ignored")
            elif raw_id in used:
                warnings.append(f"cross-theme group: id {raw_id!r} already merged, ignored")
            elif raw_id not in ids:
                ids.append(raw_id)
        if len(ids) < 2:
            continue
        if len({theme_of.get(i) for i in ids}) < 2:
            warnings.append(
                f"cross-theme group {ids}: all in one theme, rejected "
                "(already decided by that theme's dedup pass)"
            )
            continue
        used.update(ids)
        parts = [by_pid[i] for i in ids]
        merged = _combine(parts, str(g.get("name", "")).strip())
        by_pid[merged.pid] = merged
        for dropped in ids[1:]:
            absorbed_into[dropped] = merged.pid
        merge_log.append(
            {
                "mechanism": "llm_cross_theme",
                "result_name": merged.name,
                "result_pid": merged.pid,
                "member_ids": ids,
                "member_names": [p.name for p in parts],
                "member_themes": [theme_of.get(i) or "(unsorted)" for i in ids],
                "rationale": "same idea sorted into different themes",
            }
        )

    if not absorbed_into:
        return labels, merge_log, warnings

    # the merged label keeps the theme of its first member; the losing parents
    # simply lose a child
    for parent in parents:
        parent.child_pids = [p for p in parent.child_pids if p not in absorbed_into]
    out = [by_pid[lab.pid] for lab in labels if lab.pid not in absorbed_into]
    return out, merge_log, warnings


def parse_assignment(
    raw: str, labels: list[ProvisionalLabel]
) -> tuple[list[tuple[str, str, list[str]]], list[str]]:
    """Sort labels into (parent_name, description, pids) buckets.

    Anything the model failed to place lands in a trailing unnamed bucket
    rather than being dropped — an unsorted candidate is still a real
    candidate, and the reviewer needs to see it."""
    obj = extract_json(raw)
    by_id = {lab.pid: lab for lab in labels}
    placed: set[str] = set()
    groups: list[tuple[str, str, list[str]]] = []
    warnings: list[str] = []

    for g in obj.get("parents") or []:
        if not isinstance(g, dict):
            continue
        name = str(g.get("name", "")).strip()
        if not name:
            continue
        pids: list[str] = []
        for raw_id in _str_list(g.get("members")):
            if raw_id not in by_id:
                warnings.append(f"theme {name!r}: unknown id {raw_id!r} ignored")
            elif raw_id in placed:
                warnings.append(f"theme {name!r}: id {raw_id!r} already sorted elsewhere, ignored")
            elif raw_id not in pids:
                pids.append(raw_id)
        if not pids:
            warnings.append(f"theme {name!r}: no valid members, dropped")
            continue
        placed.update(pids)
        groups.append((name, str(g.get("description", "")).strip(), pids))

    leftover = [lab.pid for lab in labels if lab.pid not in placed]
    if leftover:
        warnings.append(f"{len(leftover)} candidate(s) not sorted into any theme; left unparented")
        groups.append(("", "", leftover))
    return groups, warnings


def _combine(labels: list[ProvisionalLabel], name: str) -> ProvisionalLabel:
    """Fold several provisional labels into one. Keeps the longest description
    and the union of include/exclude — no model call, no information dropped."""
    best = max(labels, key=lambda m: len(m.description))
    return ProvisionalLabel(
        pid=labels[0].pid,
        name=name or labels[0].name,
        description=best.description,
        include=_dedupe([x for m in labels for x in m.include]),
        exclude=_dedupe([x for m in labels for x in m.exclude]),
        members=[c for m in labels for c in m.members],
    )


def apply_dedup(
    raw: str, labels: list[ProvisionalLabel], parent_name: str
) -> tuple[list[ProvisionalLabel], list[Candidate], list[dict], list[str]]:
    """Apply one theme's grouping. Returns (labels, absorbed candidates,
    merge_log entries, warnings).

    Guards: invented ids ignored; an id claimed by two groups goes to the
    first; an id the model never mentioned survives as its own label rather
    than vanishing."""
    obj = extract_json(raw)
    by_id = {lab.pid: lab for lab in labels}
    used: set[str] = set()
    out: list[ProvisionalLabel] = []
    merge_log: list[dict] = []
    warnings: list[str] = []

    too_broad: list[str] = []
    for raw_id in _str_list(obj.get("too_broad")):
        if raw_id not in by_id:
            warnings.append(f"theme {parent_name!r}: unknown too_broad id {raw_id!r} ignored")
        elif raw_id not in too_broad:
            too_broad.append(raw_id)
    used.update(too_broad)

    for g in obj.get("groups") or []:
        if not isinstance(g, dict):
            continue
        ids: list[str] = []
        for raw_id in _str_list(g.get("ids")):
            if raw_id not in by_id:
                warnings.append(f"theme {parent_name!r}: unknown id {raw_id!r} ignored")
            elif raw_id in used:
                warnings.append(f"theme {parent_name!r}: id {raw_id!r} already grouped, ignored")
            elif raw_id not in ids:
                ids.append(raw_id)
        if not ids:
            continue
        used.update(ids)
        name = str(g.get("name", "")).strip()
        parts = [by_id[i] for i in ids]
        out.append(_combine(parts, name))
        if len(ids) > 1:
            merge_log.append(
                {
                    "mechanism": "llm_dedup",
                    "result_name": out[-1].name,
                    "result_pid": out[-1].pid,
                    "member_ids": ids,
                    "member_names": [p.name for p in parts],
                    "rationale": f"same idea proposed separately under theme {parent_name!r}",
                }
            )

    # never silently lose a candidate the model forgot to mention
    forgotten = [lab for lab in labels if lab.pid not in used]
    if forgotten:
        warnings.append(
            f"theme {parent_name!r}: {len(forgotten)} id(s) not mentioned, kept as-is"
        )
        out.extend(forgotten)

    absorbed: list[Candidate] = []
    for pid in too_broad:
        absorbed.extend(by_id[pid].members)
        merge_log.append(
            {
                "mechanism": "absorbed_into_parent",
                "result_name": parent_name,
                "result_pid": pid,
                "member_ids": [pid],
                "member_names": [by_id[pid].name],
                "rationale": "only restated the theme; folded into the parent",
            }
        )
    # a theme whose every member was called too broad keeps them: absorbing
    # everything would delete the theme's entire contents
    if not out and absorbed:
        warnings.append(f"theme {parent_name!r}: all members marked too_broad, absorption skipped")
        return [by_id[pid] for pid in too_broad], [], [], warnings
    return out, absorbed, merge_log, warnings



# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def build_taxonomy(
    labels: list[ProvisionalLabel],
    parents: list[ProvisionalParent],
    meta: dict,
    n_chunks: int,
) -> dict:
    def sort_key(lab: ProvisionalLabel):
        return (-len(lab.chunks), lab.name.lower())

    parent_id_of: dict[str, str] = {}
    out_parents = []
    for pseq, par in enumerate(sorted(parents, key=lambda p: p.name.lower()), start=1):
        parent_id = f"{meta['question_id']}_P{pseq:02d}"
        for pid in par.child_pids:
            parent_id_of[pid] = parent_id
        out_parents.append(
            {
                "parent_id": parent_id,
                "name": par.name,
                "description": par.description,
                "rationale": par.rationale,
                "child_label_ids": [],   # filled once labels have ids
                "absorbed": [
                    {"chunk": c.chunk_index, "cid": c.cid, "name": c.name}
                    for c in par.absorbed
                ],
            }
        )
    parents_by_id = {p["parent_id"]: p for p in out_parents}

    out_labels = []
    for seq, lab in enumerate(sorted(labels, key=sort_key), start=1):
        chunk_support = len(lab.chunks)
        # examples: round-robin across member candidates for diversity
        examples, seen = [], set()
        pools = [list(m.evidence) for m in lab.members]
        while len(examples) < MAX_EXAMPLES_PER_LABEL and any(pools):
            for pool in pools:
                while pool:
                    row = pool.pop(0)
                    if row.response_key not in seen:
                        seen.add(row.response_key)
                        examples.append({"response_key": row.response_key, "text": row.text})
                        break
                if len(examples) >= MAX_EXAMPLES_PER_LABEL:
                    break
        merged_from = [
            {
                "chunk": m.chunk_index,
                "cid": m.cid,
                "name": m.name,
                "description": m.description,
            }
            for m in lab.members
        ]
        singleton = chunk_support == 1 and n_chunks > 1
        parent_id = parent_id_of.get(lab.pid)
        needs_review = (
            singleton or len(lab.members) >= LARGE_MERGE_GROUP or parent_id is None
        )
        label_id = f"{meta['question_id']}_{seq:03d}"
        if parent_id is not None:
            parents_by_id[parent_id]["child_label_ids"].append(label_id)
        out_labels.append(
            {
                "label_id": label_id,
                "parent_id": parent_id,
                "name": lab.name,
                "description": lab.description,
                "include": lab.include,
                "exclude": lab.exclude,
                "examples": examples,
                "chunk_support": chunk_support,
                "chunk_support_note": (
                    f"proposed independently by {chunk_support} of {n_chunks} chunks; "
                    "replication signal only, NOT corpus prevalence"
                ),
                "singleton": singleton,
                "needs_review": needs_review,
                "provenance": {"merged_from": merged_from},
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_for_review",
        "dataset_id": meta["dataset_id"],
        "question_id": meta["question_id"],
        "question_text": meta["question_text"],
        "review_instructions": (
            "Hand-edit this file: rename labels, tighten descriptions/include/"
            "exclude, delete a label by removing its object, merge two labels by "
            "moving one's provenance.merged_from entries into the other and "
            "deleting it. Re-parent a label by editing its parent_id and the "
            "parent's child_label_ids to match. A label with parent_id null is "
            "unparented and needs one assigned (or a new parent added). "
            "label_id values freeze at approval — downstream labeling keys off "
            "them and NEVER off parent_id, so parents stay editable afterwards. "
            "chunk_support is a replication signal across induction chunks, not "
            "a count of respondents — real counts arrive with the labeling pass."
        ),
        "parents": out_parents,
        "labels": out_labels,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _complete_json(client: ModelClient, system: str, user: str) -> str:
    """One call, with a single retry that says plainly what went wrong. Kept
    here so every consolidation step gets the same treatment."""
    raw = client.complete(system, user)
    try:
        extract_json(raw)
        return raw
    except (ValueError, json.JSONDecodeError):
        return client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )


def run_induction(
    rows: list[ResponseRow],
    meta: dict,
    client: ModelClient,
    chunk_size: int = 120,
    seed: int = 7,
    dataset_description: str = "",
) -> tuple[dict, dict]:
    """Full pipeline over already-loaded rows. Returns (taxonomy, run_report)."""
    chunks = make_chunks(rows, chunk_size, seed)
    print(f"{len(rows)} responses -> {len(chunks)} chunk(s) "
          f"(sizes: {[len(c) for c in chunks]}, seed={seed})")

    all_candidates: list[Candidate] = []
    per_chunk_stats, total_invalid, total_truncated = [], 0, 0
    failed_chunks: list[dict] = []
    for i, chunk in enumerate(chunks):
        system, user, n_trunc = build_map_prompts(meta["question_text"], chunk, dataset_description)
        total_truncated += n_trunc
        try:
            raw = client.complete(system, user)
            try:
                cands, invalid = parse_map_output(raw, i, chunk)
            except (ValueError, json.JSONDecodeError):
                # one retry with an explicit nudge
                raw = client.complete(
                    system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
                    user,
                )
                cands, invalid = parse_map_output(raw, i, chunk)
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            # One bad chunk must not destroy an otherwise good run. The chunk's
            # responses go uncovered, so this is recorded in the manifest rather
            # than swallowed — a taxonomy built from 8 of 9 chunks has to say so.
            failed_chunks.append({"chunk": i, "n_responses": len(chunk),
                                  "error": f"{type(exc).__name__}: {exc}"[:300]})
            per_chunk_stats.append({"chunk": i, "n_responses": len(chunk),
                                    "n_candidates": 0, "invalid_citations": 0,
                                    "failed": True})
            print(f"  chunk {i}: FAILED after retry ({type(exc).__name__}) — skipping, "
                  f"{len(chunk)} responses uncovered")
            continue
        total_invalid += invalid
        per_chunk_stats.append({"chunk": i, "n_responses": len(chunk),
                                "n_candidates": len(cands), "invalid_citations": invalid})
        print(f"  chunk {i}: {len(cands)} candidates, {invalid} invalid citations")
        all_candidates.extend(cands)

    n_ok = len(chunks) - len(failed_chunks)
    if not n_ok:
        raise RuntimeError(
            f"All {len(chunks)} chunks failed; no taxonomy to build. First error: "
            f"{failed_chunks[0]['error']}"
        )

    provisional = auto_merge(all_candidates)
    exact_merge_log = [
        {
            "mechanism": "exact_name",
            "result_name": lab.name,
            "member_ids": [m.cid for m in lab.members],
            "member_names": [m.name for m in lab.members],
            "rationale": "identical normalized name across chunks",
        }
        for lab in provisional
        if len(lab.members) > 1
    ]

    warnings: list[str] = []
    llm_merge_log: list[dict] = []
    parents: list[ProvisionalParent] = []
    final = provisional
    if n_ok > 1 and len(provisional) > 1:
        # Sort into themes first, then dedupe inside each theme. Two easy jobs
        # beat one hard one: a single global "merge these 61 candidates" call
        # asks a model to hold every pairwise comparison at once, and the
        # cheaper the model the more reliably it answers by merging nothing.
        # Sorting is easy, and once sorted, duplicates can only be siblings —
        # so each dedup call sees ~10 names on one topic.
        system, user = build_assign_prompts(meta["question_text"], provisional, dataset_description)
        raw = _complete_json(client, system, user)
        groups, warns = parse_assignment(raw, provisional)
        warnings += warns
        print(f"  sorted into {len(groups)} theme(s)")

        by_pid = {lab.pid: lab for lab in provisional}
        final = []
        for name, description, pids in groups:
            members = [by_pid[p] for p in pids]
            if len(members) < 2:
                final.extend(members)
                kept, absorbed = members, []
            else:
                system, user = build_dedup_prompts(meta["question_text"], name or "Unsorted", members,
                                                   dataset_description)
                raw = _complete_json(client, system, user)
                kept, absorbed, log, warns = apply_dedup(raw, members, name or "Unsorted")
                llm_merge_log += log
                warnings += warns
                final.extend(kept)
            print(f"  theme {name or '(unsorted)'!r}: {len(members)} -> {len(kept)} label(s)"
                  + (f", {len(absorbed)} absorbed" if absorbed else ""))
            if name and kept:
                parents.append(
                    ProvisionalParent(
                        name=name,
                        description=description,
                        child_pids=[lab.pid for lab in kept],
                        rationale="theme assigned in the sort step",
                        absorbed=absorbed,
                    )
                )

        # Last pass: the same idea filed under two themes is invisible to
        # per-theme dedup by construction. Cheap here — the list is short and
        # already clean, unlike the raw candidate pile a global merge would face.
        theme_of = {pid: p.name for p in parents for pid in p.child_pids}
        if len(parents) > 1 and len(final) > 1:
            system, user = build_cross_prompts(meta["question_text"], final, theme_of, dataset_description)
            raw = _complete_json(client, system, user)
            final, log, warns = apply_cross_merges(raw, final, parents, theme_of)
            llm_merge_log += log
            warnings += warns
            if log:
                print(f"  cross-theme: {len(log)} duplicate(s) merged across themes")

    # chunk_support is "proposed by k of N"; N must be the chunks that actually
    # answered, or a failed chunk silently deflates every label's support
    taxonomy = build_taxonomy(final, parents, meta, n_chunks=n_ok)

    run_report = {
        "chunks": {
            "n_chunks": len(chunks),
            "n_chunks_succeeded": n_ok,
            "failed_chunks": failed_chunks,
            "chunk_size_target": chunk_size,
            "seed": seed,
            "assignment": {
                str(i): [r.response_key for r in chunk] for i, chunk in enumerate(chunks)
            },
        },
        "per_chunk": per_chunk_stats,
        "responses_truncated_in_prompt": total_truncated,
        "invalid_evidence_citations": total_invalid,
        "candidates_proposed": len(all_candidates),
        "labels_after_exact_merge": len(provisional),
        "labels_final": len(taxonomy["labels"]),
        "parents_final": len(taxonomy["parents"]),
        "labels_unparented": [
            l["name"] for l in taxonomy["labels"] if l["parent_id"] is None
        ],
        "labels_absorbed_into_parents": [
            m["member_names"][0] for m in llm_merge_log
            if m["mechanism"] == "absorbed_into_parent"
        ],
        "merge_log": exact_merge_log + llm_merge_log,
        "merge_warnings": warnings,
        "singletons": [l["name"] for l in taxonomy["labels"] if l["singleton"]],
    }
    return taxonomy, run_report


def write_artifacts(taxonomy: dict, manifest: dict, out_root: Path | None = None) -> Path:
    root = out_root or TAXONOMY_DIR
    run_id = manifest["run_id"]
    out_dir = root / taxonomy["dataset_id"] / taxonomy["question_id"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


def prompt_hash(dataset_description: str = "") -> str:
    blob = (
        MAP_SYSTEM + MAP_USER + ASSIGN_SYSTEM + ASSIGN_USER
        + DEDUP_SYSTEM + DEDUP_USER + CROSS_SYSTEM + CROSS_USER
        + (dataset_description or "")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def print_diagnostics(taxonomy: dict, report: dict, usage_line: str) -> None:
    n_chunks = report["chunks"]["n_chunks_succeeded"]
    labels = taxonomy["labels"]
    print("\n=== RUN DIAGNOSTICS (good-run bands in brackets) ===")
    for s in report["per_chunk"]:
        if s.get("failed"):
            print(f"chunk {s['chunk']}: FAILED — {s['n_responses']} responses "
                  f"had no chance to contribute  <-- WARN")
            continue
        flag = "" if 8 <= s["n_candidates"] <= 35 else "  <-- WARN outside [8, 35]"
        print(f"chunk {s['chunk']}: {s['n_candidates']} candidates "
              f"from {s['n_responses']} responses{flag}")
    failed = report["chunks"]["failed_chunks"]
    if failed:
        uncovered = sum(f["n_responses"] for f in failed)
        print(f"CHUNKS FAILED: {len(failed)} of {report['chunks']['n_chunks']} "
              f"({uncovered} responses uncovered) — taxonomy is incomplete, re-run "
              f"before treating it as final")
    inv = report["invalid_evidence_citations"]
    print(f"invalid evidence citations: {inv}"
          + ("" if inv <= 2 * n_chunks else "  <-- WARN model is citing sloppily"))
    cp, lf = report["candidates_proposed"], report["labels_final"]
    ratio = cp / lf if lf else 0
    band = "" if (n_chunks == 1 or 1.2 <= ratio <= 3.5) else "  <-- WARN check merge_log for over/under-merge"
    print(f"merge compression: {cp} candidates -> {lf} labels ({ratio:.1f}x){band}")
    n_single = len(report["singletons"])
    frac = n_single / lf if lf else 0
    print(f"singletons (1 chunk only): {n_single}/{lf} ({frac:.0%})"
          + ("" if frac <= 0.4 or n_chunks == 1 else "  <-- WARN chunks disagree a lot"))
    n_par = report["parents_final"]
    kids = [len(p["child_label_ids"]) for p in taxonomy["parents"]]
    print(f"parents: {n_par} over {lf} labels (children per parent: {sorted(kids, reverse=True)})"
          + ("" if n_par == 0 or 3 <= n_par <= 9 else "  <-- WARN outside [3, 9]"))
    unp = report["labels_unparented"]
    if unp:
        print(f"UNPARENTED labels ({len(unp)}/{lf}): {unp}  <-- assign a parent or add one")
    absorbed = report["labels_absorbed_into_parents"]
    if absorbed:
        print(f"absorbed into a parent (no longer standalone labels): {absorbed}")
    no_ex = [l["name"] for l in labels if not l["examples"]]
    if no_ex:
        print(f"labels with ZERO verified examples: {no_ex}  <-- WARN review these first")
    big = [m for m in report["merge_log"] if len(m["member_ids"]) >= LARGE_MERGE_GROUP]
    if big:
        print(f"large merge groups (>= {LARGE_MERGE_GROUP} members): "
              f"{[m['result_name'] for m in big]}  <-- check for over-collapse")
    for w in report["merge_warnings"]:
        print(f"merge guard: {w}")
    print(usage_line)


def estimate_dry_run(rows: list[ResponseRow], meta: dict, chunk_size: int, seed: int,
                     price_in: float, price_out: float,
                     dataset_description: str = "") -> None:
    chunks = make_chunks(rows, chunk_size, seed)
    est_in = est_out = 0
    for chunk in chunks:
        system, user, _ = build_map_prompts(meta["question_text"], chunk, dataset_description)
        est_in += (len(system) + len(user)) // 4
        est_out += 2500
    if len(chunks) > 1:
        est_in += 4000
        est_out += 2000
    cost = est_in / 1e6 * price_in + est_out / 1e6 * price_out
    print(f"DRY RUN — no API calls made, nothing written.")
    print(f"  responses: {len(rows)} usable "
          f"({meta['rows_sentinel_filtered']} sentinel non-answers filtered, "
          f"{meta['rows_empty']} empty)")
    print(f"  plan: {len(chunks)} map call(s) + {1 if len(chunks) > 1 else 0} merge call")
    print(f"  est tokens: ~{est_in:,} in / ~{est_out:,} out")
    print(f"  est cost at ${price_in}/M in, ${price_out}/M out: ~${cost:.3f}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
