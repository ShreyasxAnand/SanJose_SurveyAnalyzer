"""Answer verification — deterministic guards that DISCLOSE, never rewrite.

Two integrity defects survived every upstream control (found in the
2026-08-12 QA review): a quotation whose wording exists in no source
("junkyard-like conditions" [53]), and a number the model computed itself
(383 for sub-themes summing to 368). Both are checkable against ground truth
we already hold, so the verifier here is plain code:

  * every claim-carrying number in the answer must trace to a computed count;
  * every quoted span attached to a citation must literally appear in that
    source's text;
  * every quote must be cited under the sub-theme it is coded to;
  * every section must state its section-plan count.

Every guard runs on every answer, costs no model call, and its findings are
disclosed in the verification record and the process note.

Nothing here edits the answer. A model repair call used to run when a guard
failed; it was removed 2026-08-25 after clearing the flags on one of the six
stored answers that reached it, and because its own instructions licensed it
to reword a quotation — a model that has just fabricated a quote cannot be
the thing that authors its replacement. The value of this module is that the
text an analyst reads is exactly the text the evidence produced, with its
defects named rather than papered over.
"""
from __future__ import annotations

import hashlib
import re

# Numbers below this are prose ("two of the three filters"), not claims —
# checking them would flag ordinary writing.
MIN_CHECKED_NUMBER = 13


def verify_logic_hash() -> str:
    """Folded into the ask cache key: a change to the verifier changes what
    answers say, so stored answers from the old verifier must not be served.

    Covers the DETECTION code. It used to hash only the repair prompts and
    MIN_CHECKED_NUMBER, which meant a fix to a guard left every stored answer
    carrying the old verdict — the two 2026-08-17 false-positive bugs
    (citations read as counts, the section regex) would have been fixed in code
    and still displayed as "8 statements could not be verified" on every cached
    answer. Deliberately over-inclusive: the patterns are listed explicitly
    because a regex edit does not change any function's source text."""
    import inspect as _inspect
    blob = "".join([
        str(MIN_CHECKED_NUMBER),
        _inspect.getsource(legit_numbers),
        _inspect.getsource(find_violations),
        _inspect.getsource(_tokens),
        _inspect.getsource(_match_sub),
        _inspect.getsource(_sub_theme_regions),
        _inspect.getsource(_placement_violations),
        _inspect.getsource(plan_structure_violations),
        _BOLD_RE.pattern, _UNIT_RE.pattern, _QUOTE_RE.pattern,
        _CITE_SPAN_RE.pattern, _SECTION_RE.pattern, _PLAN_COUNT_RE.pattern,
        _BOLD_NUM_RE.pattern, _SUB_BULLET_RE.pattern,
    ]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def legit_numbers(evidence: dict, lex_counts: list[dict],
                  question_totals: dict[str, int],
                  section_plan: str = "") -> set[int]:
    """Every integer the answer may legitimately state, from the computed
    evidence — plus simple derivations the templates and prompt rules invite
    (differences within one denominator, scope remainders), plus every number
    the SECTION PLAN states.

    The plan is ground truth, not a claim: render_section_plan builds it in
    code from full-coverage counts, and the answer is ORDERED to copy its
    numbers. Some of them exist nowhere in `evidence` — the "Everything else
    — N responses" line is a union counted over response keys at plan-build
    time, deliberately not a sum of anything. Without the plan here, a model
    that copied that line perfectly was flagged for inventing it, which is
    what happened to the 2026-08-25 affordability answer's 935. Harvesting the
    plan's numbers wholesale is also drift-proof: a future plan line carrying
    a new computed figure is legitimate the day it is added."""
    nums: set[int] = set()

    for tok in re.findall(r"\d[\d,]*", section_plan or ""):
        try:
            nums.add(int(tok.replace(",", "")))
        except ValueError:
            pass

    def add(v) -> None:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.add(int(v))

    for s in evidence.get("selection") or []:
        add(s.get("count"))
        add(s.get("count_unfiltered"))
        add(s.get("sub_generic"))
        add(s.get("sub_coded"))
        for sc in s.get("sub_counts") or []:
            add(sc.get("count"))
    for g in evidence.get("group_counts") or []:
        add(g.get("count_unique_responses"))
    for lc in evidence.get("location_counts") or []:
        add(lc.get("count"))
    for u in evidence.get("uncovered_categories") or []:
        add(u.get("count"))
    for key in ("location_denominator", "actionability_denominator",
                "event_denominator", "time_denominator",
                "demographic_denominator"):
        d = evidence.get(key) or {}
        vals = [v for v in d.values() if isinstance(v, int)]
        for v in vals:
            add(v)
        for a in vals:
            for b in vals:
                if a > b:
                    nums.add(a - b)
    cov = evidence.get("scope_coverage") or {}
    add(cov.get("covered"))
    add(cov.get("scope_total"))
    if isinstance(cov.get("covered"), int) and isinstance(cov.get("scope_total"), int):
        nums.add(cov["scope_total"] - cov["covered"])
    add(evidence.get("n_unique_responses"))
    for lc in lex_counts or []:
        add(lc.get("mentions_total"))
        for v in (lc.get("mentions_by_question") or {}).values():
            add(v)
    for v in (question_totals or {}).values():
        add(v)
    return nums


# claim-carrying number forms: bold spans, and "N responses/respondents/..."
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
# Citations that land INSIDE a bold span. The model routinely bolds a whole
# finding sentence, ending it with its citation — "**…burdens for tenants
# [18].**" — and the bold-number scan then read 18 as a stated count and
# flagged it as untraceable. Six of the eight residual violations on the
# 2026-08-17 affordability answer were citation numbers, not claims. Stripped
# before any number is read out of a bold span.
_CITE_SPAN_RE = re.compile(r"\[\d{1,4}(?:\s*,\s*\d{1,4})*\]")
# A stated count is a FREE-STANDING integer. Two things that look like one to
# a naive \d+ scan, and shipped as false positives on the 2026-08-25
# financial-complaints answer:
#   * a label id — "(5_043)" splits on the underscore into "043" -> 43;
#   * a percentage — "(25%)" yields 25, though 25% was the correct computed
#     coverage figure and percentages have their own check below.
# The lookarounds refuse both: a digit run glued to a word character (either
# side) or followed by "%" is not a claim.
_BOLD_NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]{1,6})(?![\w%])")
_UNIT_RE = re.compile(
    r"\b([\d,]{2,7})\s+(?:responses|respondents|coded|mentions|members)\b",
    re.IGNORECASE)
_QUOTE_RE = re.compile(r"[“\"]([^”\"]{4,400})[”\"]\s*(\[[\d,\s\]\[]*\d\])?")
_CITES_RE = re.compile(r"\d+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'")
                  .replace("“", '"').replace("”", '"')).strip().lower()


def find_violations(answer_body: str, evidence: dict, lex_counts: list[dict],
                    question_totals: dict[str, int],
                    section_plan: str = "") -> list[dict]:
    """Deterministic checks; every entry is a statement the data does not
    support. Percentages and small prose numbers are deliberately ignored.

    `section_plan` must be the same plan text the answer was written against —
    its numbers are computed in code and are therefore legitimate to state."""
    violations: list[dict] = []
    legit = legit_numbers(evidence, lex_counts, question_totals, section_plan)

    stated: set[int] = set()
    for m in _BOLD_RE.finditer(answer_body):
        for n in _BOLD_NUM_RE.findall(_CITE_SPAN_RE.sub(" ", m.group(1))):
            try:
                stated.add(int(n.replace(",", "")))
            except ValueError:
                pass
    for m in _UNIT_RE.finditer(answer_body):
        try:
            stated.add(int(m.group(1).replace(",", "")))
        except ValueError:
            pass
    for n in sorted(stated):
        if n >= MIN_CHECKED_NUMBER and n not in legit:
            violations.append({
                "kind": "number",
                "value": n,
                "detail": f"{n} does not trace to any computed count",
            })

    # percent discipline: percentages, like counts, may only be COPIED — the
    # only computed one is scope coverage. A re-rounded 19% where the data
    # says 20% is a real shipped defect this catches.
    cov = evidence.get("scope_coverage") or {}
    legit_pct: set[int] = set()
    if cov.get("scope_total"):
        legit_pct.add(round(100 * cov["covered"] / cov["scope_total"]))
    for m in _BOLD_RE.finditer(answer_body):
        for p in re.findall(r"(\d{1,3})\s*%", m.group(1)):
            if int(p) not in legit_pct:
                violations.append({
                    "kind": "percent",
                    "value": f"{p}%",
                    "detail": f"{p}% is not the computed coverage figure"
                              + (f" ({sorted(legit_pct)[0]}%)" if legit_pct
                                 else " (no percentage was computed)"),
                })

    by_n = {q["n"]: q for q in evidence.get("quotes") or []}
    for m in _QUOTE_RE.finditer(answer_body):
        span, cite = m.group(1), m.group(2)
        if not cite:
            continue
        cited = [int(x) for x in _CITES_RE.findall(cite)]
        sources = [by_n[c] for c in cited if c in by_n]
        if not sources:
            continue  # invalid citation numbers are already flagged upstream
        # ellipsis-tolerant: every quoted piece must appear in SOME cited source
        pieces = [p for p in re.split(r"…|\.\.\.", _norm(span)) if len(p) > 3]
        ok = all(any(p in _norm(s["text"]) for s in sources) for p in pieces)
        if not ok:
            violations.append({
                "kind": "quote",
                "value": span[:120],
                "detail": (f"quoted wording not found in cited source"
                           f"{'s' if len(sources) > 1 else ''} "
                           f"[{', '.join(str(c) for c in cited)}]"),
            })

    violations += _placement_violations(answer_body, evidence, by_n)
    return violations


def _tokens(name: str) -> set[str]:
    stop = {"and", "the", "of", "to", "in", "for", "a", "on", "with",
            "general", "specific", "other"}
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w
            for w in re.findall(r"[a-z]+", name.lower()) if w not in stop}


# The heading group is `[^\n]+`, NOT `.+`: re.S makes `.` match newlines, so a
# greedy `.+` ran the heading past its own line and swallowed the rest of the
# document — the 2026-08-17 affordability answer has 9 "### " sections and this
# pattern found 1, reporting a structure defect against a correct answer. re.S
# is still required for the BODY group, which does span lines.
_SECTION_RE = re.compile(r"^### ([^\n]+)\n(.*?)(?=^### |\Z)", re.M | re.S)


# A sub-theme bullet inside a category-grain section: "- Name (140 responses):".
# The trailing "(N responses)" is what distinguishes a sub-theme bullet from an
# ordinary prose bullet, which must NOT capture a region — "…small business
# rent (13) to…" is prose and is deliberately not matched.
_SUB_BULLET_RE = re.compile(
    r"^[-*]\s+([^(:\n]{3,80}?)\s*\(\d[\d,]*\s+responses?\)", re.M)


def _match_sub(text_tokens: set[str],
               subs_by_name: dict[str, tuple[str, str]]
               ) -> tuple[str, str] | None:
    """Best-matching sub-theme by token overlap, or None below the floor."""
    best, best_score = None, 0.5
    for name, val in subs_by_name.items():
        nt = _tokens(name)
        if not text_tokens or not nt:
            continue
        score = len(text_tokens & nt) / len(text_tokens | nt)
        if score > best_score:
            best, best_score = val, score
    return best


def _sub_theme_regions(body: str, subs_by_name: dict[str, tuple[str, str]]
                       ) -> list[tuple[tuple[str, str], str]]:
    """The (sub-theme, text) spans of a CATEGORY-grain section — one per
    sub-theme bullet, each running to the next bullet. Empty when the section
    names no sub-theme bullets, which is the sub-theme-grain layout."""
    hits = list(_SUB_BULLET_RE.finditer(body))
    out: list[tuple[tuple[str, str], str]] = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        sub = _match_sub(_tokens(m.group(1)), subs_by_name)
        if sub:
            out.append((sub, body[m.start():end]))
    return out


def _placement_violations(answer_body: str, evidence: dict,
                          by_n: dict[int, dict]) -> list[dict]:
    """A quote must be cited under the sub-theme it is coded to. Quotes arrive
    in the prompt grouped under sub-theme headers, so this is the guard that
    makes "cite it where it is listed" unloseable. Checked narrowly: only
    quotes from the SAME category as the matched sub-theme — cross-category
    citation in an intro sentence is legitimate.

    The unit of attribution is the narrowest labelled span containing the
    citation, because "### " sections are not always sub-themes.
    render_section_plan switches grain on the number of selected categories
    (>4 -> one section per CATEGORY, each holding a bullet per sub-theme).
    Matching a category section's heading against sub-theme names read the
    whole section as one sub-theme and flagged every bullet belonging to the
    others: on the 2026-08-25 financial-complaints answer, "General Cost of
    Living" scored 0.667 against the sub-theme "General Cost of Living and
    Inflation" (its own "General" is a stop word) and produced seven false
    positives — every correctly-placed quote in the section's other bullets.
    So within an attributed section the bullets win, and the whole section is
    used only when it names no sub-theme of its own."""
    subs_by_name: dict[str, tuple[str, str]] = {}   # name -> (sid, lid)
    for s in evidence.get("selection") or []:
        for sc in s.get("sub_counts") or []:
            subs_by_name[sc["name"]] = (sc["sub_label_id"], s["label_id"])
    if not subs_by_name:
        return []
    out: list[dict] = []
    for m in _SECTION_RE.finditer(answer_body):
        heading, body = m.group(1), m.group(2)
        # A heading that matches no sub-theme leaves the section unchecked,
        # exactly as before. Bullets REFINE an attribution the heading already
        # made; they never create one. That keeps this change incapable of
        # raising a warning today's code does not, which matters because the
        # match is token Jaccard over two names — the metric rulings.py
        # falsified over 60,461 real pairs, where known-distinct sub-themes
        # ("Lack of Law Enforcement and Prosecution" vs "Lack of Accountability
        # and Enforcement") score like known-same ones. Widening what it
        # attributes would buy coverage with exactly that unreliability.
        sub = _match_sub(_tokens(heading), subs_by_name)
        if sub is None:
            continue
        regions = _sub_theme_regions(body, subs_by_name) or [(sub, body)]
        for (sid, lid), text in regions:
            for cn in {int(x) for x in re.findall(r"\[(\d{1,4})\]", text)}:
                q = by_n.get(cn)
                if (q and q.get("label_id") == lid and q.get("subs")
                        and sid not in q["subs"]):
                    out.append({
                        "kind": "quote_placement",
                        "value": f"[{cn}]",
                        "detail": (f"quote [{cn}] is coded to a different "
                                   f"sub-theme than the section citing it"),
                    })
    return out


_PLAN_COUNT_RE = re.compile(r"^\d+\.\s.*?(\d[\d,]*)\s+responses", re.M)


def plan_structure_violations(answer_body: str, section_plan: str) -> list[dict]:
    """Deterministic post-repair check that the SECTION PLAN survived: one
    section per plan line, in order, each stating its plan count early. The
    repairer is INSTRUCTED to preserve the plan; this verifies it did."""
    counts = [int(c.replace(",", ""))
              for c in _PLAN_COUNT_RE.findall(section_plan or "")]
    if not counts:
        return []
    sections = _SECTION_RE.findall(answer_body)
    out: list[dict] = []
    if len(sections) < len(counts):
        out.append({"kind": "structure",
                    "value": f"{len(sections)} sections",
                    "detail": f"plan has {len(counts)} lines but the answer "
                              f"has {len(sections)} sections"})
        return out
    for i, cnt in enumerate(counts):
        head, body = sections[i]
        if str(cnt) not in head + body[:300]:
            out.append({"kind": "structure",
                        "value": f"section {i + 1} “{head[:40]}”",
                        "detail": f"plan count {cnt} missing from the "
                                  f"section's opening"})
    return out
