"""Generate docs/PROMPTS.md from the prompt strings themselves.

Why generated: a hand-written prompt reference drifts from the code, and the
drift is invisible until a review reasons from the doc and reaches a wrong
conclusion. That happened twice — a doc showed the ROUTE summary as bare
category lines (hiding where the lexicon and Locations lists render) and put
the section-plan merge rule under the wrong grain. Both cost a review round
chasing a defect that did not exist.

The prompt strings are already the source of truth for cache invalidation
(router.ask_prompt_hash, labeling.prompt_hash, ...). This makes them the source
of truth for the documentation too, so the two cannot disagree.

    python -m scripts.prompt_reference           # write docs/PROMPTS.md
    python -m scripts.prompt_reference --check   # exit 1 if stale (CI/test)

Adding a prompt: add it to STAGES below. The rendered doc, its line references
and its hash stamps all follow automatically.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "docs" / "PROMPTS.md"

# (heading, module, blurb, [(constant, role)])
STAGES: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    ("Stage 1 — Lexicon (optional keyword dictionary)", "app.lexicon",
     "One call. Candidate terms are extracted deterministically (frequency, "
     "bigrams, capitalisation); the model only groups them.",
     [("GROUP_SYSTEM", "system"), ("GROUP_USER", "user")]),

    ("Stage 2 — Induction (taxonomy)", "app.induction",
     "Five prompts: map → exact-name merge → sort → dedup → "
     "cross-theme. MAP runs once per disjoint chunk; evidence is cited by "
     "response NUMBER and resolved back to response_key in code, so a quote "
     "cannot be hallucinated.",
     [("MAP_SYSTEM", "system"), ("MAP_USER", "user"),
      ("VOCAB_SYSTEM", "system"), ("VOCAB_USER", "user"),
      ("ASSIGN_BATCH_SYSTEM", "system"), ("ASSIGN_BATCH_USER", "user"),
      ("DEDUP_SYSTEM", "system"), ("DEDUP_USER", "user"),
      ("CROSS_SYSTEM", "system"), ("CROSS_USER", "user")]),

    ("Stage 3 — Labeling (100% of responses)", "app.labeling",
     "One call per batch of unique responses. Output keys are abbreviated to "
     "cut output tokens; the parser still accepts the verbose spellings. "
     "example_id is a REAL id from the taxonomy in play.",
     [("LABEL_SYSTEM", "system"), ("LABEL_USER", "user")]),

    ("Stage 4 — Sub-themes", "app.subthemes",
     "SUBMAP reuses induction's parser verbatim; consolidation reuses "
     "DEDUP with the category standing in as the theme. SUBREVIEW is the only "
     "auto-applied review in the system.",
     [("SUBMAP_SYSTEM", "system"), ("SUBMAP_USER", "user"),
      ("SUBLABEL_SYSTEM", "system"), ("SUBLABEL_USER", "user"),
      ("SUBREVIEW_SYSTEM", "system"), ("SUBREVIEW_USER", "user")]),

    ("Stage 4b — Locations", "app.locations",
     "One call over the most frequent verbatim place spans the labeling pass "
     "extracted.",
     [("GROUP_SYSTEM", "system"), ("GROUP_USER", "user")]),

    ("Stage 5 — Ask", "app.router",
     "ROUTE picks categories, route and filters; RATIFY is an add-only "
     "completeness net over a code-computed candidate-miss set; SYNTH writes "
     "the answer. The three filter blocks are injected ONLY when the artifacts "
     "carry that field — a filter the data cannot honour is never "
     "advertised.",
     [("ROUTE_SYSTEM", "system"), ("ROUTE_USER", "user"),
      ("ACTIONABILITY_BLOCK", "conditional block"),
      ("EVENT_BLOCK", "conditional block"),
      ("TIME_BLOCK", "conditional block"),
      ("DEMOGRAPHIC_NOTE", "conditional block"),
      ("RATIFY_SYSTEM", "system"), ("RATIFY_USER", "user"),
      ("SYNTH_SYSTEM", "system"), ("SYNTH_USER", "user")]),

    ("Stage 5b — Sameness rulings (taxonomy-build time)", "app.rulings",
     "Decides once per taxonomy version whether two categories or sub-themes "
     "name the same idea. Similarity NOMINATES, this prompt RULES, plan build "
     "reads the store. No ruling and a failed call both mean NO fusion.",
     [("RULING_SYSTEM", "system"), ("RULING_USER", "user")]),

    # Verification has no stage here on purpose: app.verify is pure code. Its
    # repair prompt was removed 2026-08-25, so the answer pipeline's last word
    # is SYNTH — nothing rewrites what the model wrote.
]

# Prompt surface that is BUILT, not templated. Named here so the doc cannot
# imply the templates are the whole story — this is exactly what the first
# hand-written version got wrong.
BUILT_SURFACE = [
    ("app.summary", "render_summary",
     "Produces the entire `{summary}` substitution for ROUTE_SYSTEM: the "
     "per-question category lines AND the lexicon concept list AND the "
     "Locations list. ROUTE's rules refer to these as \"(if shown)\"; both "
     "sections are conditional on the corresponding artifact existing."),
    ("app.router", "render_section_plan",
     "Produces the SECTION PLAN inside SYNTH_USER. Code decides the answer's "
     "structure from full-coverage counts; the model only narrates. Two "
     "grains, chosen by how many categories the router selected."),
    ("app.router", "render_counts_block",
     "Produces the COMPUTED COUNTS block inside SYNTH_USER — the only numbers "
     "the answer may state. Sub-theme lines are indented under a single "
     "header and never repeat the category name, so no line but the CATEGORY "
     "TOTAL reads as a statement about the category."),
    ("app.router", "build_synth_prompts",
     "Appends conditional guidance sentences to ROUTE_GUIDANCE for "
     "multi-question evidence and each active filter."),
    ("app.subthemes", "review_subthemes",
     "Builds the \"PAIRS TO RULE ON\" block inside SUBREVIEW_USER from "
     "name-similarity and member-overlap candidates, with sample member "
     "responses per side."),
]

HASHES = [
    ("app.induction", "prompt_hash", "induction run ids"),
    ("app.labeling", "prompt_hash", "labels run ids"),
    ("app.subthemes", "prompt_hash", "sub-theme run ids"),
    ("app.router", "ask_prompt_hash", "ask cache key"),
    ("app.verify", "verify_logic_hash", "ask cache key"),
    ("app.rulings", "prompt_hash", "sameness-ruling pair keys"),
]

RETRY_NUDGE = ("\\nYour previous output was not valid JSON. "
               "Return ONLY the JSON object.")


def _line_of(module, const: str) -> int | None:
    """Line number of a module-level assignment, for a clickable reference."""
    try:
        src = inspect.getsource(module)
    except (OSError, TypeError):
        return None
    m = re.search(rf"^{re.escape(const)}\s*(?::[^=]+)?=", src, re.M)
    return src[:m.start()].count("\n") + 1 if m else None


def _rel(module) -> str:
    return Path(module.__file__).resolve().relative_to(REPO_ROOT).as_posix()


def render() -> str:
    out: list[str] = []
    w = out.append

    w("# Prompt reference")
    w("")
    w("**Generated file — do not edit.** Produced from the prompt strings "
      "themselves by `backend/scripts/prompt_reference.py`; "
      "`test_prompt_reference.py` fails if it drifts. Edit the prompt in its "
      "module and regenerate:")
    w("")
    w("```")
    w("python -m scripts.prompt_reference        # from backend/")
    w("```")
    w("")
    w("Every prompt is a `system` + `user` pair sent through "
      "`ModelClient.complete(system, user)`.")
    w("")
    w("Two conventions apply throughout:")
    w("")
    w("- **`{dataset_context}`** — leads every system prompt. "
      "Rendered by `induction.context_block()` as "
      "`\"Survey context:\\n{description}\\n\\n\"`, or the empty string when "
      "no description exists.")
    w(f"- **The JSON retry nudge** — on unparseable output every call retries "
      f"once with `\"{RETRY_NUDGE}\"` appended to the system prompt.")
    w("")
    w("Braces appear as they do in source: `{{` / `}}` are literal braces "
      "surviving `.format()`; single braces are substitutions.")
    w("")

    # ---- call map
    w("## Call map")
    w("")
    w("| Stage | Prompt | Location |")
    w("|---|---|---|")
    for heading, modname, _blurb, consts in STAGES:
        module = importlib.import_module(modname)
        stage = heading.split(" — ")[0]
        for const, role in consts:
            if not role.endswith("system") and role != "conditional block":
                continue
            ln = _line_of(module, const)
            where = f"`{_rel(module)}:{ln}`" if ln else f"`{_rel(module)}`"
            w(f"| {stage} | `{const.rsplit('_', 1)[0] if role == 'system' else const}` "
              f"| {where} |")
    w("")

    # ---- built surface
    w("## Prompt surface that is built, not templated")
    w("")
    w("Grepping for `_SYSTEM` constants finds the templates and misses these. "
      "Each produces substantial prompt text at run time.")
    w("")
    for modname, func, note in BUILT_SURFACE:
        module = importlib.import_module(modname)
        fn = getattr(module, func)
        ln = inspect.getsourcelines(fn)[1]
        w(f"- **`{func}()`** (`{_rel(module)}:{ln}`) — {note}")
    w("")

    # ---- the prompts
    for heading, modname, blurb, consts in STAGES:
        module = importlib.import_module(modname)
        w(f"## {heading}")
        w("")
        w(blurb)
        w("")
        for const, role in consts:
            value = getattr(module, const)
            ln = _line_of(module, const)
            where = f"{_rel(module)}:{ln}" if ln else _rel(module)
            w(f"### `{const}` ({role})")
            w("")
            w(f"`{where}`")
            w("")
            w("```")
            w(value.rstrip("\n"))
            w("```")
            w("")

    # ---- route guidance table
    router = importlib.import_module("app.router")
    w("## `ROUTE_GUIDANCE` — one line, selected by route")
    w("")
    w(f"`{_rel(router)}:{_line_of(router, 'ROUTE_GUIDANCE')}`")
    w("")
    w("| Route | Guidance |")
    w("|---|---|")
    for route, text in router.ROUTE_GUIDANCE.items():
        w(f"| `{route}` | {text} |")
    w("")

    # ---- hashes
    w("## Version stamps")
    w("")
    w("Each hash covers its own prompts, so editing one below changes the run "
      "id or invalidates the cache automatically.")
    w("")
    w("| Function | Covers | Current value |")
    w("|---|---|---|")
    for modname, func, covers in HASHES:
        module = importlib.import_module(modname)
        w(f"| `{modname.split('.')[-1]}.{func}()` | {covers} | "
          f"`{getattr(module, func)()}` |")
    w("")
    w("`lexicon` and `locations` prompts are not covered by any hash.")
    w("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed doc is stale")
    args = ap.parse_args()

    text = render()
    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != text:
            print(f"{OUT_PATH.relative_to(REPO_ROOT)} is STALE — regenerate with "
                  "`python -m scripts.prompt_reference`", file=sys.stderr)
            return 1
        print(f"{OUT_PATH.relative_to(REPO_ROOT)} is up to date")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
