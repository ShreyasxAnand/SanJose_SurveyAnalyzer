"""Layer 2: a deterministic keyword dictionary derived from the corpus.

The taxonomy answers "what is this response about". The lexicon answers "what
does this response mention" — a different question, with different failure
modes, which is the whole point of having both.

Pipeline:
  1. EXTRACT candidate terms from the corpus with no domain knowledge:
     frequency for common terms, capitalisation for proper nouns, bigrams for
     multi-word phrases. Nobody supplies a word list.
  2. GROUP them into concepts with one model call — the only model call this
     module ever makes. Surface forms that name the same thing get merged
     ("VTA", "light rail", "bus" -> public transit); caps-emphasis artifacts
     ("EVERYWHERE", "TRASH") get dropped.
  3. MATCH deterministically forever after. A regex never hallucinates, so
     lexicon counts are computed rather than estimated, and re-running costs
     nothing.

Recall is a FLOOR, not a ceiling: paraphrase ("less of a focus on cars") is
invisible to it. It supplements the taxonomy and can never replace it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .llm import ModelClient

REPO_ROOT = Path(__file__).resolve().parents[2]
LEXICON_DIR = REPO_ROOT / "data" / "lexicon"

SCHEMA_VERSION = 1
MIN_TERM_COUNT = 4          # a term seen fewer times than this is not a concept
MIN_PROPER_COUNT = 3
MAX_CANDIDATES = 220        # cap what goes into the grouping prompt

STOPWORDS = set("""
a an the and or but if then than that this these those of in on at to for with from by as is
are was were be been being it its they them their there here we our us you your i me my he she
his her have has had do does did not no nor so too very can could would should will just about
into over under more most less least all any some each other same own one two three don dont
im ive get got make made need needs want wants like really lot lots many much people person
thing things time times also because when what which who whom how why where up down out off
again further once been being having doing say said see seen go going come came take took give
gave know knew think thought feel felt look looking way ways use used using put keep keeping
let lets make makes making seems seem etc yes maybe every everyone everything something anything
nothing someone somewhere anywhere everywhere please thank thanks stop start better best good
bad worse worst new old big small high low long short right left first last next
""".split())

# Capitalised tokens that are ordinary words, not entities. Kept tiny and
# generic on purpose — anything domain-specific belongs in the grouping call,
# not hard-coded here.
NOT_ENTITIES = set("""
police homeless homelessness crime city street streets park parks trash garbage clean cleaning
downtown housing safety drug drugs traffic business businesses government county state federal
""".split())

GROUP_SYSTEM = """\
{dataset_context}You are building a keyword dictionary for a survey corpus.

You will receive candidate terms extracted automatically from the responses —
frequent words, frequent two-word phrases, and capitalised tokens that may be
names of places, agencies or organisations. The list is noisy.

Group the terms into CONCEPTS an analyst might search for.

Rules:
- Every term in a concept must be copied EXACTLY from the candidate list.
  Never invent a term that is not in the list — the dictionary is matched
  literally against the text, so an invented term matches nothing.
- Put surface forms of the same thing together, including acronyms, informal
  names and multi-word variants.
- DROP terms that are only emphasis or noise: all-caps versions of ordinary
  words, generic verbs, fragments that are half of a longer name.
- DROP a term if it is too generic to be a useful search target on its own.
- Keep concepts SPECIFIC. "public transit" is a concept; "problems" is not.
- A concept needs at least one term. Aim for 15-40 concepts.
- Give each concept a short lowercase name an analyst would recognise.
- Never estimate counts or frequencies.

Return ONLY valid JSON, exactly this shape:
{{"concepts": [{{"name": "public transit",
"terms": ["transit", "vta", "light rail", "bus"]}}]}}
"""

GROUP_USER = """Candidate terms ({n} total):
{terms}
"""


# ---------------------------------------------------------------------------
# 1. Extraction — no model, no domain knowledge
# ---------------------------------------------------------------------------


def extract_candidates(texts: list[str]) -> dict[str, list[tuple[str, int]]]:
    """Frequency for common terms, capitalisation for proper nouns, bigrams for
    phrases. Sentence-initial capitals are skipped so "Trash everywhere" does
    not make "Trash" look like a name."""
    words: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    proper: Counter[str] = Counter()

    for t in texts:
        for sent in re.split(r"(?<=[.!?/;])\s+", t):
            toks = re.findall(r"\b[A-Za-z][A-Za-z'&.-]*\b", sent)
            for tok in toks[1:]:                       # skip sentence-initial
                if tok[0].isupper() and not tok.isupper() and len(tok) > 1 \
                        and tok.lower() not in STOPWORDS and tok.lower() not in NOT_ENTITIES:
                    proper[tok] += 1
                elif tok.isupper() and 2 <= len(tok) <= 6 and tok.lower() not in STOPWORDS:
                    proper[tok] += 1                   # acronyms: VTA, SJPD, RV
        lower = [w for w in re.findall(r"[a-z][a-z'-]{2,}", t.lower()) if w not in STOPWORDS]
        words.update(lower)
        bigrams.update(f"{a} {b}" for a, b in zip(lower, lower[1:]))

    return {
        "words": [(w, n) for w, n in words.most_common(120) if n >= MIN_TERM_COUNT],
        "bigrams": [(w, n) for w, n in bigrams.most_common(80) if n >= MIN_TERM_COUNT],
        "proper": [(w, n) for w, n in proper.most_common(80) if n >= MIN_PROPER_COUNT],
    }


def render_candidates(cands: dict[str, list[tuple[str, int]]]) -> tuple[str, int]:
    lines, total = [], 0
    for kind, label in [("proper", "possible names/acronyms"), ("words", "frequent words"),
                        ("bigrams", "frequent phrases")]:
        items = cands[kind][: MAX_CANDIDATES // 3]
        total += len(items)
        lines.append(f"[{label}] " + ", ".join(f"{w} ({n})" for w, n in items))
    return "\n\n".join(lines), total


# ---------------------------------------------------------------------------
# 2. Grouping — the module's only model call
# ---------------------------------------------------------------------------


def build_lexicon(
    texts: list[str], client: ModelClient, dataset_description: str = ""
) -> tuple[dict, list[str]]:
    """Returns (lexicon, warnings). Terms the model invents are dropped: they
    would match nothing, and silently keeping them would overstate coverage."""
    from .induction import context_block, extract_json      # local: avoid cycle

    cands = extract_candidates(texts)
    rendered, n = render_candidates(cands)
    allowed = {w.lower() for kind in cands for w, _ in cands[kind]}

    system = GROUP_SYSTEM.format(dataset_context=context_block(dataset_description))
    user = GROUP_USER.format(n=n, terms=rendered)
    raw = client.complete(system, user)
    try:
        obj = extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        raw = client.complete(
            system + "\nYour previous output was not valid JSON. Return ONLY the JSON object.",
            user,
        )
        obj = extract_json(raw)

    concepts, warnings, seen_terms = [], [], set()
    for c in obj.get("concepts") or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip().lower()
        if not name:
            continue
        terms = []
        for t in c.get("terms") or []:
            t = str(t).strip().lower()
            if not t:
                continue
            if t not in allowed:
                warnings.append(f"concept {name!r}: term {t!r} not in the corpus, dropped")
            elif t in seen_terms:
                warnings.append(f"concept {name!r}: term {t!r} already used, dropped")
            else:
                seen_terms.add(t)
                terms.append(t)
        if terms:
            concepts.append({"name": name, "terms": sorted(terms)})
        else:
            warnings.append(f"concept {name!r}: no valid terms, dropped")

    lexicon = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Terms are matched literally and case-insensitively against response "
            "text. Counts from this layer are exact. Recall is a floor: paraphrase "
            "that avoids these terms is not matched."
        ),
        "concepts": sorted(concepts, key=lambda c: c["name"]),
    }
    return lexicon, warnings


# ---------------------------------------------------------------------------
# 3. Matching — deterministic, free, repeatable
# ---------------------------------------------------------------------------


def compile_concept(terms: list[str]) -> re.Pattern:
    """Word-boundary alternation, optional trailing 's' so "bus"/"buses" and
    "park"/"parks" both hit without matching "business"."""
    parts = [re.escape(t).replace(r"\ ", r"\s+") + r"e?s?" for t in sorted(terms, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


def match_responses(lexicon: dict, keys: list[str], texts: list[str]) -> dict[str, list[str]]:
    """concept name -> response_keys mentioning it."""
    out: dict[str, list[str]] = {}
    for concept in lexicon["concepts"]:
        pat = compile_concept(concept["terms"])
        out[concept["name"]] = [k for k, t in zip(keys, texts) if pat.search(t)]
    return out


def match_query(lexicon: dict, query: str) -> list[str]:
    """Concepts a free-text query plausibly refers to. Deliberately literal —
    a concept matches if its name or one of its terms appears in the query."""
    q = query.lower()
    hits = []
    for concept in lexicon["concepts"]:
        if concept["name"] in q or any(re.search(rf"\b{re.escape(t)}s?\b", q)
                                       for t in concept["terms"]):
            hits.append(concept["name"])
    return hits


def write_lexicon(lexicon: dict, manifest: dict, dataset_id: str) -> Path:
    out_dir = LEXICON_DIR / str(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lexicon.json").write_text(
        json.dumps(lexicon, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
