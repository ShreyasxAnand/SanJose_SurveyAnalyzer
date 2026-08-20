"""Persistent ask cache — the same question against the same data pulls up
the stored answer instead of re-rolling the model.

Temperature 0 and a fixed seed make routing MOSTLY repeatable, but the
provider does not guarantee identical output for identical requests, and we
have watched the same question route differently minutes apart. For a report
someone re-runs, that is not acceptable. Every answer is already written to
disk; this module only adds recognition: "this exact request, against this
exact data state, was already answered — serve that."

Robustness rules:
  * The KEY covers everything that can change the result: normalized
    question text, dataset, scope, the analyst's actual selection and
    filters (an edited selection is a different answer), the context key
    (labels/sub-theme/locations/lexicon run state + export file stats — the
    same freshness check the in-process context cache trusts), the ask
    prompt hash, and the model ids. Any change -> different key -> miss.
  * Entries are one JSON file per key, written atomically (temp file then
    os.replace). A missing, unreadable, or schema-mismatched file is a MISS,
    never an error — the cost of a broken cache entry is one model call.
  * Nothing is ever edited in place and nothing needs manual clearing.

The cache stores full API response payloads, so a hit is byte-identical to
the original answer and carries its original run_id.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from .induction import DATA_DIR, utc_now

SCHEMA_VERSION = 1
CACHE_DIR = DATA_DIR / "answers"


# apostrophes vanish ("what's" == "whats" — accepting that contractions merge
# with real words, "we're" == "were"; no two plausible questions differ only
# by that). U+02BC is the phone-keyboard apostrophe, a *letter* to \w, so it
# needs deleting explicitly. Every other punctuation mark — and underscore,
# which \w would otherwise keep — becomes a space ("theft/vandalism" ==
# "theft vandalism", and "1.5" stays distinct from "15").
_APOSTROPHES = str.maketrans("", "", "'’ʼ")
_PUNCT_RE = re.compile(r"[^\w\s]|_")


def _norm_question(q: str) -> str:
    """Whitespace-, case- and punctuation-insensitive: 'What locations…?' and
    'what  locations…' are the same question. NFKC first, so composed and
    decomposed accents (café pasted vs typed) key identically. Nothing
    fuzzier — a wrong cache hit is worse than a missed one."""
    q = unicodedata.normalize("NFKC", q or "")
    q = _PUNCT_RE.sub(" ", q.translate(_APOSTROPHES))
    return " ".join(q.split()).casefold()


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


# Request fields the answer key NORMALIZES (so harmless variations still
# hit) — everything NOT listed here is hashed verbatim via the "extra"
# capture below, so a future filter (demographics, etc.) is part of the key
# the day it is added to the request schema. A forgotten field must cause
# cache MISSES, never a filtered ask served an unfiltered cached answer.
_NORMALIZED_FIELDS = {
    "question", "selected", "lexicon_concepts", "location_filter",
    "question_scope", "reason", "proposed_label_ids", "demographic_filter",
}


def route_key(context_key: str, question: str, scope: list[str],
              model_id: str, extra: dict | None = None) -> str:
    blob = _canonical({
        "kind": "route",
        "context": context_key,
        "question": _norm_question(question),
        "scope": sorted(str(s) for s in scope or []),
        "prompts": _ask_prompt_hash(),
        "model": model_id,
        "extra": extra or {},
    })
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def answer_key(context_key: str, request: dict, model_ids: str) -> str:
    """`request` is the client's FULL answer payload. Known list fields are
    canonicalized (sorted) so ticking the same set in a different order
    hits; `reason`/`proposed_label_ids` are excluded as non-semantic (they
    are audit metadata, not evidence choices); every OTHER field — including
    ones that do not exist yet — is hashed as-is."""
    blob = _canonical({
        "kind": "answer",
        "context": context_key,
        "question": _norm_question(str(request.get("question", ""))),
        "selected": sorted(str(c.get("label_id", ""))
                           for c in request.get("selected") or []),
        "lexicon_concepts": sorted(request.get("lexicon_concepts") or []),
        "location_filter": sorted(request.get("location_filter") or []),
        "question_scope": sorted(str(s)
                                 for s in request.get("question_scope") or []),
        # value order inside a field is meaningless — sort so ticking the
        # same set in a different order hits; empty value lists are no filter
        "demographic_filter": {
            str(f): sorted(str(v) for v in vals)
            for f, vals in sorted((request.get("demographic_filter")
                                   or {}).items()) if vals},
        "extra": {k: v for k, v in sorted(request.items())
                  if k not in _NORMALIZED_FIELDS},
        "prompts": _ask_prompt_hash(),
        "models": model_ids,
    })
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _ask_prompt_hash() -> str:
    # prompts + router-side behavior constants, plus the service layer's
    # answer-shaping fingerprint (guardrail thresholds, time regexes) and
    # the verifier's — imported lazily to avoid a cycle
    from . import ask_service, router, verify
    return (router.ask_prompt_hash() + ":" + ask_service.ask_logic_hash()
            + ":" + verify.verify_logic_hash())


def _path(dataset_id: str, key: str) -> Path:
    return CACHE_DIR / str(dataset_id) / "cache" / f"{key}.json"


def load(dataset_id: str, key: str) -> dict | None:
    """The stored payload, or None. Every failure mode is a miss."""
    p = _path(dataset_id, key)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA_VERSION:
        return None
    payload = obj.get("payload")
    return payload if isinstance(payload, dict) else None


def store(dataset_id: str, key: str, payload: dict) -> None:
    """Atomic write; best-effort — a cache that fails to store must never
    fail the answer it would have cached."""
    p = _path(dataset_id, key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"schema": SCHEMA_VERSION, "stored_utc": utc_now(),
                           "payload": payload}, f, ensure_ascii=False)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass
