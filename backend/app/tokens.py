"""Exact prompt-token counting for the pre-run estimate.

Every cost figure this app showed before a run was built on `len(text) // 4` —
a guess at the tokenizer that is wrong in a direction nobody could check.
Vertex exposes `countTokens`, which returns the real number for a real prompt,
is not billed, and does not run the model. This module is the seam that uses
it.

**Counting is sampled and pooled, not exhaustive.** A 30k-response dataset
plans hundreds of MAP chunks and labeling batches across several questions;
counting each one would turn a free instant estimate into a slow one. So:

  * each stage's system prompt is counted **exactly** — it is byte-identical
    across every chunk or batch of that stage, and it is the larger half of
    most prompts, so this is where one call buys the most;
  * a few user blocks, spread evenly across the corpus rather than taken from
    the front, are counted and fed into a **shared** tokens-per-character
    pool.

The pool is what makes this work across a multi-question dataset. Tokens per
character is a property of the tokenizer and the corpus language, not of which
question you are looking at, so what the first question measured prices the
fifth. Without it, a fixed call budget is spent entirely on question one and
every later question silently falls back to characters/4 — which is exactly
what a first cut of this module did.

Instruction text and survey prose tokenize differently (roughly 3.6 vs 4.4
characters per token on this corpus), so the two are pooled separately. Mixing
them would make the estimate depend on the ratio of system to user text in
whichever prompt happened to be sampled first.

**It degrades, never fails.** No credentials, no network, an unknown model, a
slow endpoint — any of these fall back to the old chars/4 heuristic and mark
the basis `heuristic`, because an estimate that renders is worth more than an
exception during a free planning step. The basis travels with the number so
the UI can say which kind of figure it is showing.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from . import llm

# countTokens round-trips one counter may spend in total. Each is ~0.15s, so
# this is the knob that trades estimate latency for estimate precision.
MAX_COUNT_CALLS = 24

# Calls one stage may spend. The first stage pays for the shared ratio pool
# and gets a wider sample; once the pool exists, a later stage only needs its
# own system prompt counted plus one block to confirm the ratio still holds.
FIRST_STAGE_CALLS = 4
LATER_STAGE_CALLS = 2

# The fallback when nothing at all can be counted: ~4 characters per token,
# the long-standing rule of thumb this module exists to replace.
HEURISTIC_TOKENS_PER_CHAR = 0.25

# countTokens is a small, cheap call and a person is waiting on the estimate,
# so bound it tightly rather than inheriting the 240s generate timeout.
COUNT_TIMEOUT_S = 20

# Basis values, in decreasing order of trust. Strings because they travel to
# the UI and into the estimate payload.
EXACT = "counted"
SAMPLED = "sampled"
HEURISTIC = "heuristic"


@dataclass
class TokenEstimate:
    """Input tokens for one stage's prompts, and how they were arrived at."""

    tokens: int
    # "counted"   — every prompt in this stage measured; the real number
    # "sampled"   — measured prompts extrapolated by a measured ratio
    # "heuristic" — nothing could be measured; characters/4
    basis: str
    # countTokens calls this stage actually spent
    calls: int = 0

    @property
    def is_measured(self) -> bool:
        return self.basis != HEURISTIC


class TokenCounter:
    """Counts prompt tokens for one model, with a shared budget and ratio pool.

    Build one per estimate and pass it down: both the budget and everything it
    learns are shared, so a seven-question dataset spends its calls across the
    questions and prices the last one with what the first one measured.

    The underlying SDK client is built lazily on first use, so an estimate that
    never needs counting — or an environment with no credentials — costs
    nothing and raises nothing.
    """

    def __init__(self, model_id: str | None = None,
                 max_calls: int = MAX_COUNT_CALLS) -> None:
        self.model_id = llm.resolve_model(model_id)
        self.max_calls = max(0, max_calls)
        self._client = None
        self._unavailable = False
        self._lock = threading.Lock()
        # exact counts for texts already measured, keyed by the text itself —
        # a stage's system prompt is asked for once per question and is often
        # identical across them
        self._memo: dict[str, int] = {}
        # (tokens, chars) measured so far, kept apart for instruction text and
        # survey text because they do not tokenize alike
        self._pool: dict[str, list[int]] = {"system": [0, 0], "user": [0, 0]}
        self.calls_made = 0

    # -- plumbing ---------------------------------------------------------

    def _get_client(self):
        """The raw google-genai client, or None if one cannot be built.

        Deliberately not `llm.GeminiClient`: that wrapper carries retry,
        throttling and usage accounting built for billed generate calls, none
        of which apply to a free metadata call that must fail fast.
        """
        if self._unavailable:
            return None
        if self._client is not None:
            return self._client
        try:
            project = llm.resolve_project()
            if not project:
                self._unavailable = True
                return None
            self._client = genai.Client(
                vertexai=True, project=project, location=llm.resolve_location())
        except Exception:
            # missing ADC, an unreachable metadata server, an SDK version
            # mismatch — all mean the same thing here: count nothing, estimate
            # from characters, and say so
            self._unavailable = True
            return None
        return self._client

    def _count(self, text: str, kind: str) -> int | None:
        """Exact tokens for one piece of text, or None if it cannot be
        measured (budget spent, no credentials, request failed).

        `kind` picks both the framing and the ratio pool: "system" sends the
        text as `system_instruction`, which is how `complete()` sends it and
        therefore how it will really be billed; "user" sends it as contents.
        """
        key = f"{kind}\x00{text}"
        with self._lock:
            memoed = self._memo.get(key)
            if memoed is not None:
                return memoed
            if self.calls_made >= self.max_calls:
                return None
            # reserve the slot before releasing the lock, so two threads
            # cannot both spend the last call
            self.calls_made += 1

        client = self._get_client()
        if client is None:
            return None
        try:
            config = genai_types.CountTokensConfig(
                http_options=genai_types.HttpOptions(
                    timeout=COUNT_TIMEOUT_S * 1000),
            )
            if kind == "system":
                config.system_instruction = text
                contents = " "
            else:
                contents = text or " "
            response = client.models.count_tokens(
                model=self.model_id, contents=contents, config=config)
            total = int(getattr(response, "total_tokens", 0) or 0)
        except Exception:
            # one failure is usually the whole endpoint (auth, region, model
            # id), so stop trying rather than burn the budget on retries
            self._unavailable = True
            return None
        if total <= 0:
            return None
        with self._lock:
            self._memo[key] = total
            pool = self._pool[kind]
            pool[0] += total
            pool[1] += len(text)
        return total

    def ratio(self, kind: str) -> float | None:
        """Measured tokens per character for `kind`, or None if nothing of
        that kind has been counted yet. Falls back to the other pool when one
        is empty — a measured ratio from the wrong register still beats 4
        characters per token."""
        with self._lock:
            tokens, chars = self._pool[kind]
            if chars > 0 and tokens > 0:
                return tokens / chars
            other = "user" if kind == "system" else "system"
            tokens, chars = self._pool[other]
            if chars > 0 and tokens > 0:
                return tokens / chars
        return None

    @property
    def has_measured(self) -> bool:
        with self._lock:
            return any(chars > 0 for _tokens, chars in self._pool.values())

    # -- the API callers actually use -------------------------------------

    def estimate(self, system: str, users: list[str]) -> TokenEstimate:
        """Input tokens for `len(users)` prompts that share one system prompt.

        This is the shape both stages have: induction MAP sends one system
        prompt with N chunk blocks, labeling sends one system prompt with N
        batch blocks.
        """
        if not users:
            return TokenEstimate(tokens=0, basis=EXACT, calls=0)

        spent_before = self.calls_made
        # A stage that arrives before anything has been measured pays for the
        # shared pool; later stages ride on it and need far less.
        stage_budget = (LATER_STAGE_CALLS if self.has_measured
                        else FIRST_STAGE_CALLS)

        def stage_spent() -> int:
            return self.calls_made - spent_before

        # The system prompt: one call, reused for every prompt in the stage.
        system_tokens = self._count(system, "system") if system else 0
        system_exact = system_tokens is not None

        # User blocks: count as many as the stage budget allows, exactly when
        # they all fit, otherwise a spread sample that feeds the pool.
        exact_users: dict[int, int] = {}
        room = max(0, min(stage_budget - stage_spent(),
                          self.max_calls - self.calls_made))
        if len(users) <= room:
            targets = list(range(len(users)))
        else:
            targets = _spread_indices(len(users), room)
        for i in targets:
            measured = self._count(users[i], "user")
            if measured is not None:
                exact_users[i] = measured

        user_ratio = self.ratio("user")
        if user_ratio is None:
            # nothing measurable anywhere — the system prompt is sent once per
            # prompt, so it counts once per prompt in the character total too
            total_chars = len(system) * len(users) + sum(len(u) for u in users)
            return TokenEstimate(
                tokens=round(total_chars * HEURISTIC_TOKENS_PER_CHAR),
                basis=HEURISTIC,
                calls=stage_spent(),
            )

        if not system_exact:
            system_ratio = self.ratio("system") or user_ratio
            system_tokens = round(len(system) * system_ratio)

        total = system_tokens * len(users)
        for i, text in enumerate(users):
            measured = exact_users.get(i)
            total += measured if measured is not None else round(
                len(text) * user_ratio)

        basis = EXACT if (system_exact and len(exact_users) == len(users)) \
            else SAMPLED
        return TokenEstimate(tokens=total, basis=basis, calls=stage_spent())


def heuristic_tokens(*texts: str) -> int:
    """The old chars/4 estimate, kept as one named thing so the places that
    still fall back to it are greppable."""
    return round(sum(len(t) for t in texts) * HEURISTIC_TOKENS_PER_CHAR)


def _spread_indices(count: int, n: int) -> list[int]:
    """`n` indices spread evenly across `count` items, not the first `n`.

    Which prompts get sampled matters: induction chunks are a seeded shuffle
    of the corpus but labeling batches are not, and the last batch of a run is
    short. Taking the head would calibrate the whole estimate on whichever
    responses happened to sort first.
    """
    if n <= 0 or count <= 0:
        return []
    if n >= count:
        return list(range(count))
    step = count / n
    return sorted({min(count - 1, int(i * step)) for i in range(n)})
