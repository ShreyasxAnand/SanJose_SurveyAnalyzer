"""Thin model-client interface for taxonomy induction.

One protocol, one production implementation (Gemini via raw REST — no SDK
dependency). Swap models by passing a different ModelClient to the pipeline;
nothing outside this module knows which vendor is behind it.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
# 6 attempts with exponential backoff ≈ 62s cumulative — a production run is
# hundreds of calls across 8 workers, so 429 bursts are the expected case,
# not an anomaly.
MAX_ATTEMPTS = 6

# The workhorse for anything whose call count scales with the corpus —
# induction MAP/ASSIGN/DEDUP and labeling. Hundreds to thousands of calls.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# The answer-writing (SYNTH) model. Kept as a separate constant because
# synthesis is one call per analyst question while every other stage scales
# with the corpus, so this is the one place a pricier model could be afforded.
#
# Currently the same as DEFAULT_MODEL, by decision on cost (2026-07-29).
# gemini-3.6-flash was tried here and reverted: it writes better answers —
# it narrates the computed counts flash-lite sometimes drops — but it bills
# thinking tokens at the output rate, ~8x the visible output (552 thoughts vs
# 69 answer tokens on an identical prompt), taking an ask from ~$0.011 to
# ~$0.038 and ~6x longer. To try it again, set this to "gemini-3.6-flash" or
# pass --synth-model / GEMINI_SYNTH_MODEL; nothing else needs to change.
DEFAULT_SYNTH_MODEL = DEFAULT_MODEL

# Published USD per 1M tokens, (input, output), checked against
# ai.google.dev/gemini-api/docs/pricing on 2026-07-29. The output rate
# INCLUDES thinking tokens — which is why complete() folds thoughtsTokenCount
# into output_tokens. A model absent from this table prices to None rather
# than to a guess: an unpriced run reports tokens and says the cost is
# unknown, which is recoverable; a confidently wrong dollar figure is not.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (1.50, 7.50),
}


def price_usd(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    rate = PRICES_PER_MTOK.get(model_id)
    if rate is None:
        return None
    return input_tokens / 1_000_000 * rate[0] + output_tokens / 1_000_000 * rate[1]


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read the repo-root .env into a dict. Deliberately does NOT write to
    os.environ — callers use it as a fallback, so a real environment variable
    always wins over the file.

    Kept dependency-free to match the rest of this module. Handles `KEY=value`,
    an optional `export ` prefix, `#` comments, and surrounding quotes. Reads as
    utf-8-sig because editors and PowerShell's `Out-File` on this platform write
    a BOM, which would otherwise turn the first key into "\\ufeffGEMINI_API_KEY".
    Unparseable lines are skipped rather than raising — a malformed .env should
    not crash an induction run that has a real env var set."""
    env_path = path or REPO_ROOT / ".env"
    values: dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Explicit argument, then environment, then repo-root .env."""
    for candidate in (
        explicit,
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    dotenv = load_dotenv()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        candidate = dotenv.get(name)
        if candidate and candidate.strip():
            # stripped: a trailing newline or space in the file would otherwise
            # go straight into an HTTP header and fail with an opaque error
            return candidate.strip()
    return None


@dataclass
class Usage:
    input_tokens: int = 0
    # billed output — thinking tokens included, because that is how the output
    # rate is charged
    output_tokens: int = 0
    # the thinking share of output_tokens, tracked separately for reporting
    # only. Never subtract it from output_tokens: it is billed, just invisible.
    thinking_tokens: int = 0
    calls: int = 0
    # `x += y` is a read-modify-write, not an atomic operation, so concurrent
    # callers silently lose increments and the manifest under-reports spend.
    # Excluded from repr/eq so Usage still compares and prints as plain data.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False)

    def add(self, input_tokens: int, output_tokens: int,
            thinking_tokens: int = 0) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.thinking_tokens += thinking_tokens
            self.calls += 1

    def cost_usd(self, price_in_per_mtok: float, price_out_per_mtok: float) -> float:
        return (
            self.input_tokens / 1_000_000 * price_in_per_mtok
            + self.output_tokens / 1_000_000 * price_out_per_mtok
        )


class ModelClient(Protocol):
    model_id: str
    usage: Usage

    def complete(self, system: str, user: str) -> str:
        """Return the model's text output for one system+user exchange."""
        ...


class GeminiClient:
    """Gemini generateContent over plain HTTPS. Temperature 0, JSON output.

    Key comes from GEMINI_API_KEY (or GOOGLE_API_KEY), looked up in the
    environment first and then in the repo-root .env — which is gitignored and
    gitignored, so it never reaches a commit or a transcript. Model id is a
    plain string so new releases need no code change.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 16384,
        timeout_s: int = 240,
        seed: int | None = 7,
        min_interval_s: float = 0.1,
    ) -> None:
        # Pinned to a concrete version, not a "-latest" alias: the manifest
        # records model_id so a run can be reproduced, which an alias silently
        # breaks when it moves. Override per-run with --model or GEMINI_MODEL.
        self.model_id = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.api_key = resolve_api_key(api_key)
        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the "
                f"environment, or put it in {REPO_ROOT / '.env'} as "
                "GEMINI_API_KEY=your-key-here"
            )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        # Temperature 0 alone does NOT make Gemini deterministic — the API
        # documents seed as the reproducibility knob. Even with it, Google
        # only offers best-effort determinism, so this narrows run-to-run
        # drift rather than eliminating it. 7 to match the pipeline's other
        # fixed seeds; None omits the field entirely.
        self.seed = seed
        # Instance-level rate limiter — effectively global, since every
        # script run shares one client across its worker threads. 0.1s means
        # at most ~10 requests/s regardless of worker count; 0 disables.
        self.min_interval_s = min_interval_s
        self._rl_lock = threading.Lock()
        self._next_ok = 0.0
        self.usage = Usage()

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._rl_lock:
            now = time.monotonic()
            wait = self._next_ok - now
            self._next_ok = max(now, self._next_ok) + self.min_interval_s
        if wait > 0:
            time.sleep(wait)

    def _backoff_delay(self, attempt: int, http_error: urllib.error.HTTPError | None = None) -> float:
        """Exponential backoff with jitter; a 429's Retry-After header, when
        present, overrides the exponential schedule."""
        retry_after = None
        if http_error is not None and http_error.code == 429:
            try:
                retry_after = float(http_error.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
        base = retry_after if retry_after else min(60, 2 ** attempt)
        return base * (1 + random.uniform(0.0, 0.25))

    def _url(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )

    def complete(self, system: str, user: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                **({"seed": self.seed} if self.seed is not None else {}),
            },
        }
        payload = self._post_with_retries(body)

        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {json.dumps(payload)[:500]}")
        cand = candidates[0]
        finish = cand.get("finishReason", "")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if finish == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini hit maxOutputTokens mid-response; raise --max-output-tokens "
                "or lower --chunk-size."
            )
        if not text.strip():
            raise RuntimeError(f"Gemini returned empty text (finishReason={finish}).")

        meta = payload.get("usageMetadata") or {}
        thoughts = meta.get("thoughtsTokenCount", 0) or 0
        out_tokens = meta.get("candidatesTokenCount", 0) + thoughts
        self.usage.add(meta.get("promptTokenCount", 0), out_tokens, thoughts)
        return text

    def _post_with_retries(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            req = urllib.request.Request(
                self._url(),
                data=data,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                if e.code in RETRYABLE_HTTP and attempt < MAX_ATTEMPTS:
                    last_err = e
                    time.sleep(self._backoff_delay(attempt, e))
                    continue
                raise RuntimeError(f"Gemini HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MAX_ATTEMPTS:
                    last_err = e
                    time.sleep(self._backoff_delay(attempt))
                    continue
                raise RuntimeError(f"Gemini request failed after {MAX_ATTEMPTS} attempts: {e}") from e
        raise RuntimeError(f"Gemini request failed: {last_err}")
