"""Thin model-client interface for every AI stage.

One protocol, one production implementation: Gemini on Vertex AI through the
google-genai SDK, authenticated with Application Default Credentials (ADC).
There are no API keys anywhere — locally, `gcloud auth application-default
login` puts credentials in the environment; in production, the compute
resource's service account does. Swap models by passing a different
ModelClient to the pipeline; nothing outside this module knows which vendor
is behind it.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
from google import genai
from google.auth import exceptions as gauth_exceptions
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# The SDK logs an "automatic function calling is not recommended" advisory on
# every generate_content call. We never use function calling, and a labeling
# run makes thousands of calls — that line would drown the job logs.
import logging as _logging
_logging.getLogger("google_genai.models").setLevel(_logging.ERROR)

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
# 6 attempts with exponential backoff ≈ 62s cumulative — a production run is
# hundreds of calls across 8 workers, so 429 bursts are the expected case,
# not an anomaly.
MAX_ATTEMPTS = 6
# Past this, a caller is sitting in front of a screen wondering whether the app
# has frozen. Retries and slow calls were previously silent, so "slow" and
# "hung" looked identical from outside — the whole point of these warnings.
SLOW_CALL_SECONDS = 8.0


def _warn(message: str) -> None:
    """One line to stderr. Not `logging`: these are for whoever is watching the
    uvicorn console or a pipeline stage's log.txt, and both capture stderr
    verbatim already."""
    print(f"  [llm] {message}", file=sys.stderr, flush=True)

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

# Published USD per 1M tokens, (input, output), checked against the Vertex AI
# pricing page on 2026-07-29. The output rate INCLUDES thinking tokens —
# which is why complete() folds thoughts_token_count into output_tokens. A
# model absent from this table prices to None rather than to a guess: an
# unpriced run reports tokens and says the cost is unknown, which is
# recoverable; a confidently wrong dollar figure is not.
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
    a BOM, which would otherwise corrupt the first key's name. Unparseable
    lines are skipped rather than raising — a malformed .env should not crash
    a run that has real environment variables set."""
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


def _adc_quota_project() -> str | None:
    """The quota project recorded in the local ADC file, if any — written by
    `gcloud auth application-default login`, and the project a developer
    almost always means when they have configured nothing else."""
    if os.name == "nt":
        adc = Path(os.environ.get("APPDATA", "")) / "gcloud" / \
            "application_default_credentials.json"
    else:
        adc = Path.home() / ".config" / "gcloud" / \
            "application_default_credentials.json"
    try:
        return json.loads(adc.read_text(encoding="utf-8")).get("quota_project_id")
    except (OSError, ValueError):
        return None


def resolve_project(explicit: str | None = None) -> str | None:
    """Explicit argument, then environment, then .env, then the ADC file's
    quota project."""
    for candidate in (explicit, os.environ.get("GOOGLE_CLOUD_PROJECT")):
        if candidate and candidate.strip():
            return candidate.strip()
    candidate = load_dotenv().get("GOOGLE_CLOUD_PROJECT")
    if candidate and candidate.strip():
        return candidate.strip()
    return _adc_quota_project()


def resolve_location(explicit: str | None = None) -> str:
    """Explicit argument, then environment, then .env, then the global
    endpoint — the right default for Gemini on Vertex when nobody has a
    data-residency reason to pin a region."""
    for candidate in (explicit, os.environ.get("GOOGLE_CLOUD_LOCATION")):
        if candidate and candidate.strip():
            return candidate.strip()
    candidate = load_dotenv().get("GOOGLE_CLOUD_LOCATION")
    if candidate and candidate.strip():
        return candidate.strip()
    return "global"


_ADC_HELP = (
    "Google credentials not found or not usable. Run `gcloud auth "
    "application-default login` on this machine (or attach a service "
    "account with Vertex AI access in production), and set "
    "GOOGLE_CLOUD_PROJECT in the environment or the repo-root .env if the "
    "project cannot be inferred."
)


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
    """Gemini on Vertex AI via the google-genai SDK. Temperature 0, JSON
    output, Application Default Credentials.

    No API key exists anywhere in this path: google.auth discovers the
    credentials (gcloud ADC locally, the attached service account in
    production). The project comes from GOOGLE_CLOUD_PROJECT (environment or
    .env) or the ADC file's quota project; the location from
    GOOGLE_CLOUD_LOCATION, defaulting to the global endpoint. Model id is a
    plain string so new releases need no code change.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 16384,
        timeout_s: int = 240,
        seed: int | None = 7,
        min_interval_s: float = 0.1,
        max_attempts: int = MAX_ATTEMPTS,
        timeouts: tuple[int, ...] | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        # Pinned to a concrete version, not a "-latest" alias: the manifest
        # records model_id so a run can be reproduced, which an alias silently
        # breaks when it moves. Override per-run with --model or GEMINI_MODEL.
        self.model_id = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.project = resolve_project(project)
        self.location = resolve_location(location)
        if not self.project:
            raise RuntimeError(_ADC_HELP)
        try:
            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location)
        except Exception as exc:
            raise RuntimeError(f"{_ADC_HELP} ({exc})") from exc
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        # Batch stages want to ride out a bad patch — re-paying for a 500-row
        # labeling batch costs more than waiting. An interactive ask wants the
        # opposite: bound the worst case, because a person is watching. Both
        # knobs together set that worst case (attempts x timeout + backoff).
        self.max_attempts = max(1, max_attempts)
        # Per-attempt timeouts, shortest first. Gemini's observed failure mode
        # is a response that is SLOW, not one that never comes: a stalled ROUTE
        # call measured 169.9s and then SUCCEEDED, while the same prompt ran
        # 1.49s on ten other tries. A flat short timeout is therefore actively
        # harmful — it aborts a call that was going to work, and retrying hits
        # the same slow patch (three 30s attempts in a row is a real 502 this
        # produced). Escalating gives a dead connection a fast retry and a slow
        # one the time it actually needs.
        self.timeouts = tuple(timeouts) if timeouts else None
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

    def _timeout_for(self, attempt: int) -> int:
        """This attempt's read timeout in seconds. Without an explicit
        schedule every attempt gets timeout_s, which is what the batch stages
        want."""
        if not self.timeouts:
            return self.timeout_s
        return self.timeouts[min(attempt - 1, len(self.timeouts) - 1)]

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with jitter; a 429's Retry-After header, when
        the SDK exposes one, overrides the exponential schedule."""
        base = retry_after if retry_after else min(60, 2 ** attempt)
        return base * (1 + random.uniform(0.0, 0.25))

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        """Best-effort Retry-After from an APIError's underlying response —
        every access guarded, because the SDK does not promise the response
        object survives into the exception."""
        try:
            headers = getattr(getattr(error, "response", None), "headers", None)
            if headers is None:
                return None
            return float(headers.get("Retry-After", ""))
        except (TypeError, ValueError, AttributeError):
            return None

    def _generate(self, system: str, user: str, timeout_s: int):
        """One SDK call — the single seam between this module and google-genai,
        which is also what the offline tests monkeypatch."""
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            **({"seed": self.seed} if self.seed is not None else {}),
            # per-attempt timeout; the SDK takes milliseconds
            http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        return self._client.models.generate_content(
            model=self.model_id, contents=user, config=config)

    def complete(self, system: str, user: str) -> str:
        response = self._generate_with_retries(system, user)

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        cand = candidates[0]
        finish = str(getattr(cand, "finish_reason", "") or "")
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        text = "".join(getattr(p, "text", "") or "" for p in parts)
        if "MAX_TOKENS" in finish:
            raise RuntimeError(
                "Gemini hit max_output_tokens mid-response; raise "
                "--max-output-tokens or lower --chunk-size."
            )
        if not text.strip():
            raise RuntimeError(f"Gemini returned empty text (finish_reason={finish}).")

        meta = getattr(response, "usage_metadata", None)
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        out_tokens = (getattr(meta, "candidates_token_count", 0) or 0) + thoughts
        self.usage.add(getattr(meta, "prompt_token_count", 0) or 0,
                       out_tokens, thoughts)
        return text

    def _generate_with_retries(self, system: str, user: str):
        last_err: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            try:
                started = time.time()
                response = self._generate(system, user, self._timeout_for(attempt))
                # A retrying or merely slow call used to be indistinguishable
                # from a hang: nothing was printed, and timeout_s is 240s across
                # 6 attempts. Say so once it is past the point where a human
                # starts wondering whether the app is broken.
                elapsed = time.time() - started
                if elapsed >= SLOW_CALL_SECONDS or attempt > 1:
                    _warn(f"{self.model_id}: {elapsed:.1f}s"
                          + (f" (attempt {attempt}/{self.max_attempts})"
                             if attempt > 1 else ""))
                return response
            except gauth_exceptions.GoogleAuthError as e:
                # expired or missing credentials never fix themselves by
                # retrying — fail with the instruction that actually helps
                raise RuntimeError(f"{_ADC_HELP} ({e})") from e
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                if code in RETRYABLE_HTTP and attempt < self.max_attempts:
                    last_err = e
                    delay = self._backoff_delay(attempt, self._retry_after_seconds(e))
                    _warn(f"HTTP {code} from {self.model_id}; retrying in "
                          f"{delay:.1f}s (attempt {attempt}/{self.max_attempts})")
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Gemini HTTP {code}: {getattr(e, 'message', e)}") from e
            except (httpx.HTTPError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < self.max_attempts:
                    last_err = e
                    delay = self._backoff_delay(attempt)
                    _warn(f"{type(e).__name__} from {self.model_id} "
                          f"({e}); retrying in {delay:.1f}s "
                          f"(attempt {attempt}/{self.max_attempts})")
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Gemini request failed after {self.max_attempts} attempts: {e}"
                ) from e
        raise RuntimeError(f"Gemini request failed: {last_err}")
