"""Thin model-client interface for taxonomy induction.

One protocol, one production implementation (Gemini via raw REST — no SDK
dependency). Swap models by passing a different ModelClient to the pipeline;
nothing outside this module knows which vendor is behind it.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4

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
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
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
    ) -> None:
        # Pinned to a concrete version, not a "-latest" alias: the manifest
        # records model_id so a run can be reproduced, which an alias silently
        # breaks when it moves. Override per-run with --model or GEMINI_MODEL.
        self.model_id = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
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
        self.usage = Usage()

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
        out_tokens = meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        self.usage.add(meta.get("promptTokenCount", 0), out_tokens)
        return text

    def _post_with_retries(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
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
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Gemini HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MAX_ATTEMPTS:
                    last_err = e
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Gemini request failed after {MAX_ATTEMPTS} attempts: {e}") from e
        raise RuntimeError(f"Gemini request failed: {last_err}")
