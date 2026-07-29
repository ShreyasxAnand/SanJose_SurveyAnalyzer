"""Key resolution: environment first, .env as fallback.

These cover the failure modes that would otherwise surface as an opaque HTTP
400 from Gemini rather than a clear local error — a BOM glued to the first key
name, quotes kept as part of the value, or a trailing newline landing in an
HTTP header.
"""
import pytest

from app import llm


def _write(tmp_path, text, encoding="utf-8"):
    p = tmp_path / ".env"
    p.write_text(text, encoding=encoding)
    return p


def test_plain_key(tmp_path):
    assert llm.load_dotenv(_write(tmp_path, "GEMINI_API_KEY=abc123\n")) == {
        "GEMINI_API_KEY": "abc123"
    }


def test_bom_does_not_corrupt_the_first_key(tmp_path):
    # PowerShell Out-File and several editors write a BOM by default on Windows
    env = _write(tmp_path, "GEMINI_API_KEY=abc123\n", encoding="utf-8-sig")
    assert llm.load_dotenv(env) == {"GEMINI_API_KEY": "abc123"}


@pytest.mark.parametrize("raw", ['"abc123"', "'abc123'", "  abc123  "])
def test_quotes_and_padding_are_stripped(tmp_path, raw):
    env = _write(tmp_path, f"GEMINI_API_KEY={raw}\n")
    assert llm.load_dotenv(env)["GEMINI_API_KEY"] == "abc123"


def test_comments_blanks_and_export_prefix(tmp_path):
    env = _write(tmp_path, "\n# a comment\nexport GEMINI_API_KEY=abc123\nGARBAGE\n\n")
    assert llm.load_dotenv(env) == {"GEMINI_API_KEY": "abc123"}


def test_missing_file_is_not_an_error(tmp_path):
    assert llm.load_dotenv(tmp_path / "nope.env") == {}


def test_environment_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    _write(tmp_path, "GEMINI_API_KEY=from-file\n")
    assert llm.resolve_api_key() == "from-env"


def test_dotenv_used_when_environment_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    _write(tmp_path, "GEMINI_API_KEY=from-file\n")
    assert llm.resolve_api_key() == "from-file"


def test_explicit_argument_beats_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert llm.resolve_api_key("explicit") == "explicit"


def test_blank_env_var_falls_through_to_dotenv(tmp_path, monkeypatch):
    # `set GEMINI_API_KEY=` leaves an empty string, which must not count
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    _write(tmp_path, "GEMINI_API_KEY=from-file\n")
    assert llm.resolve_api_key() == "from-file"


def test_error_message_names_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match=r"\.env"):
        llm.GeminiClient()


# --- retry / backoff / throttle (offline: urlopen and sleep are patched) ---

import io
import json as json_module
import urllib.error


def _client(monkeypatch, **kwargs):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    kwargs.setdefault("min_interval_s", 0)  # throttle off unless the test wants it
    return llm.GeminiClient(**kwargs)


def _ok_response():
    payload = {
        "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }

    class _Resp:
        def read(self):
            return json_module.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Resp()


def _http_error(code, headers=None):
    return urllib.error.HTTPError(
        url="https://example", code=code, msg="err",
        hdrs=headers or {}, fp=io.BytesIO(b""))


def test_retry_honors_retry_after_on_429(monkeypatch):
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, {"Retry-After": "3"})
        return _ok_response()

    sleeps: list[float] = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    assert client.complete("s", "u") == '{"ok": true}'
    assert calls["n"] == 2
    # Retry-After (3s) overrides the exponential schedule; jitter adds ≤25%
    assert len(sleeps) == 1 and 3.0 <= sleeps[0] <= 3.75


def test_persistent_429_exhausts_all_attempts(monkeypatch):
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise _http_error(429)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="429"):
        client.complete("s", "u")
    assert calls["n"] == llm.MAX_ATTEMPTS == 6


def test_non_retryable_http_error_raises_immediately(monkeypatch):
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="400"):
        client.complete("s", "u")
    assert calls["n"] == 1


def test_throttle_spaces_out_requests(monkeypatch):
    client = _client(monkeypatch, min_interval_s=0.05)
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout: _ok_response())
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    t = {"now": 100.0}
    monkeypatch.setattr(llm.time, "monotonic", lambda: t["now"])
    client.complete("s", "u")     # first call: no wait, books the next slot
    client.complete("s", "u")     # same instant: must wait one interval
    assert sleeps == [pytest.approx(0.05)]
