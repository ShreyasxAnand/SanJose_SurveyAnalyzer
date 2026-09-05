"""Vertex project/location discovery and the retry / backoff / throttle
behavior of the ADC-based Gemini client.

Everything here is offline: the SDK client construction and the generate
call are patched, so no credentials and no network are needed.

The settings half of this lives in test_config.py, which owns the
environment > config.json precedence rule. What is left here is the part
specific to the client: that a missing project is a named, actionable error
rather than an obscure SDK failure.
"""
from types import SimpleNamespace

import pytest

from app import config, llm
from google.genai import errors as genai_errors


# --- project / location resolution ----------------------------------------


@pytest.fixture()
def clean_env(tmp_path, monkeypatch):
    """Nothing configured anywhere: no environment, no config.json, no ADC."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(config, "_cache_stamp", None)
    monkeypatch.setattr(llm, "_adc_quota_project", lambda: None)
    return tmp_path


def test_location_defaults_to_global(clean_env):
    assert llm.resolve_location() == "global"


def test_no_project_anywhere_resolves_to_none(clean_env):
    assert llm.resolve_project() is None


def test_missing_project_names_the_fix(clean_env):
    """The failure a new install actually hits. The message has to say what to
    run, because the SDK's own error does not."""
    with pytest.raises(RuntimeError, match="gcloud auth application-default"):
        llm.GeminiClient()
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        llm.GeminiClient()


# --- retry / backoff / throttle (offline: the SDK seam is patched) --------


def _client(monkeypatch, **kwargs):
    monkeypatch.setattr(llm, "resolve_project", lambda explicit=None: "test-proj")
    monkeypatch.setattr(llm.genai, "Client",
                        lambda **kw: SimpleNamespace(models=None))
    kwargs.setdefault("min_interval_s", 0)  # throttle off unless the test wants it
    return llm.GeminiClient(**kwargs)


def _ok_response(thoughts: int | None = None):
    usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1,
                            thoughts_token_count=thoughts)
    return SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text='{"ok": true}')]),
            finish_reason="STOP")],
        usage_metadata=usage,
    )


def _api_error(code, retry_after=None):
    e = genai_errors.APIError(code, {"error": {"message": "err"}})
    if retry_after is not None:
        e.response = SimpleNamespace(headers={"Retry-After": str(retry_after)})
    return e


def test_thinking_tokens_are_billed_as_output_and_reported_separately(monkeypatch):
    """A thinking model's real cost is invisible in its answer. Thoughts must
    be counted INTO output_tokens (that is how they are charged) and also
    tracked on their own so the bill can be explained."""
    client = _client(monkeypatch)
    monkeypatch.setattr(client, "_generate",
                        lambda s, u, t: _ok_response(thoughts=500))
    client.complete("s", "u")
    assert client.usage.output_tokens == 501      # 1 visible + 500 thinking
    assert client.usage.thinking_tokens == 500
    assert llm.price_usd("gemini-3.6-flash", 0, client.usage.output_tokens) == \
        pytest.approx(501 / 1_000_000 * 7.50)


def test_synth_model_defaults_to_the_cheap_model(monkeypatch):
    """Answer synthesis runs on the same cheap model as everything else, by
    decision on cost (2026-07-29). This guards against an upgrade sneaking
    back in unnoticed — flipping it is fine, doing so silently is not."""
    assert llm.DEFAULT_SYNTH_MODEL == llm.DEFAULT_MODEL == "gemini-3.5-flash-lite"


def test_retry_honors_retry_after_on_429(monkeypatch):
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_generate(s, u, t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _api_error(429, retry_after=3)
        return _ok_response()

    sleeps: list[float] = []
    monkeypatch.setattr(client, "_generate", fake_generate)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    assert client.complete("s", "u") == '{"ok": true}'
    assert calls["n"] == 2
    # Retry-After (3s) overrides the exponential schedule; jitter adds ≤25%
    assert len(sleeps) == 1 and 3.0 <= sleeps[0] <= 3.75


def test_persistent_429_exhausts_all_attempts(monkeypatch):
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_generate(s, u, t):
        calls["n"] += 1
        raise _api_error(429)

    monkeypatch.setattr(client, "_generate", fake_generate)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="429"):
        client.complete("s", "u")
    assert calls["n"] == llm.MAX_ATTEMPTS == 6


def test_non_retryable_http_error_raises_immediately(monkeypatch):
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_generate(s, u, t):
        calls["n"] += 1
        raise _api_error(400)

    monkeypatch.setattr(client, "_generate", fake_generate)
    with pytest.raises(RuntimeError, match="400"):
        client.complete("s", "u")
    assert calls["n"] == 1


def test_auth_errors_never_retry_and_name_the_fix(monkeypatch):
    """Expired ADC does not fix itself: retrying five more times just delays
    the person, and the message must say what to actually run."""
    client = _client(monkeypatch)
    calls = {"n": 0}

    def fake_generate(s, u, t):
        calls["n"] += 1
        raise llm.gauth_exceptions.RefreshError("token expired")

    monkeypatch.setattr(client, "_generate", fake_generate)
    with pytest.raises(RuntimeError, match="gcloud auth application-default"):
        client.complete("s", "u")
    assert calls["n"] == 1


def test_throttle_spaces_out_requests(monkeypatch):
    client = _client(monkeypatch, min_interval_s=0.05)
    monkeypatch.setattr(client, "_generate", lambda s, u, t: _ok_response())
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    t = {"now": 100.0}
    monkeypatch.setattr(llm.time, "monotonic", lambda: t["now"])
    client.complete("s", "u")     # first call: no wait, books the next slot
    client.complete("s", "u")     # same instant: must wait one interval
    assert sleeps == [pytest.approx(0.05)]
