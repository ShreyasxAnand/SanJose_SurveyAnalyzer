"""Config resolution (.env parsing, project/location discovery) and the
retry / backoff / throttle behavior of the ADC-based Gemini client.

Everything here is offline: the SDK client construction and the generate
call are patched, so no credentials and no network are needed.
"""
from types import SimpleNamespace

import pytest

from app import llm
from google.genai import errors as genai_errors


def _write(tmp_path, text, encoding="utf-8"):
    p = tmp_path / ".env"
    p.write_text(text, encoding=encoding)
    return p


# --- .env parsing (unchanged behavior; auth.py depends on it too) ---------


def test_plain_key(tmp_path):
    assert llm.load_dotenv(_write(tmp_path, "GOOGLE_CLOUD_PROJECT=my-proj\n")) == {
        "GOOGLE_CLOUD_PROJECT": "my-proj"
    }


def test_bom_does_not_corrupt_the_first_key(tmp_path):
    # PowerShell Out-File and several editors write a BOM by default on Windows
    env = _write(tmp_path, "GOOGLE_CLOUD_PROJECT=my-proj\n", encoding="utf-8-sig")
    assert llm.load_dotenv(env) == {"GOOGLE_CLOUD_PROJECT": "my-proj"}


@pytest.mark.parametrize("raw", ['"my-proj"', "'my-proj'", "  my-proj  "])
def test_quotes_and_padding_are_stripped(tmp_path, raw):
    env = _write(tmp_path, f"GOOGLE_CLOUD_PROJECT={raw}\n")
    assert llm.load_dotenv(env)["GOOGLE_CLOUD_PROJECT"] == "my-proj"


def test_comments_blanks_and_export_prefix(tmp_path):
    env = _write(tmp_path, "\n# a comment\nexport GOOGLE_CLOUD_PROJECT=my-proj\nGARBAGE\n\n")
    assert llm.load_dotenv(env) == {"GOOGLE_CLOUD_PROJECT": "my-proj"}


def test_missing_file_is_not_an_error(tmp_path):
    assert llm.load_dotenv(tmp_path / "nope.env") == {}


# --- project / location resolution ----------------------------------------


@pytest.fixture()
def clean_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setattr(llm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(llm, "_adc_quota_project", lambda: None)
    return tmp_path


def test_environment_wins_over_dotenv(clean_env, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-env")
    _write(clean_env, "GOOGLE_CLOUD_PROJECT=from-file\n")
    assert llm.resolve_project() == "from-env"


def test_dotenv_used_when_environment_is_empty(clean_env):
    _write(clean_env, "GOOGLE_CLOUD_PROJECT=from-file\n")
    assert llm.resolve_project() == "from-file"


def test_explicit_argument_beats_everything(clean_env, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-env")
    assert llm.resolve_project("explicit") == "explicit"


def test_blank_env_var_falls_through_to_dotenv(clean_env, monkeypatch):
    # `set GOOGLE_CLOUD_PROJECT=` leaves an empty string, which must not count
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "   ")
    _write(clean_env, "GOOGLE_CLOUD_PROJECT=from-file\n")
    assert llm.resolve_project() == "from-file"


def test_adc_quota_project_is_the_last_resort(clean_env, monkeypatch):
    monkeypatch.setattr(llm, "_adc_quota_project", lambda: "from-adc")
    assert llm.resolve_project() == "from-adc"


def test_location_defaults_to_global(clean_env):
    assert llm.resolve_location() == "global"


def test_missing_project_names_the_fix(clean_env):
    with pytest.raises(RuntimeError, match="gcloud auth application-default"):
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
