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
