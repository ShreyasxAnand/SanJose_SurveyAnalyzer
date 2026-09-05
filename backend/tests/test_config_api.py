"""The settings endpoints: what they reveal, what they refuse, what they write.

The one rule with teeth: the passcode goes in and never comes back out. A
settings page that echoes the secret it protects leaks it to every screenshot,
every browser cache, and every person standing behind the screen.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import auth, config
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setattr(config, "_cache_stamp", None)
    # this module is about the real precedence chain, so let auth read the
    # real (temp) config rather than the conftest stub
    monkeypatch.setattr(auth, "load_config", config.load_config)
    with TestClient(app) as c:
        yield c


def _stored(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- reading ---------------------------------------------------------------


def test_get_config_reports_the_defaults_when_nothing_is_configured(client):
    body = client.get("/api/config").json()
    assert body["admin_passcode_set"] is False
    assert body["admin_passcode_source"] == ""
    assert body["default_model"] == "gemini-3.5-flash-lite"
    assert body["synth_model"] == "gemini-3.5-flash-lite"
    assert body["config_exists"] is False
    assert {p["model_id"] for p in body["prices"]} >= {
        "gemini-3.5-flash-lite", "gemini-3.6-flash"}
    assert body["calibration"]["source"] == "builtin"


def test_get_config_never_returns_the_passcode(client, tmp_path):
    client.put("/api/config", json={"admin_passcode": "hunter2"})
    res = client.get("/api/config", headers={"X-Admin-Passcode": "hunter2"})
    assert res.status_code == 200
    assert "hunter2" not in res.text
    assert res.json()["admin_passcode_set"] is True
    assert res.json()["admin_passcode_source"] == "config"


def test_get_config_is_gated_once_a_passcode_exists(client):
    client.put("/api/config", json={"admin_passcode": "hunter2"})
    # the only read in the app that is gated: it describes the lock, not the
    # data behind it
    assert client.get("/api/config").status_code == 401
    assert client.get(
        "/api/config", headers={"X-Admin-Passcode": "hunter2"}).status_code == 200


def test_env_overrides_are_disclosed(client, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    body = client.get("/api/config").json()
    # saving the model here would appear to work and change nothing; the
    # screen has to be able to say why
    assert "GEMINI_MODEL" in body["env_overrides"]
    assert body["default_model"] == "from-env"


def test_a_price_override_is_flagged_as_such(client):
    client.put("/api/config", json={"prices": [
        {"model_id": "gemini-3.5-flash-lite", "input_per_mtok": 9.0,
         "output_per_mtok": 9.0},
    ]})
    prices = {p["model_id"]: p for p in client.get("/api/config").json()["prices"]}
    assert prices["gemini-3.5-flash-lite"]["overridden"] is True
    assert prices["gemini-3.5-flash-lite"]["input_per_mtok"] == 9.0
    # a built-in row that was not overridden must not claim to be
    assert prices["gemini-3.6-flash"]["overridden"] is False


# --- writing ---------------------------------------------------------------


def test_put_sets_models_and_prices(client, tmp_path):
    res = client.put("/api/config", json={
        "default_model": "cheap-model",
        "synth_model": "fancy-model",
        "prices": [{"model_id": "cheap-model", "input_per_mtok": 0.1,
                    "output_per_mtok": 0.4}],
    })
    assert res.status_code == 200
    assert res.json()["default_model"] == "cheap-model"
    assert res.json()["synth_model"] == "fancy-model"
    stored = _stored(tmp_path / "config.json")
    assert stored["models"] == {"default": "cheap-model", "synth": "fancy-model"}
    assert stored["prices_per_mtok"]["cheap-model"] == {"input": 0.1, "output": 0.4}


def test_omitted_fields_are_left_alone(client, tmp_path):
    client.put("/api/config", json={"admin_passcode": "keep-me",
                                    "default_model": "keep-me-too"})
    headers = {"X-Admin-Passcode": "keep-me"}
    client.put("/api/config", json={"synth_model": "changed"}, headers=headers)

    stored = _stored(tmp_path / "config.json")
    assert stored["admin_passcode"] == "keep-me"
    assert stored["models"]["default"] == "keep-me-too"
    assert stored["models"]["synth"] == "changed"


def test_an_empty_passcode_clears_the_gate(client, tmp_path):
    client.put("/api/config", json={"admin_passcode": "temporary"})
    headers = {"X-Admin-Passcode": "temporary"}

    body = client.put("/api/config", json={"admin_passcode": ""},
                      headers=headers).json()
    assert body["admin_passcode_set"] is False
    # stored as null rather than "": the file should read as "not configured",
    # not as "configured to the empty string"
    assert _stored(tmp_path / "config.json")["admin_passcode"] is None
    assert client.get("/api/config").status_code == 200


def test_changing_the_passcode_requires_the_old_one(client):
    client.put("/api/config", json={"admin_passcode": "first"})
    # no header: the gate is on now, so this is a 401 and nothing changes
    assert client.put("/api/config",
                      json={"admin_passcode": "second"}).status_code == 401
    assert client.put("/api/config", json={"admin_passcode": "second"},
                      headers={"X-Admin-Passcode": "first"}).status_code == 200
    assert client.get(
        "/api/config", headers={"X-Admin-Passcode": "second"}).status_code == 200


def test_a_blank_model_is_rejected(client):
    # "" would mean "clear it", which for a model means "run nothing" — omit
    # the field to leave it unchanged instead
    res = client.put("/api/config", json={"default_model": "   "})
    assert res.status_code == 422


def test_a_negative_price_is_rejected(client, tmp_path):
    res = client.put("/api/config", json={"prices": [
        {"model_id": "m", "input_per_mtok": -1, "output_per_mtok": 1}]})
    assert res.status_code == 422
    assert not (tmp_path / "config.json").exists()


def test_duplicate_price_rows_are_rejected(client):
    res = client.put("/api/config", json={"prices": [
        {"model_id": "m", "input_per_mtok": 1, "output_per_mtok": 1},
        {"model_id": "m", "input_per_mtok": 2, "output_per_mtok": 2},
    ]})
    assert res.status_code == 422


def test_unknown_keys_in_the_file_survive_a_save(client, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"from_a_newer_build": {"x": 1}}), encoding="utf-8")
    client.put("/api/config", json={"default_model": "m"})
    assert _stored(tmp_path / "config.json")["from_a_newer_build"] == {"x": 1}


# --- recalibration ---------------------------------------------------------


def test_recalibrate_writes_a_measured_calibration(client, tmp_path, monkeypatch):
    from app import calibration

    monkeypatch.setattr(calibration, "measure", lambda **kw: {
        "source": "measured",
        "measured_utc": "2026-08-24T00-00-00Z",
        "sample": {"label_runs": 4},
        "label": {"input_tokens_per_unique": 50.0,
                  "output_tokens_per_unique": 25.0, "unique_ratio": 1.0},
    })
    body = client.post("/api/config/recalibrate").json()
    assert body["source"] == "measured"
    assert body["sample"]["label_runs"] == 4
    assert body["label_input_tokens_per_response"] == 50.0
    # persisted, so the next estimate uses it without re-measuring
    assert _stored(tmp_path / "config.json")["cost_calibration"]["source"] == "measured"


def test_recalibrate_is_admin_gated(client):
    client.put("/api/config", json={"admin_passcode": "hunter2"})
    assert client.post("/api/config/recalibrate").status_code == 401


def test_a_non_ascii_passcode_is_refused_at_save_time(client, tmp_path):
    """An HTTP header is bytes and the round trip does not preserve non-ASCII
    text, so a passcode containing one could never be typed back in. Refusing
    here beats a lockout discovered later from the wrong client."""
    res = client.put("/api/config", json={"admin_passcode": "caf\u00e9-se\u00f1or"})
    assert res.status_code == 400
    assert "ASCII" in res.json()["detail"]
    assert not (tmp_path / "config.json").exists()


def test_a_smart_quote_names_itself_in_the_error(client):
    # the usual cause is a paste out of Word or a doc, and the character is
    # invisible in a password field — so the message has to name it
    res = client.put("/api/config", json={"admin_passcode": "dont\u2019tell"})
    assert res.status_code == 400
    assert "\u2019" in res.json()["detail"]


def test_a_control_character_in_a_passcode_is_refused(client):
    res = client.put("/api/config", json={"admin_passcode": "with\ttab"})
    assert res.status_code == 400
    assert "control character" in res.json()["detail"]
