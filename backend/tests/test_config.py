"""config.json: precedence, tolerance of a broken file, and safe writes.

The precedence rule (environment > config.json > built-in) is the whole
contract, so most of this file is about proving each layer yields to the one
above it and that whitespace never counts as configured.
"""
import json

import pytest

from app import auth, config, llm


@pytest.fixture()
def config_file(tmp_path, monkeypatch):
    """Point the config module at a writable temp file."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setattr(config, "_cache_stamp", None)
    return path


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- reading ---------------------------------------------------------------


def test_absent_config_is_empty_not_an_error(config_file):
    assert config.load_config() == {}


def test_malformed_config_reads_as_empty(config_file):
    config_file.write_text("{not json at all", encoding="utf-8")
    # A broken settings file must not take the server down; the built-in
    # defaults apply instead.
    assert config.load_config() == {}
    assert llm.resolve_model() == llm.DEFAULT_MODEL


def test_a_json_list_is_not_a_config(config_file):
    config_file.write_text('["nope"]', encoding="utf-8")
    assert config.load_config() == {}


def test_bom_is_tolerated(config_file):
    # PowerShell's Out-File writes one, and it would otherwise corrupt the
    # first key's name
    config_file.write_text('﻿{"models": {"default": "m"}}', encoding="utf-8")
    assert llm.resolve_model() == "m"


def test_edits_are_picked_up_without_a_restart(config_file):
    _write(config_file, {"models": {"default": "first"}})
    assert llm.resolve_model() == "first"
    _write(config_file, {"models": {"default": "second"}})
    assert llm.resolve_model() == "second"


# --- precedence ------------------------------------------------------------


def test_env_beats_config_for_the_model(config_file, monkeypatch):
    _write(config_file, {"models": {"default": "from-file"}})
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    assert llm.resolve_model() == "from-env"


def test_explicit_argument_beats_everything(config_file, monkeypatch):
    _write(config_file, {"models": {"default": "from-file"}})
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    assert llm.resolve_model("explicit") == "explicit"


def test_blank_env_falls_through_to_config(config_file, monkeypatch):
    # `set GEMINI_MODEL=` leaves an empty string, which must not count
    _write(config_file, {"models": {"default": "from-file"}})
    monkeypatch.setenv("GEMINI_MODEL", "   ")
    assert llm.resolve_model() == "from-file"


def test_synth_model_follows_the_default_when_unset(config_file):
    _write(config_file, {"models": {"default": "workhorse"}})
    # Someone who changes "the model" means both stages; only an explicit
    # synth entry splits them.
    assert llm.resolve_synth_model() == "workhorse"


def test_synth_model_can_be_split_from_the_default(config_file):
    _write(config_file, {"models": {"default": "cheap", "synth": "fancy"}})
    assert llm.resolve_model() == "cheap"
    assert llm.resolve_synth_model() == "fancy"


def test_passcode_precedence_env_then_config(config_file, monkeypatch):
    # the autouse conftest fixture silences the file read by default, so
    # restore a real read for this one test
    monkeypatch.setattr(auth, "load_config", config.load_config)

    assert auth.configured_passcode() is None

    _write(config_file, {"admin_passcode": "from-config"})
    assert auth.resolve_passcode() == ("from-config", "config")

    monkeypatch.setenv("ADMIN_PASSCODE", "from-env")
    assert auth.resolve_passcode() == ("from-env", "env")


def test_a_blank_passcode_anywhere_means_the_gate_is_off(config_file, monkeypatch):
    monkeypatch.setattr(auth, "load_config", config.load_config)
    for blank in (None, "", "   "):
        _write(config_file, {"admin_passcode": blank})
        assert auth.configured_passcode() is None


# --- vertex project / location ---------------------------------------------
#
# These moved out of a repo-root .env, which is no longer read anywhere. The
# fallbacks below are what keep the setting optional, so they are the part
# worth guarding.


@pytest.fixture()
def no_adc(monkeypatch):
    """No ADC file, so `resolve_project` cannot quietly answer from the
    developer's own gcloud login."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setattr(llm, "_adc_quota_project", lambda: None)


def test_vertex_project_read_from_config(config_file, no_adc):
    _write(config_file, {"vertex": {"project": "from-file"}})
    assert llm.resolve_project() == "from-file"


def test_vertex_env_beats_config(config_file, no_adc, monkeypatch):
    _write(config_file, {"vertex": {"project": "from-file",
                                    "location": "from-file-loc"}})
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-env")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "from-env-loc")
    assert llm.resolve_project() == "from-env"
    assert llm.resolve_location() == "from-env-loc"


def test_explicit_vertex_argument_beats_everything(config_file, no_adc,
                                                   monkeypatch):
    _write(config_file, {"vertex": {"project": "from-file"}})
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-env")
    assert llm.resolve_project("explicit") == "explicit"


def test_blank_vertex_env_falls_through_to_config(config_file, no_adc,
                                                  monkeypatch):
    # `set GOOGLE_CLOUD_PROJECT=` leaves an empty string, which must not count
    _write(config_file, {"vertex": {"project": "from-file"}})
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "   ")
    assert llm.resolve_project() == "from-file"


def test_adc_quota_project_is_the_last_resort(config_file, no_adc, monkeypatch):
    monkeypatch.setattr(llm, "_adc_quota_project", lambda: "from-adc")
    assert llm.resolve_project() == "from-adc"
    # and a configured project still outranks it
    _write(config_file, {"vertex": {"project": "from-file"}})
    assert llm.resolve_project() == "from-file"


def test_location_defaults_to_global(config_file, no_adc):
    assert llm.resolve_location() == "global"
    _write(config_file, {"vertex": {"location": None}})
    assert llm.resolve_location() == "global"


def test_save_rejects_a_blank_vertex_value(config_file):
    with pytest.raises(ValueError, match="vertex.project"):
        config.save_config({"vertex": {"project": "   "}})
    with pytest.raises(ValueError, match="vertex must be an object"):
        config.save_config({"vertex": "my-project"})
    assert not config_file.exists(), "nothing should be written on a rejection"


# --- prices ----------------------------------------------------------------


def test_config_prices_are_merged_over_the_builtin_table(config_file):
    _write(config_file, {"prices_per_mtok": {
        "gemini-3.5-flash-lite": {"input": 1.0, "output": 2.0},
        "some-new-model": {"input": 5.0, "output": 10.0},
    }})
    table = llm.prices_per_mtok()
    assert table["gemini-3.5-flash-lite"] == (1.0, 2.0)   # re-priced
    assert table["some-new-model"] == (5.0, 10.0)         # newly priceable
    assert table["gemini-3.6-flash"] == (1.50, 7.50)      # untouched


def test_price_usd_uses_the_configured_rate(config_file):
    _write(config_file, {"prices_per_mtok": {
        "m": {"input": 1.0, "output": 10.0}}})
    assert llm.price_usd("m", 1_000_000, 1_000_000) == pytest.approx(11.0)


def test_an_unknown_model_stays_unpriced(config_file):
    # None, never a guess: an unpriced run that reports tokens is recoverable,
    # a confidently wrong dollar figure is not
    assert llm.price_usd("never-heard-of-it", 1_000_000, 0) is None


def test_malformed_price_rows_are_dropped_not_raised(config_file):
    _write(config_file, {"prices_per_mtok": {
        "good": {"input": 1.0, "output": 2.0},
        "missing-output": {"input": 1.0},
        "not-a-number": {"input": "free", "output": 2.0},
        "boolean": {"input": True, "output": 2.0},
        "negative": {"input": -1.0, "output": 2.0},
        "not-an-object": 5,
    }})
    # this is read on the cost path mid-run; a bad row must degrade to
    # "unpriced" rather than 500 in the middle of a labeling job
    overrides = config.price_overrides()
    assert set(overrides) == {"good"}


def test_resolve_prices_prefers_flags_then_the_table(config_file):
    _write(config_file, {"prices_per_mtok": {"m": {"input": 3.0, "output": 4.0}}})
    assert llm.resolve_prices("m") == (3.0, 4.0)
    assert llm.resolve_prices("m", price_in=9.0) == (9.0, 4.0)
    assert llm.resolve_prices("m", 9.0, 8.0) == (9.0, 8.0)
    # an unpriced model bills at zero and is visibly unpriced, rather than
    # inheriting some other model's rate
    assert llm.resolve_prices("unknown") == (0.0, 0.0)


# --- writing ---------------------------------------------------------------


def test_save_round_trips_and_preserves_unknown_keys(config_file):
    config.save_config({"models": {"default": "m"}, "future_key": {"a": 1}})
    reloaded = config.load_config()
    assert reloaded["models"]["default"] == "m"
    # an older build must not silently truncate a newer build's file
    assert reloaded["future_key"] == {"a": 1}


def test_save_rejects_a_bad_price_before_writing(config_file):
    with pytest.raises(ValueError, match="must be a number"):
        config.save_config({"prices_per_mtok": {"m": {"input": "free",
                                                      "output": 1.0}}})
    assert not config_file.exists(), "nothing should be written on a rejection"


def test_save_rejects_a_negative_price(config_file):
    with pytest.raises(ValueError, match="negative"):
        config.save_config({"prices_per_mtok": {"m": {"input": -1, "output": 1}}})


def test_save_rejects_a_non_string_passcode(config_file):
    with pytest.raises(ValueError, match="admin_passcode"):
        config.save_config({"admin_passcode": 1234})


def test_save_leaves_no_temp_files_behind(config_file):
    config.save_config({"models": {"default": "m"}})
    leftovers = [p.name for p in config_file.parent.iterdir()
                 if p.suffix == ".tmp"]
    assert leftovers == []


def test_store_calibration_leaves_other_keys_alone(config_file):
    config.save_config({"admin_passcode": "keep-me",
                        "models": {"default": "keep-me-too"}})
    config.store_calibration({"source": "measured", "label": {}})
    reloaded = config.load_config()
    assert reloaded["admin_passcode"] == "keep-me"
    assert reloaded["models"]["default"] == "keep-me-too"
    assert reloaded["cost_calibration"]["source"] == "measured"
