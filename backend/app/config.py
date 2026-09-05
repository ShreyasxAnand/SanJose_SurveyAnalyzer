"""Editable server settings: `config.json` at the repo root.

Everything settable lives here: the admin passcode, which Gemini model each
stage uses, the Vertex project and location, and the per-model token prices the
cost estimate is built from. A fifth key, `cost_calibration`, is written by the
calibrator (see `calibration.py`) rather than by hand.

**Precedence is always environment > config.json > the code default.** An
environment variable wins because that is how a container or a CI job overrides
a checked-out file, and reversing it would make a deployment's own settings
unexplainable.

There used to be a third layer, a repo-root `.env`, read below config.json. It
existed to hold `GEMINI_API_KEY` and outlived the key: the move to Vertex ADC
removed the secret it was invented for, and `config.json` — a file the app
writes itself, from a screen, with validation — replaced it for everything
else. Two layers is the whole rule now, and `.env` is not read anywhere.

**Nothing is cached across an edit.** A read re-checks the file's mtime and
size on every call, so saving config.json from the Settings screen (or in an
editor) takes effect on the next request with no restart. The parsed dict is
memoised between edits because the price table is consulted once per model
call, and a labeling run makes thousands.

**A broken config never takes the server down.** Unreadable or malformed JSON
reads as `{}` — the app falls back to its built-in defaults and keeps serving.
The one place that is not tolerated is `save_config`, which validates before
writing, so the file on disk is always loadable.

The passcode is stored here but resolved in `auth.configured_passcode` — that
module owns the gate, and keeping the lookup there keeps both sources patchable
in one place.

The file holds a passcode, so it is gitignored; `config.example.json` is the
checked-in template.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"


# (mtime_ns, size) of the file the cached dict was parsed from. A pair rather
# than mtime alone because a same-instant rewrite of identical length is
# exactly what an editor does, and mtime alone has bitten this pattern before.
_CACHE_LOCK = threading.Lock()
_cache_stamp: tuple[int, int] | None = None
_cache_value: dict[str, Any] = {}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """The parsed config.json, or `{}` when it is absent or unreadable.

    An explicit `path` bypasses the memo — that is the test seam, and a one-off
    read of a named file has no business poisoning the cache for the real one.
    """
    if path is not None:
        return _read(path)

    global _cache_stamp, _cache_value
    try:
        st = CONFIG_PATH.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        # absent (or unstatable) — the ordinary zero-configuration case
        with _CACHE_LOCK:
            _cache_stamp, _cache_value = None, {}
        return {}

    with _CACHE_LOCK:
        if _cache_stamp == stamp:
            return _cache_value
    parsed = _read(CONFIG_PATH)
    with _CACHE_LOCK:
        _cache_stamp, _cache_value = stamp, parsed
    return parsed


def _read(path: Path) -> dict[str, Any]:
    try:
        # utf-8-sig because editors and PowerShell's Out-File on this
        # platform write a BOM, which would otherwise corrupt the first key
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    # a JSON file whose top level is a list or a string is not a config; treat
    # it the same as unreadable rather than let .get() blow up downstream
    return parsed if isinstance(parsed, dict) else {}


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    """Write config.json atomically, validating first.

    Written to a temp file in the same directory and then `os.replace`d, so a
    concurrent reader never sees half a file and a crash mid-write cannot leave
    the app with no settings at all. Raises ValueError on anything
    `load_config` would otherwise have to defend against.
    """
    target = path or CONFIG_PATH
    validate_config(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # the file just changed under us; drop the memo so the next read re-parses
    # even if the filesystem's timestamp resolution hid the write
    global _cache_stamp
    with _CACHE_LOCK:
        _cache_stamp = None
    return target


def validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError if `config` is not a shape `load_config` can serve.

    Strict about types, permissive about unknown keys: a typo in a price is a
    number the cost estimate would silently believe, whereas an extra key is at
    worst ignored — and refusing it would mean an older build could not save a
    file written by a newer one.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")

    passcode = config.get("admin_passcode")
    if passcode is not None:
        if not isinstance(passcode, str):
            raise ValueError(
                "admin_passcode must be a string, or null to clear it.")
        # ASCII only, and that is not fussiness. The passcode travels in an
        # HTTP header, and header values are bytes: a browser encodes a
        # non-ASCII character as latin-1 while curl sends UTF-8, so the two
        # would disagree about the same passcode and one of them would never
        # get in. Characters above U+00FF cannot be put in a header by a
        # browser at all. Refusing at save time beats a lockout discovered
        # later from the wrong client.
        if not passcode.isascii():
            offending = sorted({c for c in passcode if not c.isascii()})
            raise ValueError(
                f"admin_passcode must use ASCII characters only — "
                f"{''.join(offending)!r} cannot be sent reliably in an HTTP "
                f"header. Smart quotes pasted from a document are the usual "
                f"cause."
            )
        if any(ord(c) < 32 or ord(c) == 127 for c in passcode):
            raise ValueError(
                "admin_passcode cannot contain control characters "
                "(a stray tab or newline, usually from a paste).")

    models = config.get("models")
    if models is not None:
        if not isinstance(models, dict):
            raise ValueError(
                "models must be an object keyed by stage, e.g. "
                "{'default': 'gemini-3.5-flash-lite'}."
            )
        for key, value in models.items():
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"models.{key} must be a non-empty model id, or null to use "
                    f"the built-in default."
                )

    prices = config.get("prices_per_mtok")
    if prices is not None:
        if not isinstance(prices, dict):
            raise ValueError("prices_per_mtok must be an object keyed by model id.")
        for model_id, rate in prices.items():
            if not isinstance(rate, dict):
                raise ValueError(
                    f"prices_per_mtok for {model_id!r} must be an object with "
                    f"'input' and 'output'."
                )
            for field in ("input", "output"):
                if field not in rate:
                    raise ValueError(
                        f"prices_per_mtok for {model_id!r} is missing {field!r}."
                    )
                value = rate[field]
                # bool is an int subclass; True as a price is a bug, not a rate
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"prices_per_mtok for {model_id!r}: {field} must be a "
                        f"number ($ per 1M tokens)."
                    )
                if value < 0:
                    raise ValueError(
                        f"prices_per_mtok for {model_id!r}: {field} cannot be "
                        f"negative."
                    )

    vertex = config.get("vertex")
    if vertex is not None:
        if not isinstance(vertex, dict):
            raise ValueError(
                "vertex must be an object, e.g. "
                "{'project': 'my-project', 'location': 'global'}."
            )
        fallbacks = {"project": "the ADC file's quota project",
                     "location": "the global endpoint"}
        for key in ("project", "location"):
            value = vertex.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"vertex.{key} must be a non-empty string, or null to "
                    f"fall back to {fallbacks[key]}."
                )

    calibration_block = config.get("cost_calibration")
    if calibration_block is not None and not isinstance(calibration_block, dict):
        raise ValueError("cost_calibration must be an object.")


# ---------------------------------------------------------------------------
# Typed accessors — what the rest of the app calls instead of reading the dict
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() else None


def configured_model(kind: str, env_var: str, fallback: str) -> str:
    """The model id for one stage. `kind` is a key under `models` in
    config.json ("default" for the stages whose call count scales with the
    corpus, "synth" for answer writing); `env_var` is the variable that
    overrides it."""
    from_env = _env(env_var)
    if from_env:
        return from_env
    models = load_config().get("models")
    if isinstance(models, dict):
        value = models.get(kind)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def configured_vertex(key: str, env_var: str) -> str | None:
    """`vertex.project` / `vertex.location` from config.json, or the
    environment variable that outranks it. None when neither is set.

    Returns None rather than a fallback because the two callers have different
    last resorts — the project falls back to the ADC file's quota project, the
    location to the global endpoint — and neither belongs in a settings lookup.
    """
    from_env = _env(env_var)
    if from_env:
        return from_env
    vertex = load_config().get("vertex")
    if isinstance(vertex, dict):
        value = vertex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def price_overrides() -> dict[str, tuple[float, float]]:
    """Configured `(input, output)` $/MTok pairs, keyed by model id.

    Merged over the built-in table by `llm.prices_per_mtok`, so one config
    entry can both re-price a known model and add an unknown one. Malformed
    entries are dropped rather than raised on: this is read on the cost path,
    and a bad rate should degrade to "unpriced" — which the UI already reports
    honestly — never to a 500 in the middle of a run.
    """
    prices = load_config().get("prices_per_mtok")
    if not isinstance(prices, dict):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for model_id, rate in prices.items():
        if not isinstance(model_id, str) or not isinstance(rate, dict):
            continue
        pin, pout = rate.get("input"), rate.get("output")
        if isinstance(pin, bool) or isinstance(pout, bool):
            continue
        if isinstance(pin, (int, float)) and isinstance(pout, (int, float)):
            if pin >= 0 and pout >= 0:
                out[model_id] = (float(pin), float(pout))
    return out


def calibration() -> dict[str, Any]:
    """The stored cost calibration, or `{}` when none has been measured yet."""
    value = load_config().get("cost_calibration")
    return value if isinstance(value, dict) else {}


def store_calibration(measured: dict[str, Any]) -> None:
    """Persist a freshly measured calibration, leaving every other key alone."""
    current = dict(load_config())
    current["cost_calibration"] = measured
    save_config(current)
