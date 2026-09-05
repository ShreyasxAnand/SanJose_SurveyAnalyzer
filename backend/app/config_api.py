"""Settings endpoints: read the server's configuration, change it, recalibrate.

  GET  /api/config              what is configured (never the passcode itself)
  PUT  /api/config              change it — admin only
  POST /api/config/recalibrate  re-measure cost rates from this install's runs

**GET is admin-gated too**, unlike every other read in this app. The other
reads return survey data, which this app has never pretended to protect; this
one returns the shape of the lock on the door. Which models are configured and
what they cost is not secret, but "is a passcode set, and did it come from the
file or the environment" is a straight answer to the first question an
intruder asks. When no passcode is configured the gate is a no-op and GET is
open — which is the correct behaviour for the zero-configuration desktop
install, and is what lets someone set the first passcode.

**PUT never returns the passcode and never logs it.** It also refuses to be
the only way in: the file is plain JSON at a path the response reports, so a
forgotten passcode is fixed with an editor, not a reinstall.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from . import calibration, config, llm
from .auth import require_admin, resolve_passcode
from .schemas import (
    CalibrationOut,
    ConfigOut,
    ConfigPatch,
    ModelPriceOut,
)

router = APIRouter(prefix="/config", tags=["config"])


def _env_overrides() -> list[str]:
    """Environment variables currently outranking the file."""
    return [name for name in ("ADMIN_PASSCODE", "GEMINI_MODEL",
                              "GEMINI_SYNTH_MODEL", "GOOGLE_CLOUD_PROJECT",
                              "GOOGLE_CLOUD_LOCATION")
            if (os.environ.get(name) or "").strip()]


def _calibration_out() -> CalibrationOut:
    cal = calibration.active()
    # Per RESPONSE, not per unique text: the stored rate is per unique because
    # that is what the run batches, but "tokens per response" is the number an
    # analyst can check against a row count they can see.
    ratio = cal.unique_ratio
    return CalibrationOut(
        source=cal.source,
        measured_utc=cal.measured_utc,
        sample=cal.sample,
        label_input_tokens_per_response=round(
            cal.label_input_per_unique * ratio, 2),
        label_output_tokens_per_response=round(
            cal.label_output_per_unique * ratio, 2),
        induce_output_tokens_per_chunk=round(cal.induce_output_per_chunk, 1),
        candidates_per_response=round(cal.candidates_per_response, 3),
        spread_low=cal.spread[0],
        spread_high=cal.spread[1],
    )


def _config_out() -> ConfigOut:
    overrides = config.price_overrides()
    prices = [
        ModelPriceOut(model_id=model_id, input_per_mtok=rate[0],
                      output_per_mtok=rate[1],
                      overridden=model_id in overrides)
        for model_id, rate in sorted(llm.prices_per_mtok().items())
    ]
    # Asked of auth rather than re-derived here. Someone editing the passcode
    # on this screen while ADMIN_PASSCODE is set in the environment would
    # otherwise save successfully, be told it came from config.json, and find
    # the old passcode still in force.
    _passcode, passcode_source = resolve_passcode()
    # `configured_vertex` is what the file (or an environment variable) says;
    # `resolve_*` is what a run would actually get. They differ whenever
    # nothing is configured and ADC supplies the project, which is the common
    # case — see ConfigOut for why the screen needs both.
    return ConfigOut(
        admin_passcode_set=passcode_source != "",
        admin_passcode_source=passcode_source,
        default_model=llm.resolve_model(),
        synth_model=llm.resolve_synth_model(),
        vertex_project=config.configured_vertex(
            "project", "GOOGLE_CLOUD_PROJECT"),
        vertex_project_effective=llm.resolve_project(),
        vertex_location=config.configured_vertex(
            "location", "GOOGLE_CLOUD_LOCATION"),
        vertex_location_effective=llm.resolve_location(),
        prices=prices,
        calibration=_calibration_out(),
        config_path=str(config.CONFIG_PATH),
        config_exists=config.CONFIG_PATH.exists(),
        env_overrides=_env_overrides(),
    )


@router.get("", response_model=ConfigOut,
            dependencies=[Depends(require_admin)])
def get_config() -> ConfigOut:
    """What is configured. The passcode is reported as set/unset and by
    source, never by value."""
    return _config_out()


@router.put("", response_model=ConfigOut,
            dependencies=[Depends(require_admin)])
def put_config(patch: ConfigPatch) -> ConfigOut:
    """Apply a partial change to config.json.

    Read-modify-write of the whole file rather than a targeted edit, so keys
    this build does not know about survive a save by a build that does not
    know about them.
    """
    current = dict(config.load_config())

    if patch.admin_passcode is not None:
        # "" is meaningful: it clears the passcode and turns the gate off.
        # Storing null rather than "" so the file reads as "not configured"
        # rather than "configured to the empty string".
        cleaned = patch.admin_passcode.strip()
        current["admin_passcode"] = cleaned or None

    if patch.default_model is not None or patch.synth_model is not None:
        models = dict(current.get("models") or {})
        if patch.default_model is not None:
            models["default"] = patch.default_model.strip()
        if patch.synth_model is not None:
            models["synth"] = patch.synth_model.strip()
        current["models"] = models

    if patch.vertex_project is not None or patch.vertex_location is not None:
        # Same "" convention as the passcode: a cleared box removes the
        # override rather than storing an empty string, so the file reads as
        # "not configured" and the ADC / global-endpoint fallbacks resume.
        vertex = dict(current.get("vertex") or {})
        if patch.vertex_project is not None:
            vertex["project"] = patch.vertex_project.strip() or None
        if patch.vertex_location is not None:
            vertex["location"] = patch.vertex_location.strip() or None
        current["vertex"] = vertex

    if patch.prices is not None:
        current["prices_per_mtok"] = {
            row.model_id.strip(): {"input": float(row.input_per_mtok),
                                   "output": float(row.output_per_mtok)}
            for row in patch.prices
        }

    try:
        config.save_config(current)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not write {config.CONFIG_PATH}: {exc}")
    return _config_out()


@router.post("/recalibrate", response_model=CalibrationOut,
             dependencies=[Depends(require_admin)])
def recalibrate() -> CalibrationOut:
    """Re-measure cost rates from this install's completed run manifests.

    Free and offline — it reads files this machine already wrote. Admin-gated
    only because it writes config.json, not because it costs anything.
    """
    try:
        calibration.recalibrate()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not write {config.CONFIG_PATH}: {exc}")
    return _calibration_out()
