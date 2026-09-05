"""Admin passcode gate for dataset-changing endpoints.

One shared passcode. It is read from ADMIN_PASSCODE in the environment, then
`admin_passcode` in the repo-root config.json — see `config` for why the
environment wins. When it is set, every endpoint that creates, changes,
deletes, or processes a dataset requires the passcode in an X-Admin-Passcode
request header; reading and asking stay open. When it is NOT set, nothing is
gated — the single-user desktop setup keeps working with zero configuration.

This is deliberately a passcode, not accounts: the threat it addresses is a
colleague on the LAN uploading files, deleting a dataset, or starting billed
pipeline runs, not a determined attacker (the app still has no login for
reading). The check is a constant-time comparison so the passcode cannot be
guessed a character at a time from response timing.

The configured value is re-read on every request (a stat, and a file read of a
few hundred bytes only when it changed), so adding or changing the passcode —
from the Settings screen or by hand — needs no server restart.
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException

# Imported as a bound name on purpose: it is the seam a test replaces, and
# patching `auth.load_config` must actually silence config.json.
from .config import load_config


def resolve_passcode() -> tuple[str | None, str]:
    """`(passcode, source)` — the active passcode and which layer supplied it.

    Whitespace-only counts as unset at every layer: a blank value is how
    someone turns the gate off, not a passcode made of spaces. Each source is
    read on every call — no caching — so a passcode saved from the Settings
    screen is in force for the very next request.

    The source travels with the value because the Settings screen reports it,
    and it must be derived from the same reads the gate performs. Deriving it
    separately is how a screen ends up saying "from config.json" while an
    environment variable is quietly winning.
    """
    value = os.environ.get("ADMIN_PASSCODE")
    if value is not None and value.strip():
        return value.strip(), "env"
    value = load_config().get("admin_passcode")
    if isinstance(value, str) and value.strip():
        return value.strip(), "config"
    return None, ""


def configured_passcode() -> str | None:
    """The active passcode, or None when the gate is off."""
    return resolve_passcode()[0]


def require_admin(
    x_admin_passcode: str | None = Header(default=None),
) -> None:
    """FastAPI dependency for admin-only endpoints. 401 carries a message the
    UI shows verbatim, so it must make sense to a person, not just a client."""
    expected = configured_passcode()
    if expected is None:
        return
    # Compared as BYTES, not as str. secrets.compare_digest raises TypeError
    # on a str containing any non-ASCII character, so a passcode with an
    # accent, a smart quote pasted from a document, or an emoji would turn
    # every gated request into a 500 — locking the operator out of the
    # Settings screen that would let them fix it. Encoding first is still a
    # constant-time comparison and removes the restriction entirely.
    if not x_admin_passcode or not secrets.compare_digest(
        x_admin_passcode.strip().encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="This action needs the admin passcode. Enter it when the "
            "app asks, or set one on the Settings screen (it is stored in the "
            "server's config.json) if none is configured yet.",
        )
