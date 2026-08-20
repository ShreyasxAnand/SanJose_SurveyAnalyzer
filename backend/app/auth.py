"""Admin passcode gate for dataset-changing endpoints.

One shared passcode, set as ADMIN_PASSCODE in the environment or the repo-root
.env. When it is set, every endpoint
that creates, changes, or processes a dataset requires the passcode in an
X-Admin-Passcode request header; reading and asking stay open. When it is NOT
set, nothing is gated — the single-user desktop setup keeps working with zero
configuration.

This is deliberately a passcode, not accounts: the threat it addresses is a
colleague on the LAN uploading files or starting billed pipeline runs, not a
determined attacker (the app still has no login for reading). The check is a
constant-time comparison so the passcode cannot be guessed a character at a
time from response timing.

The configured value is re-read from .env on every request (a file read of a
few hundred bytes), so adding or changing the passcode needs no server
restart. An environment variable, when present, wins over the file.
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException

from .llm import load_dotenv


def configured_passcode() -> str | None:
    """The active passcode, or None when the gate is off."""
    value = os.environ.get("ADMIN_PASSCODE")
    if value is not None and value.strip():
        return value.strip()
    value = load_dotenv().get("ADMIN_PASSCODE")
    if value is not None and value.strip():
        return value.strip()
    return None


def require_admin(
    x_admin_passcode: str | None = Header(default=None),
) -> None:
    """FastAPI dependency for admin-only endpoints. 401 carries a message the
    UI shows verbatim, so it must make sense to a person, not just a client."""
    expected = configured_passcode()
    if expected is None:
        return
    if not x_admin_passcode or not secrets.compare_digest(
        x_admin_passcode.strip(), expected
    ):
        raise HTTPException(
            status_code=401,
            detail="This action needs the admin passcode. Enter it when the "
            "app asks, or set the ADMIN_PASSCODE line in the server's .env "
            "file if none is configured yet.",
        )
