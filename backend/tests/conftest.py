"""Shared test setup.

Every test runs with the admin passcode gate OFF unless it explicitly turns
it on: a developer who has ADMIN_PASSCODE set in their environment or repo
.env must not see unrelated tests start answering 401.
"""
import pytest

from app import auth


@pytest.fixture(autouse=True)
def _no_admin_passcode(monkeypatch):
    monkeypatch.delenv("ADMIN_PASSCODE", raising=False)
    monkeypatch.setattr(auth, "load_dotenv", lambda *a, **k: {})
