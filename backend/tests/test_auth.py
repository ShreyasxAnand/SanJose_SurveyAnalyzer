"""The admin passcode gate: off by default, constant-time when on, applied to
the dataset-changing endpoints and nothing else."""
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import auth
from app import db as db_module
from app import ingest as ingest_module
from app.db import Base, get_db
from app.main import app

CSV_CONTENT = "respondent_id,q\n1,parks\n2,rent\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    data_dir = tmp_path / "data"
    uploads_dir = data_dir / "uploads"
    exports_dir = data_dir / "exports"
    uploads_dir.mkdir(parents=True)
    exports_dir.mkdir(parents=True)
    for module in (db_module, ingest_module):
        monkeypatch.setattr(module, "DATA_DIR", data_dir, raising=False)
        monkeypatch.setattr(module, "UPLOADS_DIR", uploads_dir, raising=False)
        monkeypatch.setattr(module, "EXPORTS_DIR", exports_dir, raising=False)
    monkeypatch.setattr(ingest_module, "REPO_ROOT", tmp_path)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _upload(client, headers=None):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    return client.post("/api/datasets/upload", files=files, headers=headers or {})


def test_gate_off_when_no_passcode_configured(client):
    assert _upload(client).status_code == 200


def test_upload_requires_passcode_when_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSCODE", "open-sesame")
    assert _upload(client).status_code == 401
    assert _upload(client, {"X-Admin-Passcode": "wrong"}).status_code == 401
    assert _upload(client, {"X-Admin-Passcode": "open-sesame"}).status_code == 200


def test_passcode_read_from_config_file(client, monkeypatch):
    # env var absent (autouse fixture), config.json carries the passcode
    monkeypatch.setattr(auth, "load_config",
                        lambda *a, **k: {"admin_passcode": "s3cret"})
    assert _upload(client).status_code == 401
    assert _upload(client, {"X-Admin-Passcode": "s3cret"}).status_code == 200


def test_env_var_wins_over_the_config_file(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSCODE", "from-env")
    monkeypatch.setattr(auth, "load_config",
                        lambda *a, **k: {"admin_passcode": "from-file"})
    assert _upload(client, {"X-Admin-Passcode": "from-file"}).status_code == 401
    assert _upload(client, {"X-Admin-Passcode": "from-env"}).status_code == 200


def test_surrounding_whitespace_tolerated(client, monkeypatch):
    # a trailing space pasted along with the passcode must not lock the user out
    monkeypatch.setenv("ADMIN_PASSCODE", "open-sesame")
    assert _upload(client, {"X-Admin-Passcode": " open-sesame "}).status_code == 200


def test_read_endpoints_stay_open(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSCODE", "open-sesame")
    assert client.get("/api/datasets").status_code == 200


def test_other_mutating_endpoints_are_gated(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSCODE", "open-sesame")
    ok = {"X-Admin-Passcode": "open-sesame"}
    dataset_id = _upload(client, ok).json()["dataset_id"]

    body = {"questions": [{"column": "q", "label": "What would help?"}]}
    assert client.post(f"/api/datasets/{dataset_id}/columns", json=body).status_code == 401
    assert client.patch(f"/api/datasets/{dataset_id}", json={"name": "x"}).status_code == 401
    assert client.post(f"/api/datasets/{dataset_id}/export").status_code == 401
    assert client.post(
        f"/api/datasets/{dataset_id}/append", json={"upload_dataset_id": 999}
    ).status_code == 401
    # the dependency runs before the handler, so no job setup is needed
    assert client.post(
        f"/api/datasets/{dataset_id}/pipeline/run", json={}
    ).status_code == 401
    assert client.post(f"/api/datasets/{dataset_id}/pipeline/cancel").status_code == 401
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 401

    # and the whole ingest flow still works with the passcode attached
    select = client.post(f"/api/datasets/{dataset_id}/columns", json=body, headers=ok)
    assert select.status_code == 200
    assert client.delete(f"/api/datasets/{dataset_id}", headers=ok).status_code in (204, 409)


def test_a_non_ascii_passcode_never_500s(client, monkeypatch):
    """A non-ASCII passcode must fail closed, not crash.

    `secrets.compare_digest` raises TypeError on a str containing any
    non-ASCII character, so comparing without encoding first turned an
    accented passcode into a 500 on every gated request — locking the operator
    out of the very screen that would let them change it. `config.json` now
    refuses to store one, but it can still arrive from a hand-edited file or
    an environment variable, so the gate has to cope.

    Note there is no "and the right passcode gets in" case here, because there
    isn't one: an HTTP header is bytes, and the round trip does not preserve
    the string (`café` arrives as `cafÃ©`). That is precisely why
    setting one is refused — see test_config_api.
    """
    monkeypatch.setenv("ADMIN_PASSCODE", "café-señor")

    for attempt in (None, {"X-Admin-Passcode": "wrong"},
                    {"X-Admin-Passcode": "café-señor".encode("utf-8")}):
        res = _upload(client, attempt)
        assert res.status_code == 401, f"{attempt!r} gave {res.status_code}"


def test_a_non_ascii_attempt_against_an_ascii_passcode_is_a_clean_401(
        client, monkeypatch):
    # the other half of the same bug: a non-ASCII ATTEMPT must be refused,
    # not crash the request
    monkeypatch.setenv("ADMIN_PASSCODE", "open-sesame")
    res = _upload(client, {"X-Admin-Passcode": "café".encode("utf-8")})
    assert res.status_code == 401
