"""The permanent-delete endpoints, end to end.

Deletion is the only irreversible action in this app, so the tests that matter
most are the ones about refusing: a wrong confirmation name, a running
pipeline, and a missing passcode all have to leave the dataset intact.
"""
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db as db_module
from app import ingest as ingest_module
from app import purge as purge_module
from app import summary as summary_module
from app.db import Base, get_db
from app.main import app

CSV_CONTENT = (
    "respondent_id,better_city,unsafe\n"
    "1,More parks,Dark streets\n"
    "2,Lower rent,\n"
    "3,,Speeding cars\n"
)

INGEST_BODY = {
    "respondent_id_column": "respondent_id",
    "questions": [
        {"column": "better_city", "label": "What would make the city better?"},
        {"column": "unsafe", "label": "What makes it feel unsafe?"},
    ],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False,
                                       autocommit=False)
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
    # EVERY artifact root, not just the two ingest writes. Test dataset ids
    # start at 1 and so do the real ones, so an unpatched root makes the
    # preview report the developer's actual lexicon and rulings — and a purge
    # test would delete them.
    from app import induction, lexicon, locations, rulings, subthemes

    monkeypatch.setattr(induction, "DATA_DIR", data_dir)
    monkeypatch.setattr(induction, "TAXONOMY_DIR", data_dir / "taxonomy")
    monkeypatch.setattr(summary_module, "LABELS_DIR", data_dir / "labels")
    monkeypatch.setattr(summary_module, "SUMMARY_DIR", data_dir / "summary")
    monkeypatch.setattr(subthemes, "SUBTHEMES_DIR", data_dir / "subthemes")
    monkeypatch.setattr(lexicon, "LEXICON_DIR", data_dir / "lexicon")
    monkeypatch.setattr(locations, "LOCATIONS_DIR", data_dir / "locations")
    monkeypatch.setattr(rulings, "RULINGS_DIR", data_dir / "rulings")
    monkeypatch.setattr(purge_module, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(purge_module, "EXPORTS_DIR", exports_dir)
    monkeypatch.setattr(purge_module, "LABELS_DIR", data_dir / "labels")
    monkeypatch.setattr(purge_module, "SUMMARY_DIR", data_dir / "summary")
    # a root that resolves outside tmp_path would let this suite delete real
    # artifacts; fail loudly rather than find out afterwards
    for root in purge_module.ROOTS:
        assert str(root.path()).startswith(str(tmp_path)), (
            f"purge root {root.key!r} is not redirected into tmp_path — "
            f"it resolves to {root.path()}"
        )

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        c.data_dir = data_dir
        yield c
    app.dependency_overrides.clear()


def _ingest(client, name_suffix=""):
    files = {"file": (f"survey{name_suffix}.csv",
                      io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]
    res = client.post(f"/api/datasets/{dataset_id}/columns", json=INGEST_BODY)
    assert res.status_code == 200
    return res.json()


# --- preview ---------------------------------------------------------------


def test_preview_reports_what_exists(client):
    dataset = _ingest(client)
    body = client.get(
        f"/api/datasets/{dataset['id']}/deletion-preview").json()

    assert body["dataset_name"] == dataset["name"]
    assert body["status"] == "ingested"
    # 3 rows x 2 question columns, minus the two blanks
    assert body["n_responses"] == 4
    assert body["n_questions"] == 2
    assert body["n_uploads"] == 1
    assert {r["key"] for r in body["roots"]} == {"uploads", "exports"}
    assert body["total_files"] > 0
    assert body["blocked_by_running_job"] is False


def test_preview_of_a_missing_dataset_is_404(client):
    assert client.get("/api/datasets/999/deletion-preview").status_code == 404


def test_preview_counts_recorded_spend(client):
    dataset = _ingest(client)
    run = (client.data_dir / "labels" / str(dataset["id"]) / "1" / "run1")
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        '{"tool": "scripts.label", "usage": {"est_cost_usd": 3.5}}',
        encoding="utf-8")

    body = client.get(
        f"/api/datasets/{dataset['id']}/deletion-preview").json()
    assert body["total_spent_usd"] == pytest.approx(3.5)
    labels = next(r for r in body["roots"] if r["key"] == "labels")
    assert labels["runs"] == 1


def test_preview_is_open_to_readers(client, monkeypatch):
    # a read that names what would be lost is not a privileged operation, and
    # gating it would mean the confirm dialog could not render the warning
    # that talks someone out of the delete
    dataset = _ingest(client)
    monkeypatch.setenv("ADMIN_PASSCODE", "hunter2")
    assert client.get(
        f"/api/datasets/{dataset['id']}/deletion-preview").status_code == 200


# --- deleting --------------------------------------------------------------


def test_delete_removes_rows_and_every_artifact(client):
    dataset = _ingest(client)
    dataset_id = dataset["id"]
    labels = client.data_dir / "labels" / str(dataset_id) / "1" / "run1"
    labels.mkdir(parents=True)
    (labels / "assignments.json").write_text("[]", encoding="utf-8")
    uploads = client.data_dir / "uploads" / str(dataset_id)
    assert uploads.exists()

    res = client.request(
        "DELETE", f"/api/datasets/{dataset_id}/permanently",
        params={"confirm_name": dataset["name"]})

    assert res.status_code == 200
    assert res.json()["deleted"] is True
    # the whole point of reporting this: a leftover directory is adopted by
    # whichever dataset next gets this id
    assert res.json()["undeleted_roots"] == []
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 404
    assert not uploads.exists()
    assert not (client.data_dir / "labels" / str(dataset_id)).exists()
    assert not (client.data_dir / "exports" / str(dataset_id)).exists()


def test_delete_leaves_other_datasets_alone(client):
    keep = _ingest(client, "-keep")
    doomed = _ingest(client, "-doomed")

    client.request("DELETE", f"/api/datasets/{doomed['id']}/permanently",
                   params={"confirm_name": doomed["name"]})

    assert client.get(f"/api/datasets/{keep['id']}").status_code == 200
    assert (client.data_dir / "uploads" / str(keep["id"])).exists()


def test_a_wrong_confirmation_name_deletes_nothing(client):
    dataset = _ingest(client)
    res = client.request(
        "DELETE", f"/api/datasets/{dataset['id']}/permanently",
        params={"confirm_name": "not the name"})

    assert res.status_code == 400
    assert "Nothing was deleted" in res.json()["detail"]
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 200
    assert (client.data_dir / "uploads" / str(dataset["id"])).exists()


def test_a_missing_confirmation_name_deletes_nothing(client):
    dataset = _ingest(client)
    res = client.request(
        "DELETE", f"/api/datasets/{dataset['id']}/permanently")
    assert res.status_code == 400
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 200


def test_surrounding_whitespace_in_the_name_is_tolerated(client):
    dataset = _ingest(client)
    # a name copied out of the dialog can pick up a trailing space; that is a
    # typo, not a different dataset
    res = client.request(
        "DELETE", f"/api/datasets/{dataset['id']}/permanently",
        params={"confirm_name": f"  {dataset['name']}  "})
    assert res.status_code == 200


def test_delete_refuses_while_a_pipeline_job_runs(client, monkeypatch):
    from app import pipeline

    dataset = _ingest(client)
    monkeypatch.setattr(
        pipeline, "latest_job",
        lambda ds: pipeline.Job(job_id="j1", dataset_id=str(dataset["id"]),
                                status="running"))

    res = client.request(
        "DELETE", f"/api/datasets/{dataset['id']}/permanently",
        params={"confirm_name": dataset["name"]})

    # deleting a tree a subprocess is mid-write in leaves a half-removed run
    # that still looks loadable
    assert res.status_code == 409
    assert "Nothing was deleted" in res.json()["detail"]
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 200
    # and the preview says so in advance, so the button can be disabled
    preview = client.get(
        f"/api/datasets/{dataset['id']}/deletion-preview").json()
    assert preview["blocked_by_running_job"] is True


def test_delete_is_admin_gated(client, monkeypatch):
    dataset = _ingest(client)
    monkeypatch.setenv("ADMIN_PASSCODE", "hunter2")

    unauthorized = client.request(
        "DELETE", f"/api/datasets/{dataset['id']}/permanently",
        params={"confirm_name": dataset["name"]})
    assert unauthorized.status_code == 401
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 200

    ok = client.request(
        "DELETE", f"/api/datasets/{dataset['id']}/permanently",
        params={"confirm_name": dataset["name"]},
        headers={"X-Admin-Passcode": "hunter2"})
    assert ok.status_code == 200


def test_delete_clears_the_ask_context_cache(client, monkeypatch):
    from app import ask_service

    cleared = []
    monkeypatch.setattr(ask_service, "invalidate_context_cache", cleared.append)
    dataset = _ingest(client)

    client.request("DELETE", f"/api/datasets/{dataset['id']}/permanently",
                   params={"confirm_name": dataset["name"]})

    # the export write also invalidates, so assert the delete's own call is
    # in there rather than that it is the only one
    assert str(dataset["id"]) in [str(c) for c in cleared]


def test_the_provisional_discard_still_refuses_an_ingested_dataset(client):
    dataset = _ingest(client)
    res = client.delete(f"/api/datasets/{dataset['id']}")
    assert res.status_code == 409
    # the pagehide handler fires this when a tab closes mid-upload; it must
    # never be able to destroy processed work
    assert "permanently" in res.json()["detail"]
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 200
