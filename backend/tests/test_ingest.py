import io
import json

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db as db_module
from app import ingest as ingest_module
from app.db import Base, get_db
from app.main import app

CSV_CONTENT = (
    "respondent_id,better_city,unsafe\n"
    "1,More parks,Dark streets\n"
    "2,Lower rent,\n"
    "3,,Speeding cars\n"
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
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


def test_upload_and_select_columns_writes_exports(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    upload_resp = client.post("/datasets/upload", files=files)
    assert upload_resp.status_code == 200
    body = upload_resp.json()
    assert body["row_count"] == 3
    column_names = {c["column"] for c in body["columns"]}
    assert column_names == {"respondent_id", "better_city", "unsafe"}

    dataset_id = body["dataset_id"]

    # Original file lands at data/uploads/{id}/{original_filename}, untouched.
    original_path = tmp_path / "data" / "uploads" / str(dataset_id) / "survey.csv"
    assert original_path.exists()
    assert original_path.read_bytes() == CSV_CONTENT.encode()

    select_resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
                {"column": "unsafe", "label": "What makes it feel unsafe?"},
            ],
        },
    )
    assert select_resp.status_code == 200
    dataset = select_resp.json()
    assert dataset["status"] == "ingested"
    counts = {q["label"]: q["response_count"] for q in dataset["questions"]}
    # blank cells are dropped, so each question keeps only its non-empty answers
    assert counts["What would make the city better?"] == 2
    assert counts["What makes it feel unsafe?"] == 2

    exports = dataset["exports"]
    assert exports["total_row_count"] == 4
    assert exports["per_question_counts"] == {
        "What would make the city better?": 2,
        "What makes it feel unsafe?": 2,
    }

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    parquet_path = export_dir / "responses.parquet"
    csv_path = export_dir / "responses.csv"
    manifest_path = export_dir / "manifest.json"
    assert parquet_path.exists()
    assert csv_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_filename"] == "survey.csv"
    assert manifest["total_row_count"] == 4
    assert manifest["respondent_id_column"] == "respondent_id"
    assert manifest["encoding_repairs_applied"] == 0

    # CSV is UTF-8 with BOM, per spec, so Excel renders it correctly.
    raw_csv = csv_path.read_bytes()
    assert raw_csv.startswith(b"\xef\xbb\xbf")

    csv_download = client.get(f"/datasets/{dataset_id}/exports/csv")
    assert csv_download.status_code == 200
    parquet_download = client.get(f"/datasets/{dataset_id}/exports/parquet")
    assert parquet_download.status_code == 200


def test_export_before_ingest_returns_400(client):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    upload_resp = client.post("/datasets/upload", files=files)
    dataset_id = upload_resp.json()["dataset_id"]

    resp = client.post(f"/datasets/{dataset_id}/export")
    assert resp.status_code == 400


def test_upload_rejects_unsupported_type(client):
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    resp = client.post("/datasets/upload", files=files)
    assert resp.status_code == 400


def test_question_wording_must_differ_from_raw_column_name(client):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": None,
            "questions": [{"column": "better_city", "label": "better_city"}],
        },
    )
    assert resp.status_code == 422

    resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": None,
            "questions": [{"column": "better_city", "label": "  "}],
        },
    )
    assert resp.status_code == 422


def test_reingest_keeps_stable_question_and_response_ids(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    body = {
        "respondent_id_column": "respondent_id",
        "questions": [
            {"column": "better_city", "label": "What would make the city better?"},
            {"column": "unsafe", "label": "What makes it feel unsafe?"},
        ],
    }
    first = client.post(f"/datasets/{dataset_id}/columns", json=body).json()
    first_question_ids = {q["source_column"]: q["id"] for q in first["questions"]}

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    first_table = pq.read_table(export_dir / "responses.parquet")
    first_keys = set(first_table.column("response_key").to_pylist())

    # Re-run the exact same selection — a real analyst re-confirming choices.
    second = client.post(f"/datasets/{dataset_id}/columns", json=body).json()
    second_question_ids = {q["source_column"]: q["id"] for q in second["questions"]}
    second_table = pq.read_table(export_dir / "responses.parquet")
    second_keys = set(second_table.column("response_key").to_pylist())

    assert first_question_ids == second_question_ids
    assert first_keys == second_keys
    assert len(first_keys) == 4


def test_shrinking_selection_clears_stale_export_rows(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
                {"column": "unsafe", "label": "What makes it feel unsafe?"},
            ],
        },
    )

    second = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
            ],
        },
    ).json()

    assert [q["source_column"] for q in second["questions"]] == ["better_city"]
    assert second["exports"]["per_question_counts"] == {
        "What would make the city better?": 2,
    }

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["per_question_counts"] == {
        "What would make the city better?": 2,
    }
    raw_csv = (export_dir / "responses.csv").read_text(encoding="utf-8-sig")
    assert "unsafe" not in raw_csv
    assert "What makes it feel unsafe?" not in raw_csv


def test_encoding_repair_and_raw_text_preserved(client, tmp_path):
    correct_text = "I don’t like the parks"
    mojibake_text = correct_text.encode("utf-8").decode("cp1252")
    assert mojibake_text != correct_text  # sanity: the fixture is actually mangled

    csv_content = f"respondent_id,better_city\n1,{mojibake_text}\n"
    files = {"file": ("survey.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": None,
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"}
            ],
        },
    )
    assert resp.status_code == 200

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["encoding_repairs_applied"] == 1

    table = pq.read_table(export_dir / "responses.parquet")
    row = table.to_pylist()[0]
    assert row["response_text"] == correct_text
    assert row["raw_text_original"] == mojibake_text
    assert row["was_encoding_repaired"] is True
