import io
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db as db_module
from app import ingest as ingest_module
from app import summary as summary_module
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
    # _write_exports joins the latest labels run via summary.LABELS_DIR —
    # point it into the tmp tree so a test dataset id that collides with a
    # real one (both start at 1) can't leak repo labels into test exports
    monkeypatch.setattr(summary_module, "LABELS_DIR", data_dir / "labels")

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


def test_excel_blank_cells_do_not_crash_reshape(client):
    # Unlike CSV, Excel can yield NaN (a float) for a genuinely blank cell
    # even under dtype=str/keep_default_na=False — this reproduces that.
    df = pd.DataFrame(
        {
            "respondent_id": ["1", "2", "3"],
            "better_city": ["More parks", "Lower rent", None],
            "unsafe": [None, None, "Speeding cars"],
        }
    )
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    files = {
        "file": (
            "survey.xlsx",
            buffer,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    upload_resp = client.post("/datasets/upload", files=files)
    assert upload_resp.status_code == 200
    dataset_id = upload_resp.json()["dataset_id"]

    resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
                {"column": "unsafe", "label": "What makes it feel unsafe?"},
            ],
        },
    )
    assert resp.status_code == 200
    counts = {q["label"]: q["response_count"] for q in resp.json()["questions"]}
    assert counts["What would make the city better?"] == 2
    assert counts["What makes it feel unsafe?"] == 1


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


def test_sentinel_rows_flagged_not_dropped(client, tmp_path):
    # "n/a" is a sentinel; "None" deliberately is not (a real answer to
    # "what makes you feel unsafe"). Both are stored and exported — flagged,
    # never dropped.
    csv_content = (
        "respondent_id,better_city\n"
        "1,n/a\n"
        "2,None\n"
        "3,More parks\n"
    )
    files = {"file": ("survey.csv", io.BytesIO(csv_content.encode()), "text/csv")}
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
    rows = {
        r["response_text"]: r
        for r in pq.read_table(export_dir / "responses.parquet").to_pylist()
    }
    assert len(rows) == 3  # all three stored
    assert rows["n/a"]["is_nonanswer"] is True
    assert rows["None"]["is_nonanswer"] is False
    assert rows["More parks"]["is_nonanswer"] is False

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["per_question_nonanswer_counts"] == {
        "What would make the city better?": 1,
    }
    assert manifest["per_question_counts"] == {
        "What would make the city better?": 3,
    }


def test_empty_selected_column_is_400(client):
    csv_content = (
        "respondent_id,better_city,empty_col\n"
        "1,More parks,\n"
        "2,Lower rent,\n"
    )
    files = {"file": ("survey.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": None,
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
                {"column": "empty_col", "label": "A question nobody answered?"},
            ],
        },
    )
    assert resp.status_code == 400
    assert "empty_col" in resp.json()["detail"]
    # nothing was ingested
    assert client.get(f"/datasets/{dataset_id}").json()["status"] == "uploaded"


def test_description_round_trip(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    body = {
        "respondent_id_column": "respondent_id",
        "questions": [
            {"column": "better_city", "label": "What would make the city better?"},
        ],
        "dataset_description": "A survey of city residents about their city.",
    }
    first = client.post(f"/datasets/{dataset_id}/columns", json=body).json()
    assert first["description"] == "A survey of city residents about their city."

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_description"] == (
        "A survey of city residents about their city."
    )

    # Re-selecting without the field preserves the saved description.
    body.pop("dataset_description")
    second = client.post(f"/datasets/{dataset_id}/columns", json=body).json()
    assert second["description"] == "A survey of city residents about their city."


def test_export_joins_latest_labels(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
            ],
        },
    ).json()
    question_id = resp["questions"][0]["id"]
    labeled_key = f"{dataset_id}:{question_id}:0"  # "More parks", row 0

    def write_run(run_id: str, label_ids: list[str], fit: int) -> None:
        run_dir = (
            tmp_path / "data" / "labels" / str(dataset_id) / str(question_id) / run_id
        )
        run_dir.mkdir(parents=True)
        (run_dir / "assignments.json").write_text(
            json.dumps(
                [
                    {
                        "response_key": labeled_key,
                        "label_ids": label_ids,
                        "uncategorized": False,
                        "fit": fit,
                        "locations": ["parks"],
                        "time_context": [],
                        "actionability": "general",
                        "event_occurred": False,
                    },
                    {  # key no DB row has — must be counted, not dropped
                        "response_key": f"{dataset_id}:{question_id}:999",
                        "label_ids": [],
                        "uncategorized": True,
                    },
                ]
            ),
            encoding="utf-8",
        )

    write_run("2026-01-01T00-00-00Z_aaaaaaaa", ["old_label"], 1)
    write_run("2026-06-01T00-00-00Z_bbbbbbbb", ["new_label"], 3)

    export = client.post(f"/datasets/{dataset_id}/export")
    assert export.status_code == 200

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    rows = {
        r["response_key"]: r
        for r in pq.read_table(export_dir / "responses.parquet").to_pylist()
    }
    labeled = rows[labeled_key]
    assert labeled["label_ids"] == ["new_label"]  # newer run wins
    assert labeled["fit"] == 3
    assert labeled["locations"] == ["parks"]
    assert labeled["event_occurred"] is False
    assert labeled["respondent_key"] == f"{dataset_id}:0"

    unlabeled = rows[f"{dataset_id}:{question_id}:1"]  # "Lower rent", no run entry
    assert unlabeled["label_ids"] is None
    assert unlabeled["uncategorized"] is None
    assert unlabeled["fit"] is None

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels_runs"] == {
        str(question_id): "2026-06-01T00-00-00Z_bbbbbbbb"
    }
    assert manifest["labels_unmatched_keys"] == 1


def test_export_schema_is_v2(client):
    # Pins the schema-v2 field set: sentiment is gone (labeling v2 never
    # produces it), child_category_ids became label_ids, and the labeling v2
    # fields all exist.
    names = [f.name for f in ingest_module.RESPONSE_PARQUET_SCHEMA]
    assert names == [
        "dataset_id", "question_id", "question_label", "source_column",
        "source_row_index", "response_key", "respondent_key", "respondent_id",
        "response_text", "raw_text_original", "was_encoding_repaired",
        "is_nonanswer", "label_ids", "uncategorized", "fit", "locations",
        "time_context", "actionability", "event_occurred",
    ]


def test_encoding_repair_handles_undefined_cp1252_bytes(client, tmp_path):
    # Ground truth pulled from a real response in the San Jose survey data.
    # A curly double-quote's UTF-8 bytes are E2 80 9C (open) / E2 80 9D
    # (close). Python's cp1252 codec decodes E2/80/9C fine, but *raises* on
    # 0x9D, which is undefined in that table -- so a naive cp1252-round-trip
    # repair bails and leaves this text untouched. The real corrupting tool
    # (browser/JS windows-1252, per WHATWG) instead maps 0x9D to its raw C1
    # control codepoint, which is what actually appears in the source file.
    # Built via explicit \N{codepoint} construction below, not literal
    # characters, so nothing can silently mangle it in transit.
    mojibake_text = "taxpayer funded dollars on â€œequityâ€ programs"
    correct_text = "taxpayer funded dollars on “equity” programs"
    assert mojibake_text != correct_text

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
    table = pq.read_table(export_dir / "responses.parquet")
    row = table.to_pylist()[0]
    assert row["response_text"] == correct_text
    assert row["was_encoding_repaired"] is True
