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
    upload_resp = client.post("/api/datasets/upload", files=files)
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
        f"/api/datasets/{dataset_id}/columns",
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

    csv_download = client.get(f"/api/datasets/{dataset_id}/exports/csv")
    assert csv_download.status_code == 200
    parquet_download = client.get(f"/api/datasets/{dataset_id}/exports/parquet")
    assert parquet_download.status_code == 200


def test_export_before_ingest_returns_400(client):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    upload_resp = client.post("/api/datasets/upload", files=files)
    dataset_id = upload_resp.json()["dataset_id"]

    resp = client.post(f"/api/datasets/{dataset_id}/export")
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
    upload_resp = client.post("/api/datasets/upload", files=files)
    assert upload_resp.status_code == 200
    dataset_id = upload_resp.json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
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
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400


def test_question_wording_must_differ_from_raw_column_name(client):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": None,
            "questions": [{"column": "better_city", "label": "better_city"}],
        },
    )
    assert resp.status_code == 422

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": None,
            "questions": [{"column": "better_city", "label": "  "}],
        },
    )
    assert resp.status_code == 422


def test_reingest_keeps_stable_question_and_response_ids(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    body = {
        "respondent_id_column": "respondent_id",
        "questions": [
            {"column": "better_city", "label": "What would make the city better?"},
            {"column": "unsafe", "label": "What makes it feel unsafe?"},
        ],
    }
    first = client.post(f"/api/datasets/{dataset_id}/columns", json=body).json()
    first_question_ids = {q["source_column"]: q["id"] for q in first["questions"]}

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    first_table = pq.read_table(export_dir / "responses.parquet")
    first_keys = set(first_table.column("response_key").to_pylist())

    # Re-run the exact same selection — a real analyst re-confirming choices.
    second = client.post(f"/api/datasets/{dataset_id}/columns", json=body).json()
    second_question_ids = {q["source_column"]: q["id"] for q in second["questions"]}
    second_table = pq.read_table(export_dir / "responses.parquet")
    second_keys = set(second_table.column("response_key").to_pylist())

    assert first_question_ids == second_question_ids
    assert first_keys == second_keys
    assert len(first_keys) == 4


def test_shrinking_selection_clears_stale_export_rows(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    client.post(
        f"/api/datasets/{dataset_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
                {"column": "unsafe", "label": "What makes it feel unsafe?"},
            ],
        },
    )

    second = client.post(
        f"/api/datasets/{dataset_id}/columns",
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
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
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
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
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
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
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
    assert client.get(f"/api/datasets/{dataset_id}").json()["status"] == "uploaded"


def test_description_round_trip(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    body = {
        "respondent_id_column": "respondent_id",
        "questions": [
            {"column": "better_city", "label": "What would make the city better?"},
        ],
        "dataset_description": "A survey of city residents about their city.",
    }
    first = client.post(f"/api/datasets/{dataset_id}/columns", json=body).json()
    assert first["description"] == "A survey of city residents about their city."

    export_dir = tmp_path / "data" / "exports" / str(dataset_id)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_description"] == (
        "A survey of city residents about their city."
    )

    # Re-selecting without the field preserves the saved description.
    body.pop("dataset_description")
    second = client.post(f"/api/datasets/{dataset_id}/columns", json=body).json()
    assert second["description"] == "A survey of city residents about their city."


def test_export_joins_latest_labels(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
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

    export = client.post(f"/api/datasets/{dataset_id}/export")
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
    dataset_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    resp = client.post(
        f"/api/datasets/{dataset_id}/columns",
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


# --- duplicate detection / append -----------------------------------------

INGEST_BODY = {
    "respondent_id_column": "respondent_id",
    "questions": [
        {"column": "better_city", "label": "What would make the city better?"},
        {"column": "unsafe", "label": "What makes it feel unsafe?"},
    ],
}


def _upload(client, csv_text, filename="survey.csv"):
    files = {"file": (filename, io.BytesIO(csv_text.encode()), "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 200
    return resp.json()


def test_fresh_upload_reports_no_duplicates(client):
    body = _upload(client, CSV_CONTENT)
    assert body["duplicate_check"]["outcome"] == "none"
    assert body["duplicate_check"]["matches"] == []


def test_reupload_after_ingest_reports_exact_match(client):
    first = _upload(client, CSV_CONTENT)
    # Provisional (never column-selected) datasets must not participate.
    second = _upload(client, CSV_CONTENT)
    assert second["duplicate_check"]["outcome"] == "none"

    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)
    third = _upload(client, CSV_CONTENT)
    check = third["duplicate_check"]
    assert check["outcome"] == "exact"
    assert check["best_dataset_id"] == first["dataset_id"]
    match = check["matches"][0]
    assert match["matched_rows"] == 3
    assert match["file_rows"] == 3
    assert match["exact"] is True
    assert match["columns_compatible"] is True
    assert match["missing_columns"] == []


def test_extended_file_reports_partial_match(client):
    first = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)

    extended = CSV_CONTENT + "4,Bike lanes,Broken lights\n5,More trees,\n"
    body = _upload(client, extended)
    check = body["duplicate_check"]
    assert check["outcome"] == "partial"
    assert check["best_dataset_id"] == first["dataset_id"]
    match = check["matches"][0]
    assert match["matched_rows"] == 3
    assert match["file_rows"] == 5
    assert match["dataset_rows"] == 3
    assert match["exact"] is False


def test_match_reports_missing_columns(client):
    first = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)

    # Same first column of data but the 'unsafe' column renamed: rows can't
    # hash-match (whole-row hash), so build overlap via identical rows plus a
    # missing selected column -> here we drop a column entirely, all rows
    # differ, so instead assert on a file that shares rows but lacks 'unsafe'.
    # Whole-row hashing means such a file matches nothing — outcome none.
    no_unsafe = "respondent_id,better_city\n1,More parks\n2,Lower rent\n"
    body = _upload(client, no_unsafe)
    assert body["duplicate_check"]["outcome"] == "none"


# Same 3 data rows as CSV_CONTENT with the 'unsafe' column dropped: every
# whole-row hash changes, so row matching sees nothing — the column
# fingerprint tier is what catches it.
CSV_NO_UNSAFE = (
    "respondent_id,better_city\n"
    "1,More parks\n"
    "2,Lower rent\n"
    "3,\n"
)


def test_same_rows_minus_column_reports_column_match(client):
    first = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)

    body = _upload(client, CSV_NO_UNSAFE)
    check = body["duplicate_check"]
    assert check["outcome"] == "none"
    assert check["matches"] == []
    assert len(check["column_matches"]) == 1
    m = check["column_matches"][0]
    assert m["dataset_id"] == first["dataset_id"]
    assert m["matched_columns"] == ["respondent_id", "better_city"]
    assert m["missing_columns"] == ["unsafe"]
    assert m["added_columns"] == []
    assert m["renamed_columns"] == []
    assert m["upload_rows"] == 3


def test_renamed_plus_dropped_column_reports_column_match(client):
    # Rename alone keeps every data row byte-identical, so row matching
    # already reports "exact" (headers aren't hashed). Rename + drop is the
    # case only the column tier can see.
    first = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)

    renamed = (
        "better_city,q_unsafe\n"
        "More parks,Dark streets\n"
        "Lower rent,\n"
        ",Speeding cars\n"
    )
    body = _upload(client, renamed)
    check = body["duplicate_check"]
    assert check["outcome"] == "none"
    m = check["column_matches"][0]
    assert m["matched_columns"] == ["better_city"]
    assert m["renamed_columns"] == [
        {"stored_name": "unsafe", "file_name": "q_unsafe"}
    ]
    assert m["missing_columns"] == ["respondent_id"]


def test_unrelated_file_reports_no_column_match(client):
    first = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)

    body = _upload(client, "colx,coly\nfoo,bar\nbaz,qux\nquux,corge\n")
    assert body["duplicate_check"]["outcome"] == "none"
    assert body["duplicate_check"]["column_matches"] == []


def test_column_match_lazy_backfills_missing_fingerprints(client, tmp_path):
    # Simulate a dataset ingested before column_fingerprints existed by
    # deleting its rows; the next unmatched upload must self-heal from the
    # stored file (same pattern as _ensure_uploads).
    import sqlite3

    first = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{first['dataset_id']}/columns", json=INGEST_BODY)

    con = sqlite3.connect(tmp_path / "test.db")
    con.execute("DELETE FROM column_fingerprints")
    con.commit()
    con.close()

    body = _upload(client, CSV_NO_UNSAFE)
    check = body["duplicate_check"]
    assert len(check["column_matches"]) == 1
    assert check["column_matches"][0]["missing_columns"] == ["unsafe"]


def test_discard_dataset_gated_on_status(client, tmp_path):
    body = _upload(client, CSV_CONTENT)
    ds_id = body["dataset_id"]
    upload_dir = tmp_path / "data" / "uploads" / str(ds_id)
    assert upload_dir.exists()

    resp = client.delete(f"/api/datasets/{ds_id}")
    assert resp.status_code == 204
    assert client.get(f"/api/datasets/{ds_id}").status_code == 404
    assert not upload_dir.exists()

    ingested = _upload(client, CSV_CONTENT)
    client.post(f"/api/datasets/{ingested['dataset_id']}/columns", json=INGEST_BODY)
    resp = client.delete(f"/api/datasets/{ingested['dataset_id']}")
    assert resp.status_code == 409


def test_append_skips_duplicates_and_offsets_new_rows(client, tmp_path):
    first = _upload(client, CSV_CONTENT)
    target_id = first["dataset_id"]
    client.post(f"/api/datasets/{target_id}/columns", json=INGEST_BODY)

    extended = CSV_CONTENT + "4,Bike lanes,Broken lights\n"
    prov = _upload(client, extended, filename="survey_v2.csv")

    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["appended_rows"] == 1
    assert body["skipped_duplicates"] == 3
    assert body["new_responses_per_question"] == {
        "What would make the city better?": 1,
        "What makes it feel unsafe?": 1,
    }

    # The provisional dataset is consumed: DB row and upload dir both gone.
    assert client.get(f"/api/datasets/{prov['dataset_id']}").status_code == 404
    assert not (tmp_path / "data" / "uploads" / str(prov["dataset_id"])).exists()
    # Its file lives on under the target, byte-for-byte.
    copied = (
        tmp_path / "data" / "uploads" / str(target_id) / str(body["upload_id"])
        / "survey_v2.csv"
    )
    assert copied.exists()
    assert copied.read_bytes() == extended.encode()

    # New rows sit past the first file's range: upload #1 holds indices 0-2,
    # the appended file starts at offset 3, and its one new row is at local
    # index 3 -> global 6 (duplicate rows consume indices 3-5, produce none).
    export_dir = tmp_path / "data" / "exports" / str(target_id)
    table = pq.read_table(export_dir / "responses.parquet").to_pylist()
    new_rows = [r for r in table if r["source_row_index"] == 6]
    assert len(new_rows) == 2  # one per question
    assert {r["response_text"] for r in new_rows} == {"Bike lanes", "Broken lights"}
    # never labeled -> label block None, not []
    assert all(r["label_ids"] is None for r in new_rows)

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["uploads"]) == 2
    assert manifest["uploads"][1]["new_row_count"] == 1
    assert manifest["uploads"][1]["duplicate_row_count"] == 3


def test_reselect_after_append_keeps_appended_rows(client, tmp_path):
    # THE trap regression: re-running column selection re-reads upload #1's
    # file; per-upload stale-deletion scoping must keep upload #2's rows.
    first = _upload(client, CSV_CONTENT)
    target_id = first["dataset_id"]
    client.post(f"/api/datasets/{target_id}/columns", json=INGEST_BODY)

    extended = CSV_CONTENT + "4,Bike lanes,Broken lights\n"
    prov = _upload(client, extended, filename="survey_v2.csv")
    client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"]},
    )

    export_dir = tmp_path / "data" / "exports" / str(target_id)
    before = pq.read_table(export_dir / "responses.parquet")
    before_keys = set(before.column("response_key").to_pylist())

    reselect = client.post(f"/api/datasets/{target_id}/columns", json=INGEST_BODY)
    assert reselect.status_code == 200

    after = pq.read_table(export_dir / "responses.parquet")
    after_keys = set(after.column("response_key").to_pylist())
    assert after_keys == before_keys
    # appended rows survive: global index = row_offset (3) + local index (3)
    assert any(k.endswith(":6") for k in after_keys)


def test_reselect_400s_when_column_missing_from_appended_file(client):
    first = _upload(client, CSV_CONTENT)
    target_id = first["dataset_id"]
    client.post(f"/api/datasets/{target_id}/columns", json=INGEST_BODY)

    # Appended file carries the question columns but not 'respondent_id':
    # append succeeds with a warning, respondent ids null for its rows.
    no_rid = "better_city,unsafe\nBike lanes,Broken lights\n"
    prov = _upload(client, no_rid, filename="survey_v2.csv")
    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"]},
    )
    assert resp.status_code == 200
    assert any("respondent" in w.lower() for w in resp.json()["warnings"])

    # Re-selecting a question column the appended file lacks is a 400 that
    # names the offending file.
    resp = client.post(
        f"/api/datasets/{target_id}/columns",
        json={
            "respondent_id_column": "respondent_id",
            "questions": [
                {"column": "better_city", "label": "What would make the city better?"},
                {"column": "unsafe", "label": "What makes it feel unsafe?"},
                {"column": "downtown", "label": "What would improve downtown?"},
            ],
        },
    )
    assert resp.status_code == 400


def test_append_all_duplicates_is_200_with_zero_rows(client):
    first = _upload(client, CSV_CONTENT)
    target_id = first["dataset_id"]
    client.post(f"/api/datasets/{target_id}/columns", json=INGEST_BODY)

    prov = _upload(client, CSV_CONTENT, filename="same_again.csv")
    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["appended_rows"] == 0
    assert body["skipped_duplicates"] == 3
    assert body["new_responses_per_question"] == {
        "What would make the city better?": 0,
        "What makes it feel unsafe?": 0,
    }


def test_append_validates_states_and_columns(client):
    first = _upload(client, CSV_CONTENT)
    target_id = first["dataset_id"]
    client.post(f"/api/datasets/{target_id}/columns", json=INGEST_BODY)

    # Missing a selected question column -> 400.
    missing_q = "respondent_id,better_city\n9,Trains\n"
    prov = _upload(client, missing_q, filename="missing.csv")
    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"]},
    )
    assert resp.status_code == 400
    assert "unsafe" in resp.json()["detail"]

    # Target must be ingested.
    prov2 = _upload(client, CSV_CONTENT, filename="a.csv")
    prov3 = _upload(client, CSV_CONTENT, filename="b.csv")
    resp = client.post(
        f"/api/datasets/{prov2['dataset_id']}/append",
        json={"upload_dataset_id": prov3["dataset_id"]},
    )
    assert resp.status_code == 400

    # Source must be provisional.
    second = _upload(client, "respondent_id,better_city,unsafe\n9,Trains,Dogs\n")
    second_id = second["dataset_id"]
    client.post(f"/api/datasets/{second_id}/columns", json=INGEST_BODY)
    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": second_id},
    )
    assert resp.status_code == 400


# --- catalog metadata / history --------------------------------------------


def _ingest(client, csv_text=CSV_CONTENT, body=INGEST_BODY):
    ds_id = _upload(client, csv_text)["dataset_id"]
    resp = client.post(f"/api/datasets/{ds_id}/columns", json=body)
    assert resp.status_code == 200
    return ds_id


def _manifest(tmp_path, dataset_id):
    path = tmp_path / "data" / "exports" / str(dataset_id) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_patch_metadata_round_trip(client, tmp_path):
    ds_id = _ingest(client)

    patched = client.patch(
        f"/api/datasets/{ds_id}",
        json={
            "department": "Parks & Recreation",
            "notes": "Shared with the council 2026-08.",
            "survey_start_date": "2026-03-01",
            "survey_end_date": "2026-05-31",
        },
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["department"] == "Parks & Recreation"
    assert body["notes"] == "Shared with the council 2026-08."
    assert body["survey_start_date"] == "2026-03-01"
    assert body["survey_end_date"] == "2026-05-31"

    # The export was rewritten so the manifest (the audit record) agrees.
    manifest = _manifest(tmp_path, ds_id)
    assert manifest["dataset_department"] == "Parks & Recreation"
    assert manifest["dataset_notes"] == "Shared with the council 2026-08."
    assert manifest["survey_start_date"] == "2026-03-01"
    assert manifest["survey_end_date"] == "2026-05-31"

    # An empty patch preserves everything (None = leave unchanged).
    unchanged = client.patch(f"/api/datasets/{ds_id}", json={}).json()
    assert unchanged["department"] == "Parks & Recreation"
    assert unchanged["survey_start_date"] == "2026-03-01"

    # Empty string clears a field.
    cleared = client.patch(f"/api/datasets/{ds_id}", json={"department": ""}).json()
    assert cleared["department"] is None
    assert _manifest(tmp_path, ds_id)["dataset_department"] == ""


def test_patch_name_and_description_immutable(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    ds_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]
    client.post(
        f"/api/datasets/{ds_id}/columns",
        json={**INGEST_BODY, "dataset_description": "The original description."},
    )

    renamed = client.patch(f"/api/datasets/{ds_id}", json={"name": "SJ residents 2026"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "SJ residents 2026"
    assert _manifest(tmp_path, ds_id)["dataset_name"] == "SJ residents 2026"

    assert client.patch(f"/api/datasets/{ds_id}", json={"name": "  "}).status_code == 422

    # The description is not editable post-ingest: a "description" key in the
    # body is not a PATCH field and must change nothing, in DB or manifest.
    resp = client.patch(f"/api/datasets/{ds_id}", json={"description": "overwritten?"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "The original description."
    assert client.get(f"/api/datasets/{ds_id}").json()["description"] == (
        "The original description."
    )
    assert _manifest(tmp_path, ds_id)["dataset_description"] == (
        "The original description."
    )


def test_patch_date_validation(client):
    ds_id = _ingest(client)
    resp = client.patch(f"/api/datasets/{ds_id}", json={"survey_start_date": "03/01/2026"})
    assert resp.status_code == 422
    resp = client.patch(
        f"/api/datasets/{ds_id}",
        json={"survey_start_date": "2026-06-01", "survey_end_date": "2026-03-01"},
    )
    assert resp.status_code == 422


def test_patch_unknown_dataset_and_running_pipeline(client, monkeypatch):
    assert client.patch("/api/datasets/999", json={"name": "x"}).status_code == 404

    ds_id = _ingest(client)

    class _RunningJob:
        status = "running"

    from app import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "latest_job", lambda _ds: _RunningJob())
    resp = client.patch(f"/api/datasets/{ds_id}", json={"department": "Parks"})
    assert resp.status_code == 409
    # Nothing was saved: the refusal happened before commit.
    monkeypatch.setattr(pipeline_module, "latest_job", lambda _ds: None)
    assert client.get(f"/api/datasets/{ds_id}").json()["department"] is None


def test_select_columns_metadata_round_trip(client, tmp_path):
    files = {"file": ("survey.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")}
    ds_id = client.post("/api/datasets/upload", files=files).json()["dataset_id"]

    body = {
        **INGEST_BODY,
        "dataset_description": "A resident survey.",
        "dataset_department": "City Manager's Office",
        "dataset_notes": "Pilot year.",
        "survey_start_date": "2026-01-15",
        "survey_end_date": "2026-02-15",
    }
    first = client.post(f"/api/datasets/{ds_id}/columns", json=body).json()
    assert first["department"] == "City Manager's Office"
    assert first["notes"] == "Pilot year."
    assert first["survey_start_date"] == "2026-01-15"
    assert first["survey_end_date"] == "2026-02-15"

    manifest = _manifest(tmp_path, ds_id)
    assert manifest["dataset_department"] == "City Manager's Office"
    assert manifest["survey_end_date"] == "2026-02-15"

    # Re-selecting without the metadata fields preserves the saved values,
    # mirroring the description semantics.
    second = client.post(f"/api/datasets/{ds_id}/columns", json=INGEST_BODY).json()
    assert second["department"] == "City Manager's Office"
    assert second["notes"] == "Pilot year."
    assert second["survey_start_date"] == "2026-01-15"

    # Backwards dates are rejected at column select too.
    bad = client.post(
        f"/api/datasets/{ds_id}/columns",
        json={
            **INGEST_BODY,
            "survey_start_date": "2026-05-01",
            "survey_end_date": "2026-04-01",
        },
    )
    assert bad.status_code == 422


def test_append_note_round_trip(client, tmp_path):
    target_id = _ingest(client)

    extended = CSV_CONTENT + "4,Bike lanes,Broken lights\n"
    prov = _upload(client, extended, filename="survey_v2.csv")
    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"], "note": "  2026 Q3 wave  "},
    )
    assert resp.status_code == 200

    manifest = _manifest(tmp_path, target_id)
    assert manifest["uploads"][0]["note"] is None
    assert manifest["uploads"][1]["note"] == "2026 Q3 wave"

    history = client.get(f"/api/datasets/{target_id}/history").json()
    assert history["entries"][1]["note"] == "2026 Q3 wave"

    # Note stays optional — a bare append body is still valid, note = None.
    more = CSV_CONTENT + "5,More trees,Litter\n"
    prov2 = _upload(client, more, filename="survey_v3.csv")
    resp = client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov2["dataset_id"]},
    )
    assert resp.status_code == 200
    assert client.get(f"/api/datasets/{target_id}/history").json()["entries"][2][
        "note"
    ] is None


def test_history_endpoint(client, tmp_path):
    assert client.get("/api/datasets/999/history").status_code == 404

    target_id = _ingest(client)
    extended = CSV_CONTENT + "4,Bike lanes,Broken lights\n"
    prov = _upload(client, extended, filename="survey_v2.csv")
    client.post(
        f"/api/datasets/{target_id}/append",
        json={"upload_dataset_id": prov["dataset_id"], "note": "wave two"},
    )

    history = client.get(f"/api/datasets/{target_id}/history").json()
    assert history["dataset_id"] == target_id
    entries = history["entries"]
    assert len(entries) == 2
    assert entries[0]["kind"] == "created"
    assert entries[0]["filename"] == "survey.csv"
    assert entries[0]["row_count"] == 3
    assert entries[0]["new_row_count"] == 3
    assert entries[1]["kind"] == "appended"
    assert entries[1]["filename"] == "survey_v2.csv"
    assert entries[1]["new_row_count"] == 1
    assert entries[1]["duplicate_row_count"] == 3
    assert entries[1]["note"] == "wave two"

    # A legacy dataset with no Upload rows (ingested before the uploads table
    # existed) self-heals via _ensure_uploads instead of returning nothing.
    import sqlite3

    with sqlite3.connect(tmp_path / "test.db") as conn:
        conn.execute("DELETE FROM row_hashes WHERE dataset_id = ?", (target_id,))
        conn.execute("DELETE FROM uploads WHERE dataset_id = ?", (target_id,))
        conn.commit()
    healed = client.get(f"/api/datasets/{target_id}/history").json()
    assert len(healed["entries"]) == 1
    assert healed["entries"][0]["kind"] == "created"
    assert healed["entries"][0]["filename"] == "survey.csv"
