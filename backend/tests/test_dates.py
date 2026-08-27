"""Response-date ingestion and period labeling.

Unit tests for app/dates.py (parsing, bucketing, ranges validation, facet
collapse) plus the ingest path end-to-end: a date-typed metadata column
stores ISO values, lands in the sidecar and the flat response_date column,
and travels through the manifest so the ask layer can derive period labels.
"""
import io
import json

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import dates
from app import db as db_module
from app import ingest as ingest_module
from app import summary as summary_module
from app.db import Base, get_db
from app.main import app


# --- parsing ---------------------------------------------------------------

def test_parse_date_accepts_the_formats_real_uploads_use():
    assert dates.parse_date("2023-09-19") == "2023-09-19"
    assert dates.parse_date("9/19/2023") == "2023-09-19"
    assert dates.parse_date("12/1/2026") == "2026-12-01"
    assert dates.parse_date("9/19/23") == "2023-09-19"
    assert dates.parse_date("2023/09/19") == "2023-09-19"
    assert dates.parse_date("19-Sep-2023") == "2023-09-19"
    # Excel datetime cells carry a time part
    assert dates.parse_date("2023-09-19 00:00:00") == "2023-09-19"


def test_parse_date_rejects_non_dates():
    for junk in ("", "  ", "not a date", "13/45/2023", "999", "Q3 2023"):
        assert dates.parse_date(junk) is None


def test_bucket_labels_sort_chronologically_as_strings():
    assert dates.bucket_label("2023-09-19", "quarter") == "2023 Q3"
    assert dates.bucket_label("2024-01-02", "quarter") == "2024 Q1"
    assert dates.bucket_label("2023-09-19", "month") == "2023-09"
    assert dates.bucket_label("2023-09-19", "year") == "2023"
    assert dates.bucket_label("2023-12-31", "quarter") == "2023 Q4"
    assert "2023 Q4" < "2024 Q1"


# --- ranges config ---------------------------------------------------------

def test_validate_ranges_config_bucket_and_ranges():
    assert dates.validate_ranges_config(
        {"mode": "bucket", "granularity": "quarter"}
    ) == {"mode": "bucket", "granularity": "quarter"}
    cfg = dates.validate_ranges_config(
        {"mode": "ranges", "ranges": [
            {"label": "Before", "start": "2023-01-01", "end": "2023-06-30"},
            {"label": "After", "start": "2023-07-01", "end": "2023-12-31"},
        ]}
    )
    assert [r["label"] for r in cfg["ranges"]] == ["Before", "After"]


@pytest.mark.parametrize("bad", [
    {"mode": "bucket", "granularity": "decade"},
    {"mode": "ranges", "ranges": []},
    {"mode": "ranges", "ranges": [{"label": "", "start": "2023-01-01",
                                   "end": "2023-02-01"}]},
    {"mode": "ranges", "ranges": [{"label": "X", "start": "2023-02-01",
                                   "end": "2023-01-01"}]},
    # overlapping spans
    {"mode": "ranges", "ranges": [
        {"label": "A", "start": "2023-01-01", "end": "2023-06-30"},
        {"label": "B", "start": "2023-06-30", "end": "2023-12-31"}]},
    {"mode": "ranges", "ranges": [{"label": dates.UNLABELED,
                                   "start": "2023-01-01",
                                   "end": "2023-02-01"}]},
    {"mode": "wat"},
])
def test_validate_ranges_config_rejects_malformed(bad):
    with pytest.raises(ValueError):
        dates.validate_ranges_config(bad)


def test_period_label_ranges_and_unlabeled():
    cfg = dates.validate_ranges_config(
        {"mode": "ranges", "ranges": [
            {"label": "Wave 1", "start": "2023-07-01", "end": "2023-09-30"},
        ]}
    )
    assert dates.period_label("2023-09-19", cfg) == "Wave 1"
    assert dates.period_label("2024-01-01", cfg) == dates.UNLABELED


# --- facet collapse (what the ask layer calls) -----------------------------

def _demo_state():
    values = {"Period": [("2023-09-19", 2), ("2023-08-01", 1),
                         ("2024-01-05", 3)],
              "District": [("D3", 4)]}
    members = {"Period": {"2023-09-19": {"1:5:0", "1:5:1"},
                          "2023-08-01": {"1:5:2"},
                          "2024-01-05": {"1:5:3"}},
               "District": {"D3": {"1:5:0"}}}
    coded = {"Period": {"1:5:0", "1:5:1", "1:5:2", "1:5:3"}}
    respondents = {"Period": {"2023-09-19": {"1:0", "1:1"},
                              "2023-08-01": {"1:2"},
                              "2024-01-05": {"1:3"}}}
    return values, members, coded, respondents


def test_apply_period_labels_collapses_dates_into_quarters():
    values, members, coded, respondents = _demo_state()
    dates.apply_period_labels({"Period"}, None, values, members, coded,
                              respondents)
    # chronological, not by count — and Q3 merges its two dates
    assert values["Period"] == [("2023 Q3", 3), ("2024 Q1", 3)]
    assert members["Period"]["2023 Q3"] == {"1:5:0", "1:5:1", "1:5:2"}
    assert respondents["Period"]["2024 Q1"] == {"1:3"}
    # untouched: non-date fields and the per-field coded aggregate
    assert values["District"] == [("D3", 4)]
    assert coded["Period"] == {"1:5:0", "1:5:1", "1:5:2", "1:5:3"}


def test_apply_period_labels_named_ranges_follow_config_order():
    values, members, coded, respondents = _demo_state()
    cfg = {"mode": "ranges", "ranges": [
        {"label": "Summer", "start": "2023-07-01", "end": "2023-09-30"},
    ]}
    dates.apply_period_labels({"Period"}, cfg, values, members, coded,
                              respondents)
    assert values["Period"] == [("Summer", 3), (dates.UNLABELED, 3)]
    assert members["Period"][dates.UNLABELED] == {"1:5:3"}


# --- ingest end-to-end -----------------------------------------------------

DATE_CSV = (
    "respondent_id,better_city,stopdate\n"
    "1,More parks,9/19/2023\n"
    "2,Lower rent,12/28/2023\n"
    "3,Cleaner streets,not a date\n"
    "4,More housing,\n"
)

DATE_BODY = {
    "respondent_id_column": "respondent_id",
    "questions": [
        {"column": "better_city", "label": "What would make the city better?"},
    ],
    "metadata_columns": [
        {"column": "stopdate", "label": "Period", "value_type": "date"},
    ],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
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


def _upload(client, csv_text, filename="survey.csv"):
    files = {"file": (filename, io.BytesIO(csv_text.encode()), "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 200
    return resp.json()


def test_date_column_stores_iso_and_defaults_to_quarter_config(client):
    ds_id = _upload(client, DATE_CSV)["dataset_id"]
    body = client.post(f"/api/datasets/{ds_id}/columns", json=DATE_BODY).json()

    col = body["metadata_columns"][0]
    assert col["value_type"] == "date"
    # values stored canonically; the junk and blank cells stored nothing
    stored = {v["value"]: v["n_respondents"] for v in col["values"]}
    assert stored == {"2023-09-19": 1, "2023-12-28": 1}
    assert col["n_distinct"] == 2
    # a newly selected date column gets quarter bucketing with zero setup
    assert body["date_ranges"] == {"mode": "bucket", "granularity": "quarter"}


def test_date_lands_in_flat_export_and_manifest(client, tmp_path):
    ds_id = _upload(client, DATE_CSV)["dataset_id"]
    client.post(f"/api/datasets/{ds_id}/columns", json=DATE_BODY)

    export_dir = tmp_path / "data" / "exports" / str(ds_id)
    table = pq.read_table(export_dir / "responses.parquet",
                          columns=["source_row_index", "response_date"])
    by_row = dict(zip(table.column("source_row_index").to_pylist(),
                      table.column("response_date").to_pylist()))
    assert by_row[0] == "2023-09-19"
    assert by_row[1] == "2023-12-28"
    assert by_row[2] is None          # unparseable = missing, not junk
    assert by_row[3] is None

    side = pq.read_table(export_dir / "respondents.parquet")
    fields = set(zip(side.column("field").to_pylist(),
                     side.column("value").to_pylist()))
    assert ("Period", "2023-09-19") in fields

    manifest = json.loads((export_dir / "manifest.json").read_text("utf-8"))
    assert manifest["metadata_columns"] == [
        {"source_column": "stopdate", "label": "Period",
         "value_type": "date", "n_distinct": 2}
    ]
    assert manifest["date_ranges"] == {"mode": "bucket",
                                       "granularity": "quarter"}


def test_explicit_ranges_config_wins_over_the_default(client):
    ds_id = _upload(client, DATE_CSV)["dataset_id"]
    body = client.post(
        f"/api/datasets/{ds_id}/columns",
        json={**DATE_BODY,
              "date_ranges": {"mode": "ranges", "ranges": [
                  {"label": "Fall 2023", "start": "2023-09-01",
                   "end": "2023-11-30"}]}},
    ).json()
    assert body["date_ranges"]["mode"] == "ranges"
    assert body["date_ranges"]["ranges"][0]["label"] == "Fall 2023"


def test_malformed_ranges_config_is_a_422(client):
    ds_id = _upload(client, DATE_CSV)["dataset_id"]
    resp = client.post(
        f"/api/datasets/{ds_id}/columns",
        json={**DATE_BODY,
              "date_ranges": {"mode": "ranges", "ranges": [
                  {"label": "A", "start": "2023-01-01", "end": "2023-06-30"},
                  {"label": "B", "start": "2023-05-01", "end": "2023-12-31"},
              ]}},
    )
    assert resp.status_code == 422


def test_a_date_column_with_no_parseable_dates_is_a_400(client):
    csv = (
        "respondent_id,better_city,stopdate\n"
        "1,More parks,banana\n"
        "2,Lower rent,pear\n"
    )
    ds_id = _upload(client, csv)["dataset_id"]
    resp = client.post(f"/api/datasets/{ds_id}/columns", json=DATE_BODY)
    assert resp.status_code == 400
    assert "no parseable dates" in resp.json()["detail"]


def test_date_ranges_editable_via_patch_without_reingest(client):
    ds_id = _upload(client, DATE_CSV)["dataset_id"]
    client.post(f"/api/datasets/{ds_id}/columns", json=DATE_BODY)
    body = client.patch(
        f"/api/datasets/{ds_id}",
        json={"date_ranges": {"mode": "bucket", "granularity": "year"}},
    ).json()
    assert body["date_ranges"] == {"mode": "bucket", "granularity": "year"}
    # the export manifest (what the ask layer reads) follows immediately
    manifest = json.loads(
        (ingest_module.EXPORTS_DIR / str(ds_id) / "manifest.json")
        .read_text("utf-8")
    )
    assert manifest["date_ranges"] == {"mode": "bucket", "granularity": "year"}
