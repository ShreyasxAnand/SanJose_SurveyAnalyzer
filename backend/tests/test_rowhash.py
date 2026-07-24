import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import rowhash
from app.db import Base
from app.models import ColumnFingerprint, Dataset, RowHash, Upload


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    yield session
    session.close()


def _make_dataset(db, name="ds", status="ingested", hashes=(), duplicates=(),
                  fingerprints=None):
    """Create a dataset + one upload + RowHash rows. `duplicates` is a set of
    row indices flagged is_duplicate; `fingerprints` an optional
    {column: fingerprint} dict."""
    ds = Dataset(
        name=name, original_filename=f"{name}.csv", original_path="x", status=status
    )
    db.add(ds)
    db.flush()
    up = Upload(
        dataset_id=ds.id,
        stored_filename=f"{name}.csv",
        stored_path="x",
        row_offset=0,
        row_count=len(hashes),
        new_row_count=len(hashes) - len(duplicates),
        duplicate_row_count=len(duplicates),
    )
    db.add(up)
    db.flush()
    for i, h in enumerate(hashes):
        db.add(
            RowHash(
                dataset_id=ds.id,
                upload_id=up.id,
                row_index=i,
                row_hash=h,
                is_duplicate=i in duplicates,
            )
        )
    for col, fp in (fingerprints or {}).items():
        db.add(
            ColumnFingerprint(
                dataset_id=ds.id, upload_id=up.id, column_name=col, fingerprint=fp
            )
        )
    db.commit()
    return ds


def test_hash_row_delimiter_prevents_boundary_collisions():
    assert rowhash.hash_row(["ab", ""]) != rowhash.hash_row(["a", "b"])
    assert rowhash.hash_row(["ab"]) != rowhash.hash_row(["ab", ""])
    # deterministic
    assert rowhash.hash_row(["x", "y"]) == rowhash.hash_row(["x", "y"])


def test_hash_dataframe_row_order_and_content():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    hashes = rowhash.hash_dataframe(df)
    assert hashes == [rowhash.hash_row(["1", "x"]), rowhash.hash_row(["2", "y"])]


def test_find_matches_none(db):
    _make_dataset(db, hashes=[rowhash.hash_row(["a"])])
    result = rowhash.find_matches(db, [rowhash.hash_row(["zzz"])])
    assert result["outcome"] == "none"
    assert result["best_dataset_id"] is None
    assert result["matches"] == []


def test_find_matches_exact_and_partial(db):
    h = [rowhash.hash_row([f"row{i}"]) for i in range(4)]
    ds = _make_dataset(db, hashes=h)

    exact = rowhash.find_matches(db, h)
    assert exact["outcome"] == "exact"
    assert exact["best_dataset_id"] == ds.id
    assert exact["matches"][0]["matched_rows"] == 4
    assert exact["matches"][0]["exact"] is True

    partial = rowhash.find_matches(db, h[:2] + [rowhash.hash_row(["new"])])
    assert partial["outcome"] == "partial"
    assert partial["matches"][0]["matched_rows"] == 2
    assert partial["matches"][0]["file_rows"] == 3
    assert partial["matches"][0]["exact"] is False


def test_find_matches_multiset_semantics(db):
    # File has the same row 3x; dataset stores it 2x -> matched is 2, not 3.
    h = rowhash.hash_row(["n/a"])
    _make_dataset(db, hashes=[h, h])
    result = rowhash.find_matches(db, [h, h, h])
    assert result["outcome"] == "partial"
    assert result["matches"][0]["matched_rows"] == 2


def test_find_matches_ignores_non_ingested_and_duplicate_rows(db):
    h = [rowhash.hash_row(["a"]), rowhash.hash_row(["b"])]
    _make_dataset(db, name="provisional", status="uploaded", hashes=h)
    result = rowhash.find_matches(db, h)
    assert result["outcome"] == "none"

    # An ingested dataset whose second row is a flagged duplicate: only the
    # non-duplicate row participates.
    _make_dataset(db, name="ingested", status="ingested", hashes=h, duplicates={1})
    result = rowhash.find_matches(db, h)
    assert result["outcome"] == "partial"
    assert result["matches"][0]["matched_rows"] == 1
    assert result["matches"][0]["dataset_rows"] == 1


def test_find_matches_chunks_large_hash_lists(db):
    # >900 distinct hashes exercises the IN() chunking.
    h = [rowhash.hash_row([f"r{i}"]) for i in range(1001)]
    ds = _make_dataset(db, hashes=h)
    result = rowhash.find_matches(db, h)
    assert result["outcome"] == "exact"
    assert result["best_dataset_id"] == ds.id
    assert result["matches"][0]["matched_rows"] == 1001


def test_split_new_rows_multiset_walk(db):
    ha, hb = rowhash.hash_row(["a"]), rowhash.hash_row(["b"])
    ds = _make_dataset(db, hashes=[ha, ha])
    stored = rowhash.stored_multiset(db, ds.id)
    flags = rowhash.split_new_rows([ha, ha, ha, hb], stored)
    # first two occurrences duplicate, third exceeds stored multiplicity, b new
    assert flags == [True, True, False, False]


def test_fingerprint_columns_order_and_boundaries():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    fps = rowhash.fingerprint_columns(df)
    assert list(fps) == ["a", "b"]  # file column order
    # row order matters
    assert fps["a"] != rowhash.fingerprint_columns(
        pd.DataFrame({"a": ["2", "1"]})
    )["a"]
    # cell boundaries can't collide ("ab","" vs "a","b")
    assert (
        rowhash.fingerprint_columns(pd.DataFrame({"c": ["ab", ""]}))["c"]
        != rowhash.fingerprint_columns(pd.DataFrame({"c": ["a", "b"]}))["c"]
    )
    # same values -> same fingerprint regardless of column name
    assert (
        rowhash.fingerprint_columns(pd.DataFrame({"c": ["1", "2"]}))["c"]
        == fps["a"]
    )


def _fps(**cols):
    """Fingerprints for literal column value lists."""
    return rowhash.fingerprint_columns(pd.DataFrame(cols))


def test_find_column_matches_dropped_column(db):
    stored = _fps(q1=["a", "b"], q2=["c", "d"], meta=["1", "2"])
    _make_dataset(db, hashes=[rowhash.hash_row(["x"])], fingerprints=stored)

    result = rowhash.find_column_matches(db, _fps(q1=["a", "b"], q2=["c", "d"]))
    assert len(result) == 1
    m = result[0]
    assert m["matched_columns"] == ["q1", "q2"]
    assert m["missing_columns"] == ["meta"]
    assert m["added_columns"] == []
    assert m["renamed_columns"] == []


def test_find_column_matches_renamed_and_added(db):
    stored = _fps(q1=["a", "b"], q2=["c", "d"], q3=["e", "f"])
    _make_dataset(db, hashes=[rowhash.hash_row(["x"])], fingerprints=stored)

    # q3 renamed to q3_new, plus a brand-new column
    result = rowhash.find_column_matches(
        db, _fps(q1=["a", "b"], q2=["c", "d"], q3_new=["e", "f"], extra=["z", "z"])
    )
    assert len(result) == 1
    m = result[0]
    assert m["matched_columns"] == ["q1", "q2"]
    assert m["renamed_columns"] == [{"stored_name": "q3", "file_name": "q3_new"}]
    assert m["added_columns"] == ["extra"]
    assert m["missing_columns"] == []


def test_find_column_matches_changed_content_reported(db):
    stored = _fps(q1=["a", "b"], q2=["c", "d"], q3=["e", "f"])
    _make_dataset(db, hashes=[rowhash.hash_row(["x"])], fingerprints=stored)

    result = rowhash.find_column_matches(
        db, _fps(q1=["a", "b"], q2=["c", "d"], q3=["EDITED", "f"])
    )
    assert len(result) == 1
    assert result[0]["matched_columns"] == ["q1", "q2"]
    assert result[0]["changed_columns"] == ["q3"]


def test_find_column_matches_threshold(db):
    # One shared column (a same-length 1..N counter, say) is not evidence.
    stored = _fps(counter=["1", "2"], q=["real answers", "here"])
    _make_dataset(db, hashes=[rowhash.hash_row(["x"])], fingerprints=stored)

    assert (
        rowhash.find_column_matches(
            db, _fps(counter=["1", "2"], other=["unrelated", "stuff"])
        )
        == []
    )
    # Two matched of four total columns is exactly half -> reported.
    stored4 = _fps(a=["1", "2"], b=["3", "4"], c=["5", "6"], d=["7", "8"])
    _make_dataset(db, name="ds4", hashes=[rowhash.hash_row(["y"])],
                  fingerprints=stored4)
    result = rowhash.find_column_matches(
        db, _fps(a=["1", "2"], b=["3", "4"], x=["9", "9"], y=["8", "8"])
    )
    assert [m["dataset_name"] for m in result] == ["ds4"]


def test_find_column_matches_ignores_non_ingested(db):
    stored = _fps(q1=["a", "b"], q2=["c", "d"])
    _make_dataset(db, status="uploaded", hashes=[rowhash.hash_row(["x"])],
                  fingerprints=stored)
    assert rowhash.find_column_matches(db, stored) == []
