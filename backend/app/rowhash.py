"""Whole-row content hashing and duplicate-upload detection.

Every raw row of every uploaded file is SHA-256 hashed over its cells in file
column order — before mojibake repair and independent of which columns get
selected as questions, so changing the selection never changes match results.
Hashes are stored in the `row_hashes` table; at upload time a new file's
hashes are checked against every ingested dataset's to classify the upload as
brand new, an exact re-upload, or a partial overlap that can be appended.

Matching is MULTISET matching: a file containing "n/a" rows 40 times matches
at most the 40 such rows already stored, never all 40 against one. The
delimiter scheme mirrors locations.corpus_fingerprint — a separator byte
after every cell so ["ab", ""] and ["a", "b"] cannot collide.

Known limitation, by design: the hash covers the raw row exactly as parsed,
so the same data re-exported with reordered columns, added columns, or
different Excel formatting will not match. That is the right behaviour for
catching accidental re-uploads and extended versions of the same file; a
semantically-reshaped file is genuinely new data.

A second, column-level tier catches the commonest way whole-row hashing goes
blind: the same file re-uploaded with a column added, dropped, or renamed
changes every row hash but leaves the other columns' full value sequences
byte-identical. `fingerprint_columns` hashes each column's cells in row
order (one hash per column per upload, stored in `column_fingerprints`);
`find_column_matches` compares a new file's fingerprints against every
ingested dataset's uploads when row matching found nothing. This tier only
diagnoses — it never enables append (row identity still differs), it exists
so the analyst is warned before re-running the pipeline on data they already
processed.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import ColumnFingerprint, Dataset, QuestionColumn, RowHash, Upload

# SQLite's default parameter limit is 999 — chunk IN() lists below it.
_IN_CHUNK = 900


def hash_row(cells: Iterable[str]) -> str:
    h = hashlib.sha256()
    for c in cells:
        h.update(str(c).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> list[str]:
    """One hash per row of a post-_parse_dataframe frame (every cell already
    str, blanks "")."""
    return [hash_row(row) for row in df.itertuples(index=False, name=None)]


def fingerprint_columns(df: pd.DataFrame) -> dict[str, str]:
    """One SHA-256 per column over its cells in row order, same delimiter
    scheme as hash_row. Equal fingerprints mean identical value sequences —
    including length, since every cell contributes a terminated chunk.
    Returned in file column order."""
    out: dict[str, str] = {}
    for col in df.columns:
        h = hashlib.sha256()
        for c in df[col]:
            h.update(str(c).encode("utf-8", "replace"))
            h.update(b"\x00")
        out[str(col)] = h.hexdigest()
    return out


# Report a column-level match only when at least this many columns are
# identical AND they cover at least half the larger column set — one shared
# column is too weak (a 1..N counter column of the same length matches
# between unrelated files of equal row count).
_MIN_MATCHED_COLUMNS = 2


def find_column_matches(db: Session, file_fingerprints: dict[str, str]) -> list[dict]:
    """Compare a new file's column fingerprints against every ingested
    dataset's uploads. Meant to run only when row-level matching found
    nothing — row evidence is strictly stronger.

    Per candidate upload: columns identical under the same name, identical
    under a different name (renames), present only in the stored file
    (missing), present only in the new file (added), or same-named with
    different content (changed). Reported when matched + renamed clears
    _MIN_MATCHED_COLUMNS and covers half the larger column set. One entry
    per dataset — its best-matching upload — best dataset first."""
    rows = (
        db.query(ColumnFingerprint, Upload, Dataset)
        .join(Upload, Upload.id == ColumnFingerprint.upload_id)
        .join(Dataset, Dataset.id == Upload.dataset_id)
        .filter(Dataset.status == "ingested")
        .all()
    )
    by_upload: dict[int, dict] = {}
    for fp, upload, dataset in rows:
        entry = by_upload.setdefault(
            upload.id,
            {"upload": upload, "dataset": dataset, "fingerprints": {}},
        )
        entry["fingerprints"][fp.column_name] = fp.fingerprint

    candidates: list[dict] = []
    for entry in by_upload.values():
        stored = entry["fingerprints"]
        matched = [
            c
            for c, fp in file_fingerprints.items()
            if c in stored and stored[c] == fp
        ]
        changed = [
            c
            for c, fp in file_fingerprints.items()
            if c in stored and stored[c] != fp
        ]
        # Rename detection: identical content under a different name. Multiset
        # pairing (leftovers only) so two all-blank columns of equal length —
        # which share a fingerprint — pair off one-to-one instead of double-
        # matching. Insertion order keeps it deterministic.
        matched_set = set(matched)
        stored_left = defaultdict(list)
        for c, fp in stored.items():
            if c not in matched_set and c not in file_fingerprints:
                stored_left[fp].append(c)
        renamed: list[dict] = []
        for c, fp in file_fingerprints.items():
            if c in stored or not stored_left.get(fp):
                continue
            renamed.append({"stored_name": stored_left[fp].pop(0), "file_name": c})
        renamed_stored = {r["stored_name"] for r in renamed}
        renamed_file = {r["file_name"] for r in renamed}
        missing = [
            c for c in stored if c not in matched_set and c not in renamed_stored
            and c not in file_fingerprints
        ]
        added = [
            c for c in file_fingerprints
            if c not in stored and c not in renamed_file
        ]

        n_matched = len(matched) + len(renamed)
        n_cols = max(len(file_fingerprints), len(stored))
        if n_matched < _MIN_MATCHED_COLUMNS or n_matched * 2 < n_cols:
            continue
        candidates.append(
            {
                "dataset_id": entry["dataset"].id,
                "dataset_name": entry["dataset"].name,
                "upload_filename": entry["upload"].stored_filename,
                "upload_rows": entry["upload"].row_count,
                "matched_columns": matched,
                "renamed_columns": renamed,
                "missing_columns": missing,
                "added_columns": added,
                "changed_columns": changed,
            }
        )

    # Best upload per dataset, best dataset first.
    best_by_dataset: dict[int, dict] = {}
    for c in candidates:
        prior = best_by_dataset.get(c["dataset_id"])
        score = len(c["matched_columns"]) + len(c["renamed_columns"])
        if prior is None or score > len(prior["matched_columns"]) + len(
            prior["renamed_columns"]
        ):
            best_by_dataset[c["dataset_id"]] = c
    return sorted(
        best_by_dataset.values(),
        key=lambda c: len(c["matched_columns"]) + len(c["renamed_columns"]),
        reverse=True,
    )


def find_matches(db: Session, hashes: list[str]) -> dict:
    """Classify a new file's row hashes against every ingested dataset.

    Returns {"outcome": "none"|"exact"|"partial",
             "best_dataset_id": int|None,
             "matches": [per-dataset dicts, best first]}.

    Only datasets with status == "ingested" participate — a provisional
    upload that was never column-selected must not claim rows as "already
    ingested". Only non-duplicate hash rows count: they are the multiset of
    rows the dataset actually holds.
    """
    file_counter = Counter(hashes)
    distinct = list(file_counter)

    # (dataset_id, row_hash) -> stored count, chunked under the param limit.
    stored: dict[int, Counter] = {}
    for i in range(0, len(distinct), _IN_CHUNK):
        chunk = distinct[i : i + _IN_CHUNK]
        rows = (
            db.query(RowHash.dataset_id, RowHash.row_hash, func.count(RowHash.id))
            .join(Dataset, Dataset.id == RowHash.dataset_id)
            .filter(
                Dataset.status == "ingested",
                RowHash.is_duplicate == False,  # noqa: E712 — SQL expression
                RowHash.row_hash.in_(chunk),
            )
            .group_by(RowHash.dataset_id, RowHash.row_hash)
            .all()
        )
        for ds_id, row_hash, n in rows:
            stored.setdefault(ds_id, Counter())[row_hash] = n

    matches = []
    for ds_id, stored_counter in stored.items():
        dataset = db.get(Dataset, ds_id)
        if dataset is None:
            continue
        matched = sum(
            min(file_counter[h], stored_counter[h]) for h in stored_counter
        )
        if matched == 0:
            continue
        dataset_rows = (
            db.query(func.count(RowHash.id))
            .filter(RowHash.dataset_id == ds_id, RowHash.is_duplicate == False)  # noqa: E712
            .scalar()
        )
        selected_columns = [
            q.source_column
            for q in db.query(QuestionColumn)
            .filter(QuestionColumn.dataset_id == ds_id)
            .all()
        ]
        matches.append(
            {
                "dataset_id": ds_id,
                "dataset_name": dataset.name,
                "matched_rows": matched,
                "file_rows": len(hashes),
                "dataset_rows": int(dataset_rows or 0),
                "exact": matched == len(hashes),
                "selected_columns": selected_columns,
            }
        )

    matches.sort(key=lambda m: m["matched_rows"], reverse=True)
    if not matches:
        outcome = "none"
    elif matches[0]["exact"]:
        outcome = "exact"
    else:
        outcome = "partial"
    return {
        "outcome": outcome,
        "best_dataset_id": matches[0]["dataset_id"] if matches else None,
        "matches": matches,
    }


def split_new_rows(hashes: list[str], stored_counter: Counter) -> list[bool]:
    """Walk a file's row hashes in order and flag each as duplicate (True) or
    new (False) against a target dataset's stored multiset. Occurrences beyond
    the stored multiplicity count as new — they get ingested (labeling's
    duplicate-text collapse makes their marginal cost ~zero)."""
    remaining = Counter(stored_counter)
    flags: list[bool] = []
    for h in hashes:
        if remaining[h] > 0:
            remaining[h] -= 1
            flags.append(True)
        else:
            flags.append(False)
    return flags


def stored_multiset(db: Session, dataset_id: int) -> Counter:
    """The target dataset's current non-duplicate hash multiset."""
    rows = (
        db.query(RowHash.row_hash, func.count(RowHash.id))
        .filter(RowHash.dataset_id == dataset_id, RowHash.is_duplicate == False)  # noqa: E712
        .group_by(RowHash.row_hash)
        .all()
    )
    return Counter(dict(rows))
