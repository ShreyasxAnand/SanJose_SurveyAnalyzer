"""Response-date parsing and period labeling.

A metadata column may be typed "date" (see MetadataColumn.value_type): its
cells are parsed at ingest time and stored as ISO "YYYY-MM-DD" strings, so
everything downstream — the respondents sidecar, the responses export, the
ask layer — sees one canonical format regardless of what the source file
used. An unparseable cell stores NO row, the same missing-data asymmetry a
blank demographic cell has.

Period labels ("2023 Q3", "Pre-election") are NOT stored per respondent.
They are derived at read time from the dataset's date-ranges config
(Dataset.date_ranges_json), so renaming or re-cutting periods is a metadata
edit — never a re-ingest. Two config shapes:

    {"mode": "bucket", "granularity": "quarter" | "month" | "year"}
    {"mode": "ranges", "ranges": [{"label": ..., "start": ISO, "end": ISO}]}

Bucket mode needs no boundaries — every date gets a calendar label. Ranges
mode maps a date to the first range containing it; dates outside every range
get UNLABELED, a real facet value (the analyst can filter for "responses I
never labeled"), unlike a blank cell which is missing data.
"""

from __future__ import annotations

import datetime as dt

UNLABELED = "(unlabeled)"

# Accepted source formats, tried in order. ISO first (already-clean data and
# our own round-trips), then the US forms this project's real uploads use
# ("9/19/2023"). Two-digit years are %y-interpreted (69 → 2069? no: Python
# maps 00-68 to 2000s, 69-99 to 1900s) — survey data is recent, so that rule
# never bites in practice, and the alternative (rejecting them) loses rows.
_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d-%b-%Y")


def parse_date(raw: str) -> str | None:
    """One cell → ISO "YYYY-MM-DD", or None when it isn't a date."""
    value = str(raw).strip()
    if not value:
        return None
    # Excel datetime-ish cells arrive as "2023-09-19 00:00:00" — split off
    # any time part before trying date formats.
    value = value.split(" ")[0].split("T")[0]
    for fmt in _FORMATS:
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def bucket_label(iso: str, granularity: str) -> str:
    """Calendar bucket for one ISO date. Labels sort chronologically as
    plain strings ("2023 Q3" < "2024 Q1", "2023-09" < "2024-01")."""
    year, month = iso[:4], int(iso[5:7])
    if granularity == "quarter":
        return f"{year} Q{(month - 1) // 3 + 1}"
    if granularity == "month":
        return f"{year}-{month:02d}"
    if granularity == "year":
        return year
    raise ValueError(f"Unknown granularity {granularity!r}")


def validate_ranges_config(config: dict) -> dict:
    """Normalize + validate a date-ranges config dict. Raises ValueError
    with an analyst-readable message on anything malformed."""
    mode = config.get("mode")
    if mode == "bucket":
        granularity = config.get("granularity")
        if granularity not in ("quarter", "month", "year"):
            raise ValueError(
                "date_ranges bucket granularity must be quarter, month, or year"
            )
        return {"mode": "bucket", "granularity": granularity}
    if mode == "ranges":
        ranges = config.get("ranges") or []
        if not ranges:
            raise ValueError("date_ranges mode 'ranges' needs at least one range")
        cleaned = []
        for r in ranges:
            label = str(r.get("label", "")).strip()
            if not label:
                raise ValueError("Every date range needs a label")
            if label == UNLABELED:
                raise ValueError(f"{UNLABELED!r} is reserved for unmatched dates")
            try:
                start = dt.date.fromisoformat(str(r.get("start", "")))
                end = dt.date.fromisoformat(str(r.get("end", "")))
            except ValueError as exc:
                raise ValueError(
                    f"Range '{label}': dates must be ISO YYYY-MM-DD"
                ) from exc
            if start > end:
                raise ValueError(f"Range '{label}': start is after end")
            cleaned.append(
                {"label": label, "start": start.isoformat(), "end": end.isoformat()}
            )
        for a, b in zip(cleaned, cleaned[1:]):
            if b["start"] <= a["end"]:
                raise ValueError(
                    f"Ranges '{a['label']}' and '{b['label']}' overlap or are "
                    "out of order — sort them and keep them disjoint"
                )
        return {"mode": "ranges", "ranges": cleaned}
    raise ValueError("date_ranges mode must be 'bucket' or 'ranges'")


def period_label(iso: str, config: dict) -> str:
    """One ISO date → its period label under the given (validated) config."""
    if config["mode"] == "bucket":
        return bucket_label(iso, config["granularity"])
    for r in config["ranges"]:
        if r["start"] <= iso <= r["end"]:
            return r["label"]
    return UNLABELED


def period_sort_key(config: dict):
    """Sort key putting period labels in chronological order: bucket labels
    sort as strings by construction; ranges follow the config's own order,
    with UNLABELED last."""
    if config["mode"] == "bucket":
        return lambda label: (label == UNLABELED, label)
    order = {r["label"]: i for i, r in enumerate(config["ranges"])}
    return lambda label: (label not in order, order.get(label, 0), label)


def apply_period_labels(
    date_fields: set[str],
    config: dict | None,
    demographic_values: dict[str, list[tuple[str, int]]],
    demographic_members: dict[str, dict[str, set[str]]],
    demographic_coded: dict[str, set[str]],
    demographic_respondents: dict[str, dict[str, set[str]]],
) -> None:
    """Rewrite date-typed demographic fields in place: raw ISO-date values
    (thousands of distinct facets — useless to filter on) collapse into
    period labels under `config`. Falls back to quarter bucketing when no
    config is stored, so a date column is useful with zero setup. Fields not
    in `date_fields` are untouched; `demographic_coded` already aggregates
    per field, so it needs no rewrite."""
    if not date_fields:
        return
    if config is None:
        config = {"mode": "bucket", "granularity": "quarter"}
    for field in date_fields:
        raw_members = demographic_members.get(field)
        raw_respondents = demographic_respondents.get(field)
        by_period_members: dict[str, set[str]] = {}
        by_period_respondents: dict[str, set[str]] = {}
        if raw_members:
            for iso, keys in raw_members.items():
                by_period_members.setdefault(
                    period_label(iso, config), set()).update(keys)
            demographic_members[field] = by_period_members
        if raw_respondents:
            for iso, pks in raw_respondents.items():
                by_period_respondents.setdefault(
                    period_label(iso, config), set()).update(pks)
            demographic_respondents[field] = by_period_respondents
        if field in demographic_values:
            counts: dict[str, int] = {}
            for iso, n in demographic_values[field]:
                counts[period_label(iso, config)] = (
                    counts.get(period_label(iso, config), 0) + n)
            key = period_sort_key(config)
            demographic_values[field] = [
                (label, counts[label]) for label in sorted(counts, key=key)
            ]
