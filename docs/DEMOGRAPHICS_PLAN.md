# Demographic filtering — implementation scope

Status: **phases 1 and 2 built** (phase 1 landed 2026-08-17/18, phase 2 on
2026-08-18). Phase 3 (group-by / comparative sides) remains unbuilt.

**Design change, decided 2026-08-18:** the demographic filter does NOT live
in the asking layer. There is no `DEMOGRAPHIC_BLOCK`, the router never
proposes or normalizes a demographic filter, and §6's route-step design below
is superseded. Instead the filter works like the question-scope selector: the
analyst picks values on the ask form, one dropdown per field, served by
`POST /ask/demographics` — POST because the counts are **faceted**: send the
current selection and each field comes back recounted under the *other*
fields' ticked values (never its own, so an OR selection stays extendable),
plus an `n_matching_respondents` figure the form shows as a running total.
The ask requests then carry `{field: [values]}` through both steps, the
server validates it against the dataset's own values (422 on anything else),
and `gather_evidence` applies it deterministically as one more link in the
filter chain, with the `{in_scope, coded, matching}` denominator. The one concession the route step makes: when a filter is
active, `DEMOGRAPHIC_NOTE` is appended to the route prompt telling the model
the restriction is *already handled in code* — without it, a question worded
"what do women say…" was refused as unanswerable (observed on the first live
filtered ask). The note is informational only; the router cannot choose or
change the filter. Everything else below (§2–§5, §7–§9 semantics,
denominators, thin-cell notice, OR-within-field / AND-across-fields) shipped
as designed. Written 2026-08-17.

*The rest of this document is the original plan, kept as the design record —
read it with the status and design change above in mind.*

When this was written no dataset could carry demographics: `select_columns`
only accepted open-ended question columns, so no other column reached the
database, the export, or the ask path. `docs/PRODUCT_GUIDE.md` §11 recorded
this as deliberate deferral. This document was the plan for closing it.

The goal: an analyst marks "District", "Age band", "Tenure" as metadata columns at
ingest, and can then ask *"what do District 3 residents say about parking?"* with
the same computed-counts-and-real-quotes guarantees every other answer has.

---

## 1. The seam that already exists

Three things make this cheaper than it looks.

**`respondent_key` is already in the export.** `ingest.py:463` writes
`f"{dataset.id}:{r.source_row_index}"` into every parquet row. Demographics are a
property of a *respondent* (one source row), not of a response — one respondent
contributes N responses, one per question. `respondent_key` is the join key, and
it is already there.

**Filters already compose as orthogonal dimensions.** `gather_evidence` narrows a
single `allowed` set of response keys through a chain of independent filters
(`router.py:880-940`), each recording an `{in_scope, coded, matching}` denominator.
A demographic filter is one more link in that chain. It is **not** a new route —
see `memory/router-dimensions-pattern.md`.

**Conditional prompt blocks are already the pattern.** `ACTIONABILITY_BLOCK`,
`EVENT_BLOCK` and `TIME_BLOCK` are injected into `ROUTE_SYSTEM` only when the
artifacts carry that field, so the router is never offered a filter the data
cannot honour. `DEMOGRAPHIC_BLOCK` follows the same rule.

And the answer-cache key needs no new field: `ask_cache._NORMALIZED_FIELDS` hashes
every request field it does *not* know about via the `extra` catch-all, precisely
so "a future filter (demographics, etc.) is part of the key the day it is added to
the request schema" (`ask_cache.py:54`).

---

## 2. Storage — new tables only, no `ALTER TABLE`

There is no migration framework. `Base.metadata.create_all` creates missing tables
but never alters existing ones; past column additions were hand-written ALTERs
(`scripts/backfill_uploads.py`). **This design adds only new tables**, so
`create_all` handles it and no existing row is touched.

```python
class MetadataColumn(Base):          # mirrors QuestionColumn
    __tablename__ = "metadata_columns"
    __table_args__ = (UniqueConstraint("dataset_id", "source_column"),)
    id, dataset_id, source_column, label, position
    kind: str          # "categorical" | "ordinal" — see §3
    n_distinct: int    # cardinality at ingest, for the guard in §3

class RespondentAttribute(Base):
    __tablename__ = "respondent_attributes"
    __table_args__ = (UniqueConstraint("dataset_id", "source_row_index",
                                       "metadata_column_id"),)
    id, dataset_id, upload_id, source_row_index, metadata_column_id
    value: str | None  # None = the cell was blank; NOT a category
```

Keyed by `source_row_index`, not `response_id` — storing it per response would
duplicate each value once per question and invite the two copies to disagree.

**Blank is not a value.** A respondent who left the district cell empty is missing
data, never an "Unknown" bucket that can be filtered *for*. This is the same
asymmetry `event_occurred` and `time_context` already enforce, and it is the single
easiest thing to get wrong here.

---

## 3. Ingest — the cardinality guard is the real work

The API change is small: `SelectColumnsRequest` gains
`metadata: list[MetadataColumnSelection]` alongside `questions`. The judgement is
in what may be selected.

A demographic column is only useful if it is **low-cardinality and categorical**.
A free-text "comments" column or a continuous "age in years" column would produce a
dropdown with thousands of entries and filters matching one person each. So
`select_columns` computes, per proposed metadata column:

- `n_distinct` over non-blank values
- the blank rate
- whether values parse as unbounded numerics or dates

**`METADATA_MAX_DISTINCT = 100`, and it warns — it does not block.** (Decided
2026-08-17.) Past 100 distinct values, or when the values look continuous, the
selection screen shows a warning naming the count and the analyst proceeds anyway
if they want to. No 422, no silent drop. The tool should not guess bin edges for
someone, and it should not refuse a column its owner understands better than it
does.

The guidance goes in the docs and the UI copy instead — see §7.2.

The column-suggestion endpoint should also *propose* likely metadata columns the
same way it proposes question columns, ranked by low cardinality and short values.

**Incremental appends.** `select_columns` re-reads every upload, so metadata rows
are written per upload exactly like responses. A column present in the first file
and absent from an appended one yields `value=None` for the new rows — missing, not
an error, and the disclosure in §6 will say how many.

---

## 4. Export — a sidecar, not a schema change

`RESPONSE_PARQUET_SCHEMA` (`ingest.py:60`) is a fixed `pa.schema` that every
existing artifact and the whole ask path depend on. Do **not** add per-dataset
demographic columns to it.

Instead write `data/exports/{id}/respondents.parquet`:

| column | type |
|---|---|
| `respondent_key` | string |
| `field` | string (the `MetadataColumn.label`) |
| `value` | string |

Long format, blanks omitted. One small file, no change to the response schema, and
it reads naturally as "which respondents hold which value".

`ask_service._context_key` must stat this file so a re-ingest invalidates the
in-process context cache and the persistent answer cache together — same as it
already stats `locations.json` and `lexicon.json`.

---

## 5. AskContext — mirror `location_members`

```python
# field -> value -> response_keys        (mirrors location_members)
demographic_members: dict[str, dict[str, set[str]]]
# field -> [(value, n_respondents)], for the router summary and the UI dropdown
demographic_values: dict[str, list[tuple[str, int]]]
# response_keys whose respondent HAS a value for this field — the `coded` set
demographic_coded: dict[str, set[str]]
```

Built in `load_context` by joining `respondents.parquet` to the corpus on
`respondent_key`, which is derivable from every `response_key`
(`dataset:question:row` → `dataset:row`).

`demographic_coded` is tracked separately for the same reason `event_coded` is: a
respondent with no value is not evidence of anything, and every denominator must
be able to say so.

---

## 6. Router — one more dimension in the chain

**Route step.** `DEMOGRAPHIC_BLOCK` is injected only when
`demographic_values` is non-empty, listing each field, its values and their counts.
The route dict gains:

```python
"demographic_filter": {"District": ["3", "5"], "Tenure": ["10+ years"]}
```

Semantics, which must be stated in the prompt *and* the disclosure: **values within
one field are OR, fields are AND.** "District 3 or 5, who have lived here 10+
years."

`normalize_demographics()` validates every field and value against the context and
drops unknown ones — same posture as `location_filter`, where an invented place is
never trusted. An unknown field in the *analyst's* request (step 2) raises, giving
a 422, matching `route_from_selection`'s treatment of unknown filter values.

**Evidence step.** In `gather_evidence`, after the time filter:

```python
demographic_denominator = None
if demo_filter and demographic_members:
    pre = scope_under(allowed)
    hits = set.intersection(*[
        set().union(*(demographic_members[f].get(v, set()) for v in vals))
        for f, vals in demo_filter.items()])
    demographic_denominator = {
        "in_scope": len(pre),
        "coded": len(pre & coded_for_all_filtered_fields),
        "matching": len(pre & hits),
    }
    allowed = restrict(hits)
```

**Synthesis guidance.** `build_synth_prompts` appends a sentence in the same shape
as the location/actionability/event ones: state the restriction, copy the
denominator from COMPUTED COUNTS, and never characterise the respondents who have
no value for the field.

---

## 7. Disclosure, not blocking

**Decided 2026-08-17: no hard blocks anywhere in this feature.** Nothing below
refuses an action; everything below tells the analyst what they are looking at.

### 7.1 Thin cells — disclosed, never blocked

Re-identification is **out of scope**: this is an internal tool for curated
employees, not a public publishing surface. (Decided 2026-08-17.) No privacy
gating, no quote withholding.

What remains is an evidence-quality point. A filter narrow enough ("District 3,
65+, renter") can reduce a cell to a handful of responses, and an answer built on
four verbatims reads with the same confidence as one built on four hundred. That is
the same failure `SMALL_BASE_N` already guards, one order of magnitude down.

So a **notice**, in the same family as the existing small-base and event caveats:

- when a demographic-filtered answer rests on fewer than `DEMOGRAPHIC_NOTICE_N`
  (propose 10) matching responses, the answer screen and the process note say so
- quotes are still shown, the answer is still produced, nothing is withheld
- the constant folds into `ask_logic_hash`, so tuning it invalidates cached answers
  (that is what the hash is for)

### 7.2 Guidance belongs in the docs and the UI copy

Since neither the cardinality guard nor the notice blocks anything, the steering
happens in words. Two places, same message:

**`docs/PRODUCT_GUIDE.md`**, in the demographics section when phase 1 ships:

> Demographic columns work best when they are **general groupings, not
> ultra-specific ones** — "District", "Age band", "Own/Rent", "Years in the city"
> rather than exact age, street address, or ZIP+4. Broad groups give each filter
> enough respondents for the counts to mean something. Nothing stops you selecting
> a fine-grained column; the answers simply get thinner as the groups get smaller,
> until each one rests on a handful of responses.

**The column-selection screen**, as one line under the metadata picker: *"Best with
general groupings (District, Age band) rather than exact values."*

### 7.3 Small bases: no change

`SMALL_BASE_N = 200` stays exactly as it is. (Decided 2026-08-17.) Demographic
filtering will trip it more often, and that is the guard doing its job — a
200-response answer should say so whether or not a filter caused it. No ratio
rework, no new threshold.

---

## 8. Frontend

The `CheckboxDropdown` built for multi-select question scope is already the right
control. Needed:

- `GET /datasets/{id}/ask/demographics` → `[{field, values: [{value, n}]}]`
- one dropdown per field, rendered only when the dataset has that field
- the request carries `demographic_filter`; `askRoute` passes it through as `extra`
  so it lands in the cache key
- an active-filter chip on the answer screen, and a row in the Summary card giving
  the matching/in-scope denominator
- the small-cell notice from §7.1 rendered as an `ask-caveat`, beside the existing
  small-base and event caveats — quotes still render underneath it
- the "general groupings" hint from §7.2 under the metadata picker on the
  column-selection screen

---

## 9. Phasing

| Phase | Scope | Ships |
|---|---|---|
| **1** | `MetadataColumn` + `RespondentAttribute`, cardinality guard, `respondents.parquet`, ingest UI | Analyst can mark and store demographics; visible in the catalog. No ask changes. |
| **2** | `AskContext` fields, `DEMOGRAPHIC_BLOCK`, filter + denominator in `gather_evidence`, synth guidance, quote floor, frontend dropdowns | The feature as asked for. |
| **3** | `group_by: "demographic"`, and demographic sides in `comparative` routes | "How do District 3 and District 7 differ?" as one answer. |

Phase 3 is deliberately separate: grouping by demographic multiplies the section
plan's structure logic, and phase 2 delivers the asked-for capability without it.

**Testing per phase** follows existing files: `test_ingest.py` for the cardinality
guard and append behaviour, `test_router.py` for filter composition and the OR/AND
semantics, `test_verify.py` for the quote-floor disclosure, `test_ask_api.py` for
the 422s.

---

## 10. Decisions and remaining questions

Settled 2026-08-17:

| Question | Decision |
|---|---|
| `METADATA_MAX_DISTINCT` | **100**, and it warns rather than rejects |
| Hard blocks anywhere in the feature | **None.** Disclose, never refuse |
| Re-identification | **Out of scope** — internal tool, curated users. No privacy gating |
| Thin-cell handling | Caveat at `DEMOGRAPHIC_NOTICE_N` (~10); quotes still shown |
| Steering toward general groupings | Product guide section + one line of UI copy (§7.2) |
| `SMALL_BASE_N` | **Unchanged at 200**; no ratio rework |

Still open:

1. Should phase 1 backfill dataset 2, or apply only to new uploads? (The production
   file has no demographic columns, so there may be nothing to backfill.)
2. Is a demographic filter allowed to be the *only* filter on an `aggregate_direct`
   tally — "how many respondents per district" — or does that want its own
   `aggregate_target`?
3. `DEMOGRAPHIC_NOTICE_N` at 10 — the number is a guess; worth revisiting once
   there is a dataset with real demographic columns to look at.
