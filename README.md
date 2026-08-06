# Survey Analyzer

Ask plain-language questions about open-ended survey responses and get answers
grounded in the actual text — with real counts and quotes that trace back to
source rows.

The docs are the reference: full architecture, design
principles, measured costs, and the reasoning behind decisions that look
arbitrary. This file is just how to run it.

## Status

Phases 1–6 built. Upload a wide-format CSV/Excel file, pick the open-ended
columns, process the dataset, and ask questions — all from the browser. The CLI
covers the same pipeline plus the taxonomy review loop.

Validated on a 28,906-response production file: full pipeline in ~15 minutes
for ~$6.75, answers at ~$0.01 each.

## Setup

Backend runs in a conda env named `surveyanalyzer` (Python 3.12 — the system
Python is too new for prebuilt pandas/pyarrow wheels, and there's no compiler
here to build from source).

```
cd frontend && npm install       # once
conda activate surveyanalyzer    # env built from requirements.txt
```

Model calls need `GEMINI_API_KEY` in the environment or in a `.env` at the repo
root.

## Running it

**One command** — builds the UI if it's out of date, then serves the whole app,
UI and API, from a single process on http://localhost:8000:

```
cd backend
python -m scripts.serve
```

The build is a snapshot, so the script compares the newest file under
`frontend/src` against `frontend/dist` and rebuilds when it's behind — serving a
stale bundle would silently show you an older app. A build takes ~5s; when
nothing changed it's skipped.

```
python -m scripts.serve --port 9000
python -m scripts.serve --skip-build     # serve dist as-is (warns if stale)
python -m scripts.serve --reload         # reload the backend on change
python -m scripts.serve --open           # open a browser once it's up
```

**Two commands, with hot reload** — the better setup while editing the UI, since
`scripts.serve` has no HMR and needs a rebuild for every frontend change:

```
cd backend && uvicorn app.main:app --reload      # terminal 1
cd frontend && npm run dev                       # terminal 2
```

That serves the UI at http://localhost:5173, proxying `/api/*` to the backend on
:8000. Both setups hit identical URLs — the API genuinely lives under `/api`, so
the dev proxy forwards without rewriting. API docs at
http://localhost:8000/docs.

> **Don't run the backend with `--reload` while a pipeline job is in flight.** A
> reload restarts the process and kills the run. Induction checkpoints its
> expensive phase, so re-running skips what it already paid for — but a
> 15-minute run dying because you saved a file is avoidable.

## Using it

**From the browser.** Ingest tab: upload → tick the open-ended columns and give
each one its real question wording → *Reshape & persist*. The done screen then
offers **Process this dataset**, which shows a free cost estimate before
spending anything; confirm and it runs induce → label → lexicon → locations as
a background job with per-stage cost and progress. When it finishes, the Ask tab
can answer questions about that dataset.

**From the CLI**, from `backend/` inside the conda env. Same stages the button
runs, but you can target one question, resume, or override flags:

```
python -m scripts.induce --question 2 --dry-run   # call plan + cost, free
python -m scripts.induce --question 2
python -m scripts.label --all                     # then auto-refreshes exports
python -m scripts.build_lexicon
python -m scripts.build_locations
python -m scripts.ask "what makes people feel unsafe downtown?"
```

With more than one dataset on disk, pass `--parquet ../data/exports/{id}/responses.parquet`
— these commands otherwise auto-discover the most recently exported one.

**The review loop is CLI-only on purpose.** It's the human quality gate on a
freshly induced taxonomy, and a button that auto-applied edits would remove the
gate rather than serve it:

```
python -m scripts.review --question 2             # free defect report
python -m scripts.review --question 2 --edits ../data/review/1/2/edits.json
```

## Where things land

Everything is under `./data/` at the repo root (gitignored, created on boot).
The SQLite database is the source of truth; exports and pipeline artifacts are
reproducible and safe to delete. Original uploads are kept byte-for-byte and
never mutated.

```
data/uploads/{id}/          the file exactly as uploaded
data/exports/{id}/          responses.parquet + .csv + manifest.json
data/taxonomy/{ds}/{q}/     induced taxonomies, versioned per run
data/labels/{ds}/{q}/       label assignments, versioned per run
data/answers/{ds}/{run}/    answer.md + a full audit manifest per question
data/jobs/{ds}/{job}/       pipeline run status + captured stage output
```

Old runs are never overwritten. "Latest" means the lexicographically last run
directory, so a `*_review` run supersedes the run it derived from.

## Tests

```
cd backend
pytest
```

The offline suite makes no API calls and finishes in a few seconds. **If it
suddenly takes ~40s, a test is making real billed calls** — that runtime jump is
the tell, and it has happened before (a missed client patch).

```
cd frontend
npx tsc --noEmit
npm run build
```

## Resetting

Drops and recreates the tables and clears `./data` — including every taxonomy,
label run, and answer:

```
cd backend
python -m scripts.reset
```
