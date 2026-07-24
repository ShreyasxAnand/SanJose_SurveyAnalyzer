# Survey Analyzer

See `the project notes` for the full project scope and architecture. This is Phase 1
(Ingest) only.

## Backend

```
cd backend
conda activate surveyanalyzer   # env built from requirements.txt, Python 3.12
uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs once running.

Uploads and exports land in `./data/` at the repo root (gitignored, created
on boot). Run tests:

```
cd backend
pytest
```

Reset local state (drop+recreate tables, clear `./data`) while iterating on
the ingest transform:

```
cd backend
python -m scripts.reset
```

## Frontend

Requires Node.js (not yet installed on this machine as of project setup).

```
cd frontend
npm install
npm run dev
```

Dev server runs at http://localhost:5173 and proxies `/api/*` to the backend
at http://localhost:8000.
