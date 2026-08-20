# Install

## Requirements

- **Python 3.12**, via conda. Newer versions have no prebuilt pandas/pyarrow
  wheels and try to compile from source.
- **Node.js 18+** (built against 22).
- **Google Cloud credentials (ADC)** — the AI stages run Gemini on Vertex AI
  and authenticate with Application Default Credentials; there are no API
  keys and no offline mode. Install the gcloud CLI, then once per machine:

  ```
  gcloud auth application-default login
  ```

  The account (or, in production, the attached service account) needs Vertex
  AI access (role: Vertex AI User) on a project with the Vertex AI API
  enabled.

## Install

```
git clone https://github.com/ShreyasXAnand/survey-analyzer.git
cd survey-analyzer

conda create -n surveyanalyzer python=3.12 -y
conda activate surveyanalyzer
pip install -r backend/requirements.txt      # requirements-dev.txt to run tests

cd frontend && npm install && cd ..
```

Optional `.env` settings at the repo root (gitignored). The project defaults
to the one recorded by `gcloud auth application-default login`, and the
location to the global endpoint — set these only to override:

```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

Environment variables of the same names take precedence over the file.

Optional, recommended when serving on a network: an admin passcode. With it
set, uploading, reshaping, appending, discarding, metadata edits, exports,
and pipeline runs all require the passcode (the UI prompts for it once per
tab); viewing datasets and asking questions stay open. Without it, nothing
is gated.

```
ADMIN_PASSCODE=pick-a-passcode
```

## Run

```
cd backend
python -m scripts.serve
```

The whole app — UI and API — on <http://localhost:8000>. Builds the frontend
first if it's out of date.

```
--port 9000      different port
--open           open a browser
--skip-build     don't rebuild
--reload         restart the backend on code changes
--host 0.0.0.0   expose on the LAN (there is no authentication)
```

For UI work use the two-process setup instead — `scripts.serve` serves a build,
so it has no hot reload:

```
cd backend && uvicorn app.main:app --reload      # terminal 1
cd frontend && npm run dev                       # terminal 2
```

That serves the UI on :5173, proxying `/api` to :8000.

## Tests

```
cd backend
pytest
```

337 tests, a few seconds, no API calls.

## Reset

```
cd backend
python -m scripts.reset
```

Drops all tables and clears `data/uploads/` + `data/exports/`. Leaves
taxonomies, labels, and answers in place — delete `data/` outright for a clean
slate.

## Notes

- `conda activate surveyanalyzer` is needed in every new terminal.
- After installing Node or conda, open a new terminal so `PATH` picks them up.
- Don't run with `--reload` while a processing job is going; a restart kills it.
- Costs, roughly: $6.75 to process 30k responses, ~$0.01 per question.
