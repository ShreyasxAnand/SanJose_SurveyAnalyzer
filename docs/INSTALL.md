# Install

## Requirements

- **Python 3.12**, via conda. Newer versions have no prebuilt pandas/pyarrow
  wheels and try to compile from source.
- **Node.js 18+** (built against 22).
- **A Google Gemini API key** — <https://aistudio.google.com/apikey>. Every AI
  stage needs it; there is no offline mode.

## Install

```
git clone https://github.com/ShreyasXAnand/survey-analyzer.git
cd survey-analyzer

conda create -n surveyanalyzer python=3.12 -y
conda activate surveyanalyzer
pip install -r backend/requirements.txt      # requirements-dev.txt to run tests

cd frontend && npm install && cd ..
```

Put the key in a `.env` file at the repo root (gitignored):

```
GEMINI_API_KEY=your-key-here
```

`GEMINI_API_KEY` in the environment takes precedence if you'd rather set one.

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

255 tests, a few seconds, no API calls.

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
