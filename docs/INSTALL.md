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

Nothing else to set up: the project comes from the credentials you just
created, and the region defaults to the global endpoint. Override either only
if you need to — see **Google Cloud** under Server settings below.

## Server settings

Copy `config.example.json` to `config.json` at the repo root, or set them from
the app's **Settings** screen once it is running. The file is gitignored — it
holds the passcode.

```json
{
  "admin_passcode": "pick-a-passcode",
  "models": {
    "default": "gemini-3.5-flash-lite",
    "synth": "gemini-3.5-flash-lite"
  },
  "prices_per_mtok": {
    "gemini-3.5-flash-lite": { "input": 0.30, "output": 2.50 }
  }
}
```

**Admin passcode** — optional, recommended when serving on a network. With it
set, uploading, reshaping, appending, discarding, metadata edits, exports,
dataset deletion, and pipeline runs all require the passcode (the UI prompts
for it once per tab); viewing datasets and asking questions stay open. Without
it, nothing is gated, which is the right setting for a single-user desktop
install. Also readable from `ADMIN_PASSCODE` in the environment.

**Google Cloud** — optional. `vertex.project` and `vertex.location` pin which
Vertex AI project and region the model calls bill to. Leave them out and the
project comes from whatever `gcloud auth application-default login` recorded
on the machine, and the region is the global endpoint; set them for a second
project or a data-residency requirement. Also readable from
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in the environment.

**Models** — `default` is the workhorse (induction, labeling, sub-themes,
routing: everything whose call count grows with the corpus); `synth` writes
the answer text, one call per question asked. Leave `synth` out and it follows
`default`. Any model id the Vertex endpoint accepts works.

**Prices** — USD per 1,000,000 tokens, as published on the Vertex AI pricing
page. Merged over a built-in table, so listing a model re-prices it and
listing an unknown one makes it priceable. A model with no rate reports its
token counts and says the cost is unknown rather than showing $0.00.

Precedence for every setting: environment variable > `config.json` > built-in
default. Nothing needs a restart — the file is re-read when its timestamp
changes.

There is no `.env` file. One used to sit below `config.json` in that chain,
left over from the API-key era; it is not read anywhere any more, so a
leftover copy from an older checkout has no effect and can be deleted.

### Cost estimates

Before a run, prompt tokens are counted with Vertex's `countTokens` — free,
and not a model call. Output tokens cannot be counted in advance by anyone, so
those rates are measured from runs that already finished on this machine.
**Settings → Recalibrate from completed runs** re-derives them from the run
manifests on disk (also free and offline) and stores them in `config.json`.
Until you do, built-in rates apply and the estimate says so.

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
taxonomies, labels, sub-themes, and answers in place — delete `data/` outright
for a clean slate.

That leftover matters more than it looks: dataset ids restart after a reset,
so an old `data/labels/1/` would be silently adopted by whatever becomes
dataset 1 next. To remove one dataset and everything derived from it, use
**Delete dataset** in the catalog's ⋯ menu instead — it clears all twelve
artifact roots and reports anything it could not.

## Notes

- `conda activate surveyanalyzer` is needed in every new terminal.
- After installing Node or conda, open a new terminal so `PATH` picks them up.
- Don't run with `--reload` while a processing job is going; a restart kills it.
- Costs, roughly: $6.75 to process 30k responses, ~$0.01 per question.
