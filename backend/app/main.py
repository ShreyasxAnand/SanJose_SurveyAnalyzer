from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ask_api import router as ask_router
from app.db import REPO_ROOT, init_db
from app.ingest import router as datasets_router
from app.pipeline_api import router as pipeline_router

# Every API route lives under /api so the SPA catch-all below can own every
# other path. The frontend has always called /api/... — in the two-process dev
# setup Vite's proxy stripped the prefix, which meant the prefix only existed
# in the dev server. Now it is real on both paths (see vite.config.ts).
API_PREFIX = "/api"

# Built by `npm run build`; absent in a fresh clone, which is fine — the app
# then serves the API only, exactly as it did before single-process serving.
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Survey Analyzer API", lifespan=lifespan)

# Only needed for the two-process dev setup (Vite on :5173 calling :8000).
# Single-process serving is same-origin, where this never comes into play.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router, prefix=API_PREFIX)
app.include_router(pipeline_router, prefix=API_PREFIX)
app.include_router(ask_router, prefix=API_PREFIX)


@app.get("/health")
@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA from the same process as the API.

    Registered last on purpose: FastAPI matches routes in registration order,
    so every real API route above wins before the catch-all is consulted.
    """
    dist_root = FRONTEND_DIST.resolve()
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        # An unmatched /api/... path is a client bug, not a deep link — it must
        # 404 rather than hand back index.html with a 200.
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if full_path:
            candidate = (FRONTEND_DIST / full_path).resolve()
            # is_relative_to keeps ../ out of the data dir and the DB.
            if candidate.is_file() and candidate.is_relative_to(dist_root):
                return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


if FRONTEND_DIST.is_dir():
    _mount_frontend(app)
