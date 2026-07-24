"""Run the whole app — API and UI — as a single process on one port.

From backend/, inside the surveyanalyzer conda env:

    python -m scripts.serve                 # build if stale, then serve :8000
    python -m scripts.serve --port 9000
    python -m scripts.serve --skip-build    # serve whatever dist/ holds now
    python -m scripts.serve --host 0.0.0.0  # reachable from the LAN (no auth!)

The UI is served from frontend/dist, which `npm run build` produces. That
build is a snapshot, so this script compares the newest frontend source file
against dist/index.html and rebuilds when the snapshot is behind — serving a
stale bundle silently shows an older app, which is the one failure mode of
single-process serving that is genuinely hard to notice.

For UI work, prefer the two-process setup — it has hot reload, which this
does not:

    uvicorn app.main:app --reload     # terminal 1, from backend/
    npm run dev                       # terminal 2, from frontend/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from app.db import REPO_ROOT

FRONTEND_DIR = REPO_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
BUILD_STAMP = DIST_DIR / "index.html"

# Everything `npm run build` reads. A change to any of these makes dist stale.
SOURCE_DIRS = [FRONTEND_DIR / "src"]
SOURCE_FILES = [
    FRONTEND_DIR / "index.html",
    FRONTEND_DIR / "package.json",
    FRONTEND_DIR / "package-lock.json",
    FRONTEND_DIR / "vite.config.ts",
    FRONTEND_DIR / "tsconfig.json",
]

# Node isn't always on PATH in a shell that predates its install (see the
# PATH quirk on this machine); check the default install location too.
NPM_FALLBACKS = [
    Path("C:/Program Files/nodejs/npm.cmd"),
    Path("C:/Program Files (x86)/nodejs/npm.cmd"),
]


def newest_source_mtime() -> tuple[float, Path | None]:
    """Latest mtime across the frontend sources, with the file responsible."""
    newest = 0.0
    culprit: Path | None = None
    candidates = list(SOURCE_FILES)
    for directory in SOURCE_DIRS:
        if directory.is_dir():
            candidates.extend(p for p in directory.rglob("*") if p.is_file())
    for path in candidates:
        if not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime > newest:
            newest, culprit = mtime, path
    return newest, culprit


def build_state() -> tuple[str, Path | None]:
    """One of 'missing' | 'stale' | 'fresh', plus the file that made it stale."""
    if not BUILD_STAMP.is_file():
        return "missing", None
    newest, culprit = newest_source_mtime()
    if newest > BUILD_STAMP.stat().st_mtime:
        return "stale", culprit
    return "fresh", None


def find_npm() -> str | None:
    found = shutil.which("npm")
    if found:
        return found
    for candidate in NPM_FALLBACKS:
        if candidate.is_file():
            return str(candidate)
    return None


def run_build() -> None:
    npm = find_npm()
    if npm is None:
        sys.exit(
            "npm not found on PATH.\n"
            "  Node is installed at C:\\Program Files\\nodejs on this machine — a\n"
            "  terminal opened before that install won't see it; open a new one.\n"
            "  Or pass --skip-build to serve the existing frontend/dist as-is."
        )
    if not (FRONTEND_DIR / "node_modules").is_dir():
        sys.exit(f"{FRONTEND_DIR / 'node_modules'} is missing — run `npm install` first.")

    print(f"building frontend  ({npm} run build)", flush=True)
    started = time.monotonic()
    result = subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR)
    if result.returncode != 0:
        sys.exit(
            f"\nfrontend build failed (exit {result.returncode}) — not serving a stale "
            f"bundle.\nFix the build, or pass --skip-build to serve {DIST_DIR} as it is."
        )
    print(f"build ok in {time.monotonic() - started:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) or 0.0.0.0 to expose on the LAN")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--skip-build", action="store_true",
                    help="serve frontend/dist as-is, even if it is stale")
    ap.add_argument("--build-only", action="store_true",
                    help="build the frontend and exit without serving")
    ap.add_argument("--reload", action="store_true",
                    help="reload the BACKEND on change (the UI still needs a rebuild)")
    ap.add_argument("--open", action="store_true", help="open a browser once serving")
    args = ap.parse_args()

    state, culprit = build_state()
    if args.skip_build:
        if state == "missing":
            sys.exit(f"--skip-build, but there is no build at {DIST_DIR}. "
                     f"Run without --skip-build (or `npm run build` in frontend/).")
        if state == "stale":
            print(f"WARN serving a STALE build — {culprit} is newer than the bundle",
                  flush=True)
    elif state == "missing":
        print(f"no build at {DIST_DIR}", flush=True)
        run_build()
    elif state == "stale":
        print(f"build is stale ({culprit} is newer)", flush=True)
        run_build()
    else:
        print("frontend build is current", flush=True)

    if args.build_only:
        return 0

    url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"\n  Survey Analyzer  ->  {url}\n", flush=True)
    if args.host == "0.0.0.0":
        print("  NOTE --host 0.0.0.0 exposes this to your network, and the app has no\n"
              "       authentication: anyone who can reach the port can read every\n"
              "       dataset and spend the API key.\n", flush=True)
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
