"""Deleting a dataset, and telling the truth about it first.

Until now only a provisional upload could be discarded. An ingested dataset
could not, and the reason was written into `discard_dataset`'s docstring: it
"has derived artifacts (taxonomies, labels, answers) and deleting it from a
button would orphan them silently". That reasoning was right, and this module
is the answer to it rather than an exception from it — every root a dataset
owns is enumerated here, in one list, and the delete either clears all of them
or reports what it could not.

**Orphans are not a tidiness problem, they are a correctness problem.**
`Dataset.id` is a plain SQLite primary key with no AUTOINCREMENT, so deleting
the highest-numbered dataset means the next upload reuses that id. Any
directory left behind — `data/labels/7`, `data/answers/7`, `data/summary/7` —
would then be adopted by an unrelated dataset that happens to become id 7, and
the app would present one survey's labels as another's. That is the failure
this module exists to prevent, and it is why `ROOTS` must stay exhaustive.

**Preview before destroy.** `preview` walks the same list the delete walks and
reports counts, bytes, and the model spend recorded in the manifests about to
be deleted. Sunk cost is the number that actually makes someone stop and
think: "12,000 responses" sounds like data you could re-upload, "$47 of model
spend, not recoverable" is what it really is.

**Refuse rather than race.** A pipeline job writing into these directories is
grounds for a 409, exactly as it is for an append or a metadata edit. Deleting
a tree a subprocess is mid-write in is how you get a half-removed run that
still looks loadable.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import ask_service, induction, lexicon, locations, rulings, subthemes
from .db import EXPORTS_DIR, UPLOADS_DIR
from .summary import LABELS_DIR, SUMMARY_DIR


@dataclass(frozen=True)
class Root:
    """One directory tree keyed by dataset id at its first level."""

    key: str
    label: str
    # a callable, not a Path: several of these constants are monkeypatched in
    # tests and reassigned per-install, so resolving them at call time is the
    # only way the purge follows the same paths the writers used
    path: object
    # what the analyst loses, in their words — shown in the confirm dialog
    describes: str = ""


def _data_dir() -> Path:
    return induction.DATA_DIR


# EVERY per-dataset root. Adding a new artifact directory anywhere in the app
# means adding it here; `test_purge_covers_every_dataset_keyed_root` fails
# loudly when one is missing, because a miss is silent and permanent.
ROOTS: tuple[Root, ...] = (
    Root("uploads", "uploaded files", lambda: UPLOADS_DIR,
         "the original spreadsheets, exactly as uploaded"),
    Root("exports", "exports", lambda: EXPORTS_DIR,
         "the parquet and CSV exports every other stage reads"),
    Root("taxonomy", "taxonomies", lambda: induction.TAXONOMY_DIR,
         "every induced category tree, all versions"),
    Root("labels", "labels", lambda: LABELS_DIR,
         "every response's category assignments, all runs"),
    Root("subthemes", "sub-themes", lambda: subthemes.SUBTHEMES_DIR,
         "the second-level breakdown inside categories"),
    Root("summary", "summary", lambda: SUMMARY_DIR,
         "the rendered dataset summary"),
    Root("lexicon", "keyword lexicon", lambda: lexicon.LEXICON_DIR,
         "the derived keyword concepts"),
    Root("locations", "place concepts", lambda: locations.LOCATIONS_DIR,
         "the canonicalized place mentions"),
    Root("rulings", "sameness rulings", lambda: rulings.RULINGS_DIR,
         "analyst decisions about which categories mean the same thing"),
    Root("review", "review reports", lambda: _data_dir() / "review",
         "taxonomy defect reports and their edit files"),
    Root("answers", "answers", lambda: _data_dir() / "answers",
         "every saved answer and the whole ask cache"),
    Root("jobs", "pipeline job logs", lambda: _data_dir() / "jobs",
         "the run logs proving what was spent and when"),
)

# Manifests whose recorded spend counts toward "you cannot get this back".
# Every family writes `usage.est_cost_usd`; the ones that do not are simply
# skipped rather than guessed at.
_COST_ROOTS = ("taxonomy", "labels", "subthemes", "lexicon", "locations")


@dataclass
class RootPreview:
    key: str
    label: str
    describes: str
    exists: bool = False
    files: int = 0
    bytes: int = 0
    # runs recorded under this root, where the layout has run directories
    runs: int = 0
    # model spend recorded in this root's manifests
    spent_usd: float = 0.0


@dataclass
class Preview:
    dataset_id: str
    roots: list[RootPreview] = field(default_factory=list)
    total_files: int = 0
    total_bytes: int = 0
    total_spent_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "roots": [vars(r) for r in self.roots],
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "total_spent_usd": round(self.total_spent_usd, 4),
        }


def dataset_dir(root: Root, dataset_id: str | int) -> Path:
    return root.path() / str(dataset_id)


def _walk(path: Path) -> tuple[int, int]:
    """(files, bytes) under `path`. A file that vanishes or cannot be statted
    mid-walk counts as zero rather than raising: this is a preview, and a
    racing writer must not turn it into a 500."""
    files = total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                files += 1
                total += child.stat().st_size
        except OSError:
            continue
    return files, total


def _recorded_spend(path: Path) -> tuple[int, float]:
    """(runs, dollars) recorded in the manifests under `path`.

    Counts `usage.est_cost_usd` from every manifest.json found. Deliberately
    counts every run, not just the latest: the point is what was spent to
    produce this tree, and superseded runs were paid for too.
    """
    runs = 0
    spent = 0.0
    for manifest_path in path.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        runs += 1
        cost = (manifest.get("usage") or {}).get("est_cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            spent += float(cost)
    return runs, spent


def preview(dataset_id: str | int) -> Preview:
    """What deleting this dataset would destroy. Reads only."""
    result = Preview(dataset_id=str(dataset_id))
    for root in ROOTS:
        path = dataset_dir(root, dataset_id)
        entry = RootPreview(key=root.key, label=root.label,
                            describes=root.describes)
        if path.is_dir():
            entry.exists = True
            entry.files, entry.bytes = _walk(path)
            if root.key in _COST_ROOTS:
                entry.runs, spent = _recorded_spend(path)
                entry.spent_usd = round(spent, 4)
                result.total_spent_usd += spent
            elif root.key == "jobs":
                entry.runs = sum(1 for p in path.iterdir() if p.is_dir())
        result.roots.append(entry)
        result.total_files += entry.files
        result.total_bytes += entry.bytes
    return result


def running_job(dataset_id: str | int):
    """The in-flight pipeline job for this dataset, or None.

    Imported lazily for the same reason ingest.py does it: this module must
    stay importable without the pipeline runner loaded.
    """
    from . import pipeline

    job = pipeline.latest_job(str(dataset_id))
    return job if job is not None and job.status == "running" else None


def purge_files(dataset_id: str | int) -> dict[str, str]:
    """Remove every on-disk artifact for `dataset_id`.

    Returns `{root_key: error}` for roots that could not be fully removed —
    an empty dict means everything went. Failures are collected rather than
    raised because the database rows are already gone by the time this runs:
    stopping halfway would leave MORE orphans than continuing, and the caller
    needs to be able to name what is left.
    """
    failures: dict[str, str] = {}
    for root in ROOTS:
        path = dataset_dir(root, dataset_id)
        if not path.exists():
            # normal: most datasets have no review, rulings, or subthemes
            continue
        errors: list[str] = []
        # onexc rather than ignore_errors: a file held open by a reader — the
        # ordinary Windows failure — is exactly the case the operator must be
        # told about, because id reuse will later graft what is left onto a
        # different dataset. (onexc is the 3.12 spelling; onerror is deprecated.)
        shutil.rmtree(
            path,
            onexc=lambda _fn, target, exc: errors.append(
                f"{Path(target).name}: {exc}"),
        )
        if path.exists() or errors:
            failures[root.key] = "; ".join(errors[:3]) or "directory remained"
    return failures


def forget_in_memory(dataset_id: str | int) -> None:
    """Drop this dataset from the caches that outlive its files.

    Two of them, and both matter. The ask context cache would otherwise answer
    from a snapshot of a dataset that no longer exists for up to its TTL. The
    pipeline job registry is keyed by job id, so nothing prunes it by dataset;
    a finished job for a deleted dataset is harmless but its presence makes
    `latest_job` report on something that is gone.
    """
    from . import pipeline

    ask_service.invalidate_context_cache(dataset_id)
    with pipeline._LOCK:
        stale = [job_id for job_id, job in pipeline._JOBS.items()
                 if job.dataset_id == str(dataset_id)]
        for job_id in stale:
            pipeline._JOBS.pop(job_id, None)
