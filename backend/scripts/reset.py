"""Drop and recreate all tables, and clear ./data/uploads and ./data/exports.

Run repeatedly while iterating on the ingest transform, so each test upload
starts from a clean slate instead of half-committed state from a prior run.

    cd backend
    python -m scripts.reset
"""

import shutil

from app.db import Base, EXPORTS_DIR, UPLOADS_DIR, engine
from app import models  # noqa: F401  (registers tables on Base.metadata)


def main() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    for directory in (UPLOADS_DIR, EXPORTS_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    print("Reset complete: tables recreated, ./data/uploads and ./data/exports cleared.")


if __name__ == "__main__":
    main()
