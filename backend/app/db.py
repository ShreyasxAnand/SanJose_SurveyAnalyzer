from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = DATA_DIR / "exports"
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR / 'survey_analyzer.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  (ensures models are registered)

    Base.metadata.create_all(bind=engine)
    _ensure_columns()


# Columns added to tables that predate them. create_all only creates missing
# TABLES, never missing columns, and this project has no migration framework —
# so each additive column gets one idempotent ALTER here (the same move
# scripts/backfill_uploads.py documented for responses.upload_id, made
# automatic so a git pull never leaves a DB the ORM can't query).
_ADDITIVE_COLUMNS = (
    ("datasets", "date_ranges_json", "TEXT"),
    ("metadata_columns", "value_type", "VARCHAR DEFAULT 'categorical' NOT NULL"),
)


def _ensure_columns() -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for table, column, ddl_type in _ADDITIVE_COLUMNS:
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            if existing and column not in existing:
                conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                )
