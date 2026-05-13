"""
SQLite + SQLAlchemy persistence layer.

Stores a structured copy of each /evaluate run alongside the existing
JSON sibling in OUTPUT_DIR. The JSON file remains the source of truth
for the renderer — the DB is additive shadow storage that enables
queryable history, teacher-override audit trails, and the future
analytics dashboard.

Single-file SQLite DB lives next to the backend code so it travels
with the deployment unit. PRAGMA foreign_keys=ON is set on every new
connection to enforce ON DELETE CASCADE for exam re-runs.
"""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "exam_demo.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

Base = declarative_base()


def init_db():
    """Idempotent: registers all ORM models then creates missing tables."""
    import db_models  # noqa: F401  - registers tables on Base.metadata
    Base.metadata.create_all(bind=engine)
