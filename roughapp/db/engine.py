from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from roughapp.config import get_settings

settings = get_settings()

# SQLite needs check_same_thread=False for FastAPI's threadpool; Postgres and
# other backends take no such connect arg.
if settings.is_sqlite:
    db_file = settings.database_url.replace("sqlite:///", "", 1)
    if db_file:
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(settings.database_url, connect_args=connect_args)

if settings.is_sqlite:
    # SQLite ships with foreign-key checks DISABLED per connection. Without
    # this pragma, deleting a referenced row silently orphans its children —
    # the schema would only be enforced on PostgreSQL, and the two engines
    # would behave differently.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_fks(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Bring the database schema up to date via Alembic migrations.

    Migrations are the single source of truth for the schema — for a fresh
    database this creates every table, for an existing one it applies any
    pending revisions.  Replaces the old create_all(), which could silently
    drift from the models (as happened when tree-stat columns were added).
    """
    from pathlib import Path
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent.parent
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(cfg, "head")


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
