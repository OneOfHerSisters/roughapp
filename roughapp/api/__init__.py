"""
FastAPI application entry point.

Sets up the application and registers the auth, datasets, and experiments routers.
The API layer exists only to expose the existing service logic over HTTP —
it does not contain business logic of its own.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from roughapp.api.routers import datasets, experiments, auth, trees
from roughapp.api.exception_handlers import register_exception_handlers
from roughapp.db.engine import init_db, SessionLocal
from roughapp.db.guest_sessions import cleanup_expired_guest_sessions
from roughapp.db.seed import seed_builtin_datasets
from roughapp.db.reference_seed import seed_reference_tables
from roughapp.db.storage import LocalFileStorage
from roughapp.jobs.scheduler import start_background_jobs, stop_background_jobs

_storage = LocalFileStorage()

app = FastAPI(title="roughapp")
register_exception_handlers(app)


@app.on_event("startup")
def startup():
    init_db()
    with SessionLocal() as db:
        cleanup_expired_guest_sessions(db, storage=_storage)
        seed_reference_tables(db)
        seed_builtin_datasets(db)
    from roughapp.jobs.storage_cleanup import cleanup_orphan_tree_artifacts
    cleanup_orphan_tree_artifacts(_storage)
    start_background_jobs()


@app.on_event("shutdown")
def shutdown():
    stop_background_jobs()

app.include_router(auth.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(trees.router, prefix="/api")

_FRONTEND = Path(__file__).parent.parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return _FRONTEND.read_text(encoding="utf-8")
