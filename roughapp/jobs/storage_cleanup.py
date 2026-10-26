"""Remove on-disk files that no longer have a database row."""
from __future__ import annotations

import logging
from pathlib import Path

from roughapp.db.engine import SessionLocal
from roughapp.db.models import TreeRun
from roughapp.db.storage import FileStorage, LocalFileStorage

logger = logging.getLogger(__name__)


def _artifact_paths_in_db() -> set[Path]:
    with SessionLocal() as db:
        rows = (
            db.query(TreeRun.artifact_path)
            .filter(TreeRun.artifact_path.isnot(None))
            .all()
        )
    return {Path(p).resolve() for (p,) in rows if p}


def cleanup_orphan_tree_artifacts(storage: FileStorage | None = None) -> int:
    """Delete tree text files not referenced by any tree_run.artifact_path."""
    storage = storage or LocalFileStorage()
    root = Path(storage.tree_artifact_key(0)).parent
    known = _artifact_paths_in_db()
    removed = 0
    for path in root.glob("*.txt"):
        if path.resolve() not in known:
            storage.delete(str(path))
            removed += 1
    if removed:
        logger.info("Removed %d orphan tree artifact(s)", removed)
    return removed
