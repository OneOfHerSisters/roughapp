"""One-shot maintenance: guest sessions + orphan files on disk."""
from __future__ import annotations

import logging

from roughapp.jobs.guest_cleanup import run_guest_cleanup_once
from roughapp.jobs.storage_cleanup import cleanup_orphan_tree_artifacts

logger = logging.getLogger(__name__)


def run_maintenance_once() -> dict[str, int]:
    guests = run_guest_cleanup_once()
    orphans = cleanup_orphan_tree_artifacts()
    return {"guest_sessions": guests, "orphan_tree_artifacts": orphans}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_maintenance_once()
    print(f"guest_sessions={result['guest_sessions']} orphan_tree_artifacts={result['orphan_tree_artifacts']}")
