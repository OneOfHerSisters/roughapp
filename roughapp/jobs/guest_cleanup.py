"""Remove expired guest sessions and their files.

Runnable as a one-shot CLI (for host cron or a compose sidecar) or from the
in-process scheduler started at API startup.
"""
from __future__ import annotations

import logging

from roughapp.db.engine import SessionLocal
from roughapp.db.guest_sessions import cleanup_expired_guest_sessions
from roughapp.db.storage import LocalFileStorage

logger = logging.getLogger(__name__)
_storage = LocalFileStorage()


def run_guest_cleanup_once() -> int:
    """Delete all expired guest sessions; return how many were removed."""
    with SessionLocal() as db:
        removed = cleanup_expired_guest_sessions(db, storage=_storage)
    if removed:
        logger.info("Removed %d expired guest session(s)", removed)
    return removed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = run_guest_cleanup_once()
    print(f"removed={n}")
