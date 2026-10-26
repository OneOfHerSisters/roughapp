"""Lightweight in-process scheduler for periodic maintenance."""
from __future__ import annotations

import logging
import threading

from roughapp.config import get_settings
from roughapp.jobs.maintenance import run_maintenance_once

logger = logging.getLogger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None


def _loop(interval_seconds: int) -> None:
    while not _stop.wait(interval_seconds):
        try:
            run_maintenance_once()
        except Exception:
            logger.exception("Scheduled maintenance failed")


def start_background_jobs() -> None:
    """Start daemon threads for configured periodic jobs."""
    global _thread
    interval_min = get_settings().guest_cleanup_interval_minutes
    if interval_min <= 0:
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_loop,
        args=(interval_min * 60,),
        name="guest-cleanup",
        daemon=True,
    )
    _thread.start()
    logger.info("Guest cleanup scheduled every %d minute(s)", interval_min)


def stop_background_jobs() -> None:
    _stop.set()
