from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from roughapp.db.models import Dataset, Experiment, GuestSession, ensure_utc, utcnow
from roughapp.db.repositories import DatasetRepository, apply_dataset_deletion_plan
from roughapp.db.storage import FileStorage, LocalFileStorage


def delete_guest_session_data(
    db: Session,
    guest_session: GuestSession,
    storage: FileStorage | None = None,
) -> None:
    """Delete one guest session with all owned rows and best-effort file cleanup."""
    storage = storage or LocalFileStorage()
    repo = DatasetRepository(db)
    plans = [repo.deletion_plan(ds.id) for ds in db.query(Dataset).filter_by(guest_session_id=guest_session.id).all()]

    for exp in db.query(Experiment).filter_by(guest_session_id=guest_session.id).all():
        db.delete(exp)

    for ds in db.query(Dataset).filter_by(guest_session_id=guest_session.id).all():
        db.delete(ds)

    db.delete(guest_session)

    for plan in plans:
        apply_dataset_deletion_plan(plan, storage)


def cleanup_expired_guest_sessions(
    db: Session,
    now: datetime | None = None,
    storage: FileStorage | None = None,
) -> int:
    """Delete expired guest sessions and return how many were removed."""
    current = ensure_utc(now) if now is not None else utcnow()
    expired = db.query(GuestSession).filter(GuestSession.expires_at <= current).all()
    for guest_session in expired:
        delete_guest_session_data(db, guest_session, storage=storage)
    if expired:
        db.commit()
    return len(expired)
