"""Guest session cleanup removes DB rows and on-disk files via deletion_plan."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from roughapp.db.guest_sessions import delete_guest_session_data
from roughapp.db.models import Base, Dataset, Experiment, GuestSession, TreeRun, utcnow
from roughapp.db.reference_seed import seed_reference_tables
from roughapp.db.storage import LocalFileStorage


@pytest.fixture()
def guest_cleanup_env(tmp_path: Path):
    upload_dir = tmp_path / "uploads"
    tree_dir = tmp_path / "tree_artifacts"
    storage = LocalFileStorage(upload_dir=upload_dir, tree_artifacts_dir=tree_dir)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    seed_reference_tables(db)
    db.commit()
    yield db, storage
    db.close()


def test_delete_guest_session_removes_csv_and_tree_artifacts(guest_cleanup_env):
    db, storage = guest_cleanup_env
    guest = GuestSession(token_hash="hash", expires_at=utcnow() + timedelta(hours=1))
    db.add(guest)
    db.flush()

    csv_path = storage.upload_key(1, "guest.csv")
    Path(csv_path).write_text("a,dec\n1,P\n", encoding="utf-8")
    ds = Dataset(
        name="guest.csv",
        storage_path=csv_path,
        rows=1,
        cols=2,
        decision_col="dec",
        guest_session_id=guest.id,
    )
    db.add(ds)
    db.flush()

    tree_run = TreeRun(
        dataset_id=ds.id,
        guest_session_id=guest.id,
        tree_algorithm_id=1,
        artifact_path=storage.tree_artifact_key(1),
    )
    db.add(tree_run)
    db.flush()
    tree_run.artifact_path = storage.tree_artifact_key(tree_run.id)
    Path(tree_run.artifact_path).write_text("|--- class=0", encoding="utf-8")
    db.commit()

    delete_guest_session_data(db, guest, storage=storage)
    db.commit()

    assert db.query(GuestSession).count() == 0
    assert db.query(Dataset).count() == 0
    assert db.query(TreeRun).count() == 0
    assert not Path(csv_path).exists()
    assert not Path(tree_run.artifact_path).exists()
