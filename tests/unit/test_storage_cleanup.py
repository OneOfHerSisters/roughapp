"""Orphan tree artifact files are removed when no tree_run points at them."""
from __future__ import annotations

from pathlib import Path

from roughapp.db.models import TreeRun
from roughapp.jobs.storage_cleanup import cleanup_orphan_tree_artifacts
from roughapp.db.storage import LocalFileStorage


def test_cleanup_orphan_tree_artifacts(tmp_path: Path):
    tree_dir = tmp_path / "tree_artifacts"
    upload_dir = tmp_path / "uploads"
    storage = LocalFileStorage(upload_dir=upload_dir, tree_artifacts_dir=tree_dir)

    orphan = tree_dir / "99.txt"
    orphan.write_text("orphan tree", encoding="utf-8")
    kept = tree_dir / "1.txt"
    kept.write_text("kept tree", encoding="utf-8")

    # Patch DB lookup: only 1.txt is known.
    import roughapp.jobs.storage_cleanup as mod

    original = mod._artifact_paths_in_db

    def fake_known():
        return {kept.resolve()}

    mod._artifact_paths_in_db = fake_known
    try:
        removed = cleanup_orphan_tree_artifacts(storage)
    finally:
        mod._artifact_paths_in_db = original

    assert removed == 1
    assert not orphan.exists()
    assert kept.exists()
