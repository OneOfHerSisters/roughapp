"""add ON DELETE CASCADE to guest_session_id FKs on datasets, experiments, tree_runs

Revision ID: 0005_guest_session_cascade
Revises: 0004_id3_rename
Create Date: 2026-11-14
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0005_guest_session_cascade"
down_revision: Union[str, Sequence[str], None] = "0004_id3_rename"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_fks = [
    ("datasets",    "datasets_guest_session_id_fkey"),
    ("experiments", "experiments_guest_session_id_fkey"),
    ("tree_runs",   "tree_runs_guest_session_id_fkey"),
]


def upgrade() -> None:
    for table, name in _fks:
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(
            name, table, "guest_sessions",
            ["guest_session_id"], ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    for table, name in _fks:
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(
            name, table, "guest_sessions",
            ["guest_session_id"], ["id"],
        )
