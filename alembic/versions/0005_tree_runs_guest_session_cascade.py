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

_tables = ["datasets", "experiments", "tree_runs"]


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    for table in _tables:
        with op.batch_alter_table(table) as batch_op:
            if not _is_sqlite():
                batch_op.drop_constraint(
                    f"{table}_guest_session_id_fkey", type_="foreignkey"
                )
            batch_op.create_foreign_key(
                f"{table}_guest_session_id_fkey",
                "guest_sessions",
                ["guest_session_id"], ["id"],
                ondelete="CASCADE",
            )


def downgrade() -> None:
    for table in _tables:
        with op.batch_alter_table(table) as batch_op:
            if not _is_sqlite():
                batch_op.drop_constraint(
                    f"{table}_guest_session_id_fkey", type_="foreignkey"
                )
            batch_op.create_foreign_key(
                f"{table}_guest_session_id_fkey",
                "guest_sessions",
                ["guest_session_id"], ["id"],
            )
