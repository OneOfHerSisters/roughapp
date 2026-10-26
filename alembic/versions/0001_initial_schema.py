"""initial schema

Single baseline migration. The schema is defined by the SQLAlchemy models in
roughapp.db.models, and this migration materializes them in one step: every
table, index, unique and CHECK constraint is created exactly as the ORM
declares it, so the migrated schema and the models can never drift apart.

It replaces the incremental migration history that was squashed once the
schema stabilized; the reference tables (strategies, tie policies, tree
algorithms) are seeded by the application at startup, not here.

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-13
"""
from typing import Sequence, Union

from alembic import op

from roughapp.db.models import Base

revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
