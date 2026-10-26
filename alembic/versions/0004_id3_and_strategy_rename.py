"""drop MVD, rename qualitative-removal, replace tree algorithms with ID3

Removes the many-valued-decision strategy, renames qualitative removal to
rough approximation, and replaces the tree-algorithm lookup with a single
row for ID3. Experiment results and tree runs produced by the removed
entries are deleted, because their metrics are no longer reproducible.

Revision ID: 0004_id3_rename
Revises: 0003_collapse_outputs
Create Date: 2026-08-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_id3_rename"
down_revision: Union[str, Sequence[str], None] = "0003_collapse_outputs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(conn, name: str) -> bool:
    return sa.inspect(conn).has_table(name)


def _delete_results_for_strategy(conn, name: str) -> None:
    row = conn.execute(
        sa.text("SELECT id FROM strategies WHERE name = :n"), {"n": name}
    ).fetchone()
    if row is None:
        return
    conn.execute(
        sa.text("DELETE FROM experiment_results WHERE strategy_id = :sid"),
        {"sid": row[0]},
    )
    conn.execute(sa.text("DELETE FROM strategies WHERE id = :sid"), {"sid": row[0]})


def upgrade() -> None:
    conn = op.get_bind()

    if _has_table(conn, "strategies"):
        # MVD is gone: its results reference the strategy row, so they go first.
        _delete_results_for_strategy(conn, "mvd")
        target = conn.execute(
            sa.text("SELECT id FROM strategies WHERE name = 'rough-approximation'")
        ).fetchone()
        old = conn.execute(
            sa.text("SELECT id FROM strategies WHERE name = 'qualitative-removal'")
        ).fetchone()
        if target is not None and old is not None:
            # The new name was already seeded by a dev server: repoint results
            # of the old row and drop it instead of renaming into a collision.
            conn.execute(
                sa.text(
                    "UPDATE experiment_results SET strategy_id = :new "
                    "WHERE strategy_id = :old"
                ),
                {"new": target[0], "old": old[0]},
            )
            conn.execute(
                sa.text("DELETE FROM strategies WHERE id = :old"), {"old": old[0]}
            )
        elif old is not None:
            conn.execute(
                sa.text(
                    "UPDATE strategies SET name = :new, description = :descr "
                    "WHERE id = :old"
                ),
                {
                    "new": "rough-approximation",
                    "old": old[0],
                    "descr": "Qualitative removal: keep the decision whose class "
                             "has the highest accuracy of approximation",
                },
            )

    if _has_table(conn, "tree_algorithms"):
        # Tree runs built by the removed algorithms cannot be reproduced by ID3.
        old = conn.execute(
            sa.text("SELECT id FROM tree_algorithms WHERE name IN ('cart', 'greedy-u')")
        ).fetchall()
        for (algo_id,) in old:
            if _has_table(conn, "tree_runs"):
                conn.execute(
                    sa.text("DELETE FROM tree_runs WHERE tree_algorithm_id = :aid"),
                    {"aid": algo_id},
                )
            conn.execute(
                sa.text("DELETE FROM tree_algorithms WHERE id = :aid"), {"aid": algo_id}
            )
        exists = conn.execute(
            sa.text("SELECT id FROM tree_algorithms WHERE name = 'id3'")
        ).fetchone()
        if exists is None:
            conn.execute(
                sa.text(
                    "INSERT INTO tree_algorithms (name, description) "
                    "VALUES ('id3', :descr)"
                ),
                {"descr": "ID3 — information-gain tree with one branch per "
                          "attribute value (Quinlan)"},
            )


def downgrade() -> None:
    conn = op.get_bind()
    if _has_table(conn, "strategies"):
        conn.execute(
            sa.text("UPDATE strategies SET name = :old WHERE name = :new"),
            {"old": "qualitative-removal", "new": "rough-approximation"},
        )
        conn.execute(
            sa.text(
                "INSERT INTO strategies (name, description) VALUES "
                "('mvd', 'Many-valued decision (set-valued collapsed table)')"
            )
        )
    if _has_table(conn, "tree_algorithms"):
        conn.execute(
            sa.text(
                "INSERT INTO tree_algorithms (name, description) VALUES "
                "('cart', 'Classification and Regression Tree')"
            )
        )
