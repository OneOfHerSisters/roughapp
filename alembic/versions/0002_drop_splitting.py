"""drop splitting leftovers

Remove max_subtables, multi-output sort_order, and the splitting strategy
lookup row. Existing splitting experiment results (and their output datasets)
are deleted so the new 1:1 result→dataset rule can be enforced.

Revision ID: 0002_drop_splitting
Revises: 0001_initial
Create Date: 2026-08-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_drop_splitting"
down_revision: Union[str, Sequence[str], None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(conn, name: str) -> bool:
    return sa.inspect(conn).has_table(name)


def _has_column(conn, table: str, column: str) -> bool:
    if not _has_table(conn, table):
        return False
    return column in {c["name"] for c in sa.inspect(conn).get_columns(table)}


def _has_unique(conn, table: str, name: str) -> bool:
    if not _has_table(conn, table):
        return False
    return any(u["name"] == name for u in sa.inspect(conn).get_unique_constraints(table))


def upgrade() -> None:
    conn = op.get_bind()

    if _has_table(conn, "strategies"):
        splitting = conn.execute(
            sa.text("SELECT id FROM strategies WHERE name = 'splitting'")
        ).fetchone()
        if splitting is not None:
            strategy_id = splitting[0]
            result_ids = [
                row[0]
                for row in conn.execute(
                    sa.text(
                        "SELECT id FROM experiment_results WHERE strategy_id = :sid"
                    ),
                    {"sid": strategy_id},
                ).fetchall()
            ]
            dataset_ids: list[int] = []
            if result_ids:
                # Dialect-portable: delete one result at a time is fine for cleanup.
                for rid in result_ids:
                    dataset_ids.extend(
                        row[0]
                        for row in conn.execute(
                            sa.text(
                                "SELECT dataset_id FROM experiment_result_outputs "
                                "WHERE result_id = :rid"
                            ),
                            {"rid": rid},
                        ).fetchall()
                    )
                conn.execute(
                    sa.text(
                        "DELETE FROM experiment_results WHERE strategy_id = :sid"
                    ),
                    {"sid": strategy_id},
                )
                for did in dataset_ids:
                    conn.execute(
                        sa.text("DELETE FROM datasets WHERE id = :did"),
                        {"did": did},
                    )
            conn.execute(
                sa.text("DELETE FROM strategies WHERE id = :sid"),
                {"sid": strategy_id},
            )

    if _has_column(conn, "experiment_result_outputs", "sort_order"):
        # Keep the lowest sort_order row per result when duplicates exist.
        conn.execute(
            sa.text(
                """
                DELETE FROM experiment_result_outputs
                WHERE id NOT IN (
                    SELECT MIN(id) FROM experiment_result_outputs GROUP BY result_id
                )
                """
            )
        )

    if _has_column(conn, "experiments", "max_subtables"):
        op.drop_column("experiments", "max_subtables")

    if _has_unique(conn, "experiment_result_outputs", "uq_experiment_result_outputs_result_sort"):
        op.drop_constraint(
            "uq_experiment_result_outputs_result_sort",
            "experiment_result_outputs",
            type_="unique",
        )
    if _has_column(conn, "experiment_result_outputs", "sort_order"):
        op.drop_column("experiment_result_outputs", "sort_order")
    if (
        _has_table(conn, "experiment_result_outputs")
        and not _has_unique(conn, "experiment_result_outputs", "uq_experiment_result_outputs_result")
    ):
        op.create_unique_constraint(
            "uq_experiment_result_outputs_result",
            "experiment_result_outputs",
            ["result_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _has_unique(conn, "experiment_result_outputs", "uq_experiment_result_outputs_result"):
        op.drop_constraint(
            "uq_experiment_result_outputs_result",
            "experiment_result_outputs",
            type_="unique",
        )
    if not _has_column(conn, "experiment_result_outputs", "sort_order"):
        op.add_column(
            "experiment_result_outputs",
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        )
    if not _has_unique(conn, "experiment_result_outputs", "uq_experiment_result_outputs_result_sort"):
        op.create_unique_constraint(
            "uq_experiment_result_outputs_result_sort",
            "experiment_result_outputs",
            ["result_id", "sort_order"],
        )
    if _has_column(conn, "experiment_result_outputs", "sort_order"):
        op.alter_column("experiment_result_outputs", "sort_order", server_default=None)

    if not _has_column(conn, "experiments", "max_subtables"):
        op.add_column(
            "experiments",
            sa.Column("max_subtables", sa.Integer(), nullable=True),
        )
    op.execute(
        sa.text(
            "INSERT INTO strategies (name, description) VALUES "
            "('splitting', 'Cartesian product of per-class resolutions into subtables')"
        )
    )
