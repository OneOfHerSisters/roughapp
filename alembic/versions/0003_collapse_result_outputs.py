"""collapse experiment_result_outputs into FK

Each experiment result owns at most one output dataset. Replace the join
table with experiment_results.output_dataset_id.

Revision ID: 0003_collapse_outputs
Revises: 0002_drop_splitting
Create Date: 2026-08-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_collapse_outputs"
down_revision: Union[str, Sequence[str], None] = "0002_drop_splitting"
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


def _has_fk(conn, table: str, name: str) -> bool:
    if not _has_table(conn, table):
        return False
    return any(fk["name"] == name for fk in sa.inspect(conn).get_foreign_keys(table))


def upgrade() -> None:
    conn = op.get_bind()

    if not _has_column(conn, "experiment_results", "output_dataset_id"):
        op.add_column(
            "experiment_results",
            sa.Column("output_dataset_id", sa.Integer(), nullable=True),
        )
        if _has_table(conn, "experiment_result_outputs"):
            conn.execute(
                sa.text(
                    """
                    UPDATE experiment_results
                    SET output_dataset_id = (
                        SELECT dataset_id
                        FROM experiment_result_outputs
                        WHERE experiment_result_outputs.result_id = experiment_results.id
                        LIMIT 1
                    )
                    """
                )
            )
        if not _has_fk(conn, "experiment_results", "fk_experiment_results_output_dataset"):
            op.create_foreign_key(
                "fk_experiment_results_output_dataset",
                "experiment_results",
                "datasets",
                ["output_dataset_id"],
                ["id"],
                ondelete="SET NULL",
            )
        if not _has_unique(conn, "experiment_results", "uq_experiment_results_output_dataset"):
            op.create_unique_constraint(
                "uq_experiment_results_output_dataset",
                "experiment_results",
                ["output_dataset_id"],
            )

    if _has_table(conn, "experiment_result_outputs"):
        op.drop_table("experiment_result_outputs")


def downgrade() -> None:
    conn = op.get_bind()

    if not _has_table(conn, "experiment_result_outputs"):
        op.create_table(
            "experiment_result_outputs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "result_id",
                sa.Integer(),
                sa.ForeignKey("experiment_results.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "dataset_id",
                sa.Integer(),
                sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.UniqueConstraint("result_id", name="uq_experiment_result_outputs_result"),
            sa.UniqueConstraint("dataset_id", name="uq_experiment_result_outputs_dataset"),
        )
        if _has_column(conn, "experiment_results", "output_dataset_id"):
            conn.execute(
                sa.text(
                    """
                    INSERT INTO experiment_result_outputs (result_id, dataset_id)
                    SELECT id, output_dataset_id
                    FROM experiment_results
                    WHERE output_dataset_id IS NOT NULL
                    """
                )
            )

    if _has_column(conn, "experiment_results", "output_dataset_id"):
        if _has_fk(conn, "experiment_results", "fk_experiment_results_output_dataset"):
            op.drop_constraint(
                "fk_experiment_results_output_dataset",
                "experiment_results",
                type_="foreignkey",
            )
        if _has_unique(conn, "experiment_results", "uq_experiment_results_output_dataset"):
            op.drop_constraint(
                "uq_experiment_results_output_dataset",
                "experiment_results",
                type_="unique",
            )
        op.drop_column("experiment_results", "output_dataset_id")
