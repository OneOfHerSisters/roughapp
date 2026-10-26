from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, JSON, BigInteger,
    UniqueConstraint, CheckConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.dialects.postgresql import JSONB

# Canonical closed value sets — validated at the API layer. Seeded into lookup
# tables on startup; tests/test_schema_constants.py keeps them in sync with the
# resolution strategy signatures.
MCD_TIE_POLICIES = ("min", "keep", "first", "random", "raise")
QR_TIE_POLICIES = ("keep", "random", "raise")
ALL_STRATEGIES = ("mcd", "rough-approximation", "gd")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Normalize DB datetimes for comparison (SQLite may return naive UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _metric_value(metrics, name: str):
    for m in metrics:
        if m.metric_name == name:
            return m.value
    return None


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(254), unique=True, nullable=False)
    password_hash = Column(String(60), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    datasets = relationship("Dataset", back_populates="user")
    experiments = relationship("Experiment", back_populates="user")


class GuestSession(Base):
    __tablename__ = "guest_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="ck_guest_sessions_expires_after_created"),
    )

    token_hash = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    datasets = relationship("Dataset", back_populates="guest_session")
    experiments = relationship("Experiment", back_populates="guest_session")


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(512), nullable=False, default="")

    results = relationship("ExperimentResult", back_populates="strategy_ref")


class TreeAlgorithm(Base):
    __tablename__ = "tree_algorithms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(512), nullable=False, default="")

    runs = relationship("TreeRun", back_populates="algorithm")


class McdTiePolicy(Base):
    __tablename__ = "mcd_tie_policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32), unique=True, nullable=False)
    description = Column(String(512), nullable=False, default="")

    experiments = relationship("Experiment", back_populates="mcd_policy")


class QrTiePolicy(Base):
    __tablename__ = "qr_tie_policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32), unique=True, nullable=False)
    description = Column(String(512), nullable=False, default="")

    experiments = relationship("Experiment", back_populates="qr_policy")


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("user_id", "storage_path", name="uq_datasets_user_storage_path"),
        UniqueConstraint("guest_session_id", "storage_path", name="uq_datasets_guest_storage_path"),
        CheckConstraint(
            "(user_id IS NOT NULL AND guest_session_id IS NULL) "
            "OR (user_id IS NULL AND guest_session_id IS NOT NULL) "
            "OR (user_id IS NULL AND guest_session_id IS NULL)",
            name="ck_datasets_owner_mode",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    storage_path = Column(String(1024), nullable=False)
    rows = Column(Integer)
    cols = Column(Integer)
    decision_col = Column(String(255), nullable=False, default="")
    gamma_actual = Column(Float, nullable=True)
    alpha_actual = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    guest_session_id = Column(Integer, ForeignKey("guest_sessions.id"), nullable=True)

    user = relationship("User", back_populates="datasets")
    guest_session = relationship("GuestSession", back_populates="datasets")
    provenance = relationship(
        "DatasetProvenance",
        back_populates="dataset",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="DatasetProvenance.dataset_id",
    )
    produced_by_result = relationship(
        "ExperimentResult",
        back_populates="output_dataset",
        uselist=False,
        foreign_keys="ExperimentResult.output_dataset_id",
    )
    experiments = relationship("Experiment", back_populates="dataset")
    tree_runs = relationship("TreeRun", back_populates="dataset", cascade="all, delete-orphan")


class DatasetProvenance(Base):
    """How a dataset was produced — separate from the stored table (3NF)."""
    __tablename__ = "dataset_provenance"

    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True,
    )
    kind = Column(String(32), nullable=False)
    source_dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True,
    )

    dataset = relationship(
        "Dataset", back_populates="provenance", foreign_keys=[dataset_id],
    )
    source = relationship("Dataset", foreign_keys=[source_dataset_id])


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL AND guest_session_id IS NULL) "
            "OR (user_id IS NULL AND guest_session_id IS NOT NULL)",
            name="ck_experiments_owner_xor",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    guest_session_id = Column(Integer, ForeignKey("guest_sessions.id"), nullable=True)
    random_state = Column(Integer, nullable=True)
    mcd_tie_policy_id = Column(Integer, ForeignKey("mcd_tie_policies.id"), nullable=True)
    qr_tie_policy_id = Column(Integer, ForeignKey("qr_tie_policies.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    dataset = relationship("Dataset", back_populates="experiments")
    user = relationship("User", back_populates="experiments")
    guest_session = relationship("GuestSession", back_populates="experiments")
    mcd_policy = relationship("McdTiePolicy", back_populates="experiments")
    qr_policy = relationship("QrTiePolicy", back_populates="experiments")
    results = relationship("ExperimentResult", back_populates="experiment", cascade="all, delete-orphan")

    @property
    def tie_policy(self) -> str | None:
        return self.mcd_policy.name if self.mcd_policy else None

    @property
    def qr_tie_policy(self) -> str | None:
        return self.qr_policy.name if self.qr_policy else None


class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    __table_args__ = (
        UniqueConstraint("experiment_id", "strategy_id", name="uq_experiment_results_experiment_strategy"),
        UniqueConstraint("output_dataset_id", name="uq_experiment_results_output_dataset"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    output_dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True,
    )
    time_ms = Column(BigInteger, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    details = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)

    experiment = relationship("Experiment", back_populates="results")
    strategy_ref = relationship("Strategy", back_populates="results")
    output_dataset = relationship(
        "Dataset",
        back_populates="produced_by_result",
        foreign_keys=[output_dataset_id],
    )
    metrics = relationship(
        "ExperimentMetric", back_populates="result",
        cascade="all, delete-orphan", foreign_keys="ExperimentMetric.result_id",
    )

    @property
    def strategy(self) -> str:
        return self.strategy_ref.name if self.strategy_ref else ""

    @property
    def gamma_after(self) -> float | None:
        return _metric_value(self.metrics, "gamma")

    @property
    def alpha_after(self) -> float | None:
        return _metric_value(self.metrics, "alpha")

    @property
    def rows_after(self) -> int | None:
        v = _metric_value(self.metrics, "rows")
        return int(v) if v is not None else None

    @property
    def depth_after(self) -> int | None:
        v = _metric_value(self.tree_metrics, "depth")
        return int(v) if v is not None else None

    @property
    def avg_depth_after(self) -> float | None:
        return _metric_value(self.tree_metrics, "avg_depth")

    @property
    def nodes_after(self) -> int | None:
        v = _metric_value(self.tree_metrics, "nodes")
        return int(v) if v is not None else None

    @property
    def n_gd_codes(self) -> int | None:
        v = _metric_value(self.metrics, "n_gd_codes")
        return int(v) if v is not None else None

    @property
    def n_gd_conflict_codes(self) -> int | None:
        v = _metric_value(self.metrics, "n_gd_conflict_codes")
        return int(v) if v is not None else None

    @property
    def result_path(self) -> str | None:
        return self.output_dataset.storage_path if self.output_dataset else None

    @property
    def tree_run(self) -> "TreeRun | None":
        if self.output_dataset is None:
            return None
        for run in self.output_dataset.tree_runs:
            if run.artifact_path:
                return run
        return None

    @property
    def tree_metrics(self) -> list["TreeMetric"]:
        run = self.tree_run
        return run.metrics if run else []


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"
    __table_args__ = (
        UniqueConstraint("result_id", "metric_name", name="uq_experiment_metrics_result_metric"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey("experiment_results.id", ondelete="CASCADE"), nullable=False)
    metric_name = Column(String(64), nullable=False)
    value = Column(Float, nullable=True)

    result = relationship("ExperimentResult", back_populates="metrics", foreign_keys=[result_id])


class TreeRun(Base):
    __tablename__ = "tree_runs"
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL AND guest_session_id IS NULL) "
            "OR (user_id IS NULL AND guest_session_id IS NOT NULL) "
            "OR (user_id IS NULL AND guest_session_id IS NULL)",
            name="ck_tree_runs_owner_mode",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    guest_session_id = Column(Integer, ForeignKey("guest_sessions.id"), nullable=True)
    tree_algorithm_id = Column(Integer, ForeignKey("tree_algorithms.id"), nullable=False)
    params_json = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    params_hash = Column(String(64), nullable=False, default="")
    artifact_path = Column(String(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    dataset = relationship("Dataset", back_populates="tree_runs")
    user = relationship("User")
    guest_session = relationship("GuestSession")
    algorithm = relationship("TreeAlgorithm", back_populates="runs")
    metrics = relationship("TreeMetric", back_populates="tree_run", cascade="all, delete-orphan")


class TreeMetric(Base):
    __tablename__ = "tree_metrics"
    __table_args__ = (
        UniqueConstraint("tree_run_id", "metric_name", name="uq_tree_metrics_run_metric"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    tree_run_id = Column(Integer, ForeignKey("tree_runs.id", ondelete="CASCADE"), nullable=False)
    metric_name = Column(String(64), nullable=False)
    value = Column(Float, nullable=True)

    tree_run = relationship("TreeRun", back_populates="metrics")


Index("ix_experiments_dataset_created", Experiment.dataset_id, Experiment.created_at.desc())
Index("ix_experiments_user_created", Experiment.user_id, Experiment.created_at.desc())
Index("ix_experiments_guest_created", Experiment.guest_session_id, Experiment.created_at.desc())
Index("ix_datasets_user_created", Dataset.user_id, Dataset.created_at.desc())
Index("ix_datasets_guest_created", Dataset.guest_session_id, Dataset.created_at.desc())
Index("ix_dataset_provenance_source", DatasetProvenance.source_dataset_id)
Index("ix_guest_sessions_expires_at", GuestSession.expires_at)
Index("ix_tree_runs_dataset_id", TreeRun.dataset_id)
