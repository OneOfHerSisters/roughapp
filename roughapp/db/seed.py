from pathlib import Path

from sqlalchemy.orm import Session

from roughapp.business.dataset_provenance import KIND_UPLOAD
from roughapp.core.csv_table import read_csv_table
from roughapp.core.metrics import quality_of_classification, accuracy_of_classification
from roughapp.db.models import Dataset, DatasetProvenance

BUILTIN_DATASETS = [
    ("breast-cancer",   "data/modified_breast-cancer.csv"),
    ("hayes-roth-data", "data/modified_hayes-roth-data.csv"),
    ("cars",            "data/modified_cars.csv"),
    ("lymphography",    "data/modified_lymphography.csv"),
    ("zoo-data",        "data/modified_zoo-data.csv"),
    ("tic-tac-toe",     "data/modified_tic-tac-toe.csv"),
    ("mushroom",        "data/modified_mushroom.csv"),
    ("nursery",         "data/modified_nursery.csv"),
]




def _add_dataset(db: Session, name: str, path: str) -> None:
    if not Path(path).exists():
        return
    existing = (
        db.query(Dataset)
        .filter(
            Dataset.name == name,
            Dataset.user_id.is_(None),
            Dataset.guest_session_id.is_(None),
        )
        .first()
    )
    if existing:
        return
    try:
        df, _ = read_csv_table(path)
        decision_col = str(df.columns[-1])
        gamma = round(quality_of_classification(df, decision_col), 4)
        alpha = round(accuracy_of_classification(df, decision_col), 4)
        ds = Dataset(
            name=name,
            storage_path=path,
            rows=len(df),
            cols=len(df.columns),
            decision_col=decision_col,
            gamma_actual=gamma,
            alpha_actual=alpha,
        )
        ds.provenance = DatasetProvenance(kind=KIND_UPLOAD)
        db.add(ds)
    except Exception:
        pass


def seed_builtin_datasets(db: Session) -> None:
    for name, path in BUILTIN_DATASETS:
        _add_dataset(db, name, path)
    db.commit()
