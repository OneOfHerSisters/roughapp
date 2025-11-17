import pandas as pd

from roughapp.core.grouping import ind_classes_table


def lu_per_decision(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    counts = ind_classes_table(df, decision_col)
    class_size = counts.sum(axis=1)

    rows = []
    for d in counts.columns:
        d_count = counts[d]
        L = int(class_size[(d_count == class_size) & (class_size > 0)].sum())
        U = int(class_size[(d_count > 0)].sum())
        rows.append({"decision": d, "L": L, "U": U, "boundary": U - L})

    return pd.DataFrame(rows).sort_values("decision").reset_index(drop=True)


def alpha_rho_per_decision(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    lu = lu_per_decision(df, decision_col)
    alphas = []

    for _, row in lu.iterrows():
        L = row["L"]
        U = row["U"]
        if U > 0:
            alpha_val = L / U
        else:
            alpha_val = 1.0
        alphas.append(alpha_val)

    lu["alpha"] = alphas
    lu["rho"] = 1.0 - lu["alpha"]
    return lu[["decision", "alpha", "rho", "L", "U", "boundary"]]


def boundary_summary(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    lu = lu_per_decision(df, decision_col)
    per_decision = lu[["decision", "boundary"]]
    total_boundary = int(per_decision["boundary"].sum())
    total_row = pd.DataFrame([{"decision": "TOTAL", "boundary": total_boundary}])
    return pd.concat([per_decision, total_row], ignore_index=True)


def accuracy_of_classification(df: pd.DataFrame, decision_col: str) -> float:
    lu = lu_per_decision(df, decision_col)
    sum_lower = lu["L"].sum()
    sum_upper = lu["U"].sum()

    if sum_upper == 0:
        return 1.0

    return float(sum_lower / sum_upper)


def quality_of_classification(df: pd.DataFrame, decision_col: str) -> float:
    lu = lu_per_decision(df, decision_col)
    total_objects = len(df)

    if total_objects == 0:
        return 1.0

    sum_lower = lu["L"].sum()

    return float(sum_lower / total_objects)
