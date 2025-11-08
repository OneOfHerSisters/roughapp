import pandas as pd

from roughapp.core.grouping import ind_classes_table


def lu_per_decision(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    # Compute lower (L) and upper (U) approximations for each decision
    counts = ind_classes_table(df, decision_col)
    class_size = counts.sum(axis=1)

    rows = []
    for d in counts.columns:
        d_count = counts[d]
        
        # Lower: classes that have ONLY this decision
        L = int(class_size[(d_count == class_size) & (class_size > 0)].sum())
        
        # Upper: classes that have at least one object with this decision
        U = int(class_size[(d_count > 0)].sum())
        
        rows.append({"decision": d, "L": L, "U": U, "boundary": U - L})

    return pd.DataFrame(rows).sort_values("decision").reset_index(drop=True)


def alpha_rho_per_decision(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    # Calculate alpha (accuracy) and rho (roughness) for each decision
    # alpha = L / U, rho = 1 - alpha
    lu = lu_per_decision(df, decision_col)
    lu["alpha"] = lu.apply(lambda r: (r["L"] / r["U"]) if r["U"] > 0 else 1.0, axis=1)
    lu["rho"] = 1.0 - lu["alpha"]
    return lu[["decision", "alpha", "rho", "L", "U", "boundary"]]


def boundary_summary(df: pd.DataFrame, decision_col: str) -> pd.DataFrame:
    # Get boundary region sizes per decision and total
    per = lu_per_decision(df, decision_col)[["decision", "boundary"]]
    total = pd.DataFrame([{"decision": "TOTAL", "boundary": int(per["boundary"].sum())}])
    return pd.concat([per, total], ignore_index=True)