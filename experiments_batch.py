"""
Offline script (not part of the web app): uses business.legacy_runner.
Batch experiment: introduce inconsistency at multiple gamma levels,
apply all resolution strategies, build CART trees, collect metrics.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from roughapp.business.algorithms.introduction.introduce_until_quality import run as introduce_until_quality
from roughapp.business.legacy_runner import run_experiment
from roughapp.business.algorithms import cart as tree_builder

from roughapp.core.csv_table import read_csv_table


DATASETS = {
    "breast-cancer":    "data/modified_breast-cancer.csv",
    "hayes-roth-data":  "data/modified_hayes-roth-data.csv",
    "cars":             "data/modified_cars.csv",
    "lymphography":     "data/modified_lymphography.csv",
    "zoo-data":         "data/modified_zoo-data.csv",
    "tic-tac-toe":      "data/modified_tic-tac-toe.csv",
}

GAMMA_TARGETS = [0.9, 0.7, 0.5]
STRATEGIES    = ["mvd", "mcd", "qualitative-removal", "gd"]
RANDOM_STATE  = 42

rows = []

for ds_name, ds_path in DATASETS.items():
    print(f"\n=== {ds_name} ===")
    df_orig, column_types = read_csv_table(ds_path)
    decision_col = df_orig.columns[-1]
    print(f"    {len(df_orig)} rows, {len(df_orig.columns)} cols, decision='{decision_col}'")

    for gamma_target in GAMMA_TARGETS:
        print(f"  gamma_target={gamma_target}", end="  ", flush=True)

        try:
            df_dirty, gamma_actual = introduce_until_quality(
                df_orig,
                decision_col=decision_col,
                target_gamma=gamma_target,
                random_state=RANDOM_STATE,
            )
        except Exception as e:
            print(f"introduce failed: {e}")
            continue

        print(f"gamma_actual={gamma_actual:.3f}  rows={len(df_dirty)}")

        # tree on dirty (before any resolution)
        try:
            before_tree = tree_builder.build(df_dirty, decision_col, column_types=column_types)
        except Exception as e:
            before_tree = None
            print(f"    tree(before) failed: {e}")

        save_dir = Path(f"data/batch_results/{ds_name}/{gamma_target}")
        results = run_experiment(
            df_dirty,
            decision_col=decision_col,
            strategies=STRATEGIES,
            random_state=RANDOM_STATE,
            save_dir=save_dir,
            column_types=column_types,
        )

        for r in results:
            strategy = r["strategy"]
            after_df_path = r.get("result_path")

            # build tree on resolved table
            if after_df_path and Path(after_df_path).exists():
                after_df, after_types = read_csv_table(after_df_path)
                try:
                    after_tree = tree_builder.build(after_df, decision_col, column_types=after_types)
                except Exception:
                    after_tree = None
            else:
                after_tree = None

            bt = before_tree["stats"] if before_tree else {}
            at = after_tree["stats"] if after_tree else {}

            rows.append({
                "dataset":           ds_name,
                "gamma_target":      gamma_target,
                "gamma_actual":      round(gamma_actual, 4),
                "strategy":          strategy,
                "rows_before":       r.get("rows_before"),
                "rows_after":        r.get("rows_after"),
                "rows_removed":      r.get("rows_removed"),
                "gamma_after":       round(r["gamma_after"], 4) if r.get("gamma_after") is not None else None,
                "depth_before":      bt.get("depth"),
                "avg_depth_before":  bt.get("avg_depth"),
                "nodes_before":      (bt.get("internal_nodes", 0) + bt.get("leaves", 0)) or None,
                "depth_after":       at.get("depth"),
                "avg_depth_after":   at.get("avg_depth"),
                "nodes_after":       (at.get("internal_nodes", 0) + at.get("leaves", 0)) or None,
                "internal_nodes_after": at.get("internal_nodes"),
                "leaves_after":      at.get("leaves"),
            })
            print(f"    {strategy:<22} depth {at.get('depth','?'):>3}  avg {at.get('avg_depth','?'):>6}  nodes {(at.get('internal_nodes',0)+at.get('leaves',0)) or '?':>5}")

df_results = pd.DataFrame(rows)
out_path = "data/experiment_results_batch.csv"
df_results.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(df_results.to_string(index=False))
