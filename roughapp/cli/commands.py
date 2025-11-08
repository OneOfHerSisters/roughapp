import argparse
import sys
from typing import Any, Dict

import pandas as pd

from roughapp.core.utils import dataset_stats
from roughapp.services.analyzer import find_inconsistencies
from roughapp.core.metrics import alpha_rho_per_decision
from roughapp.services.introduction import duplicate_flip, drop_attributes, drop_random_attribute

INTRODUCERS = {
    "duplicate-flip": duplicate_flip.run,
    "drop-attrs": drop_attributes.run,
    "drop-random": drop_random_attribute.run,
}


def load_csv(path: str) -> pd.DataFrame:
    # Load CSV file
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    
    if df.empty:
        raise ValueError("CSV is empty")
    
    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roughapp",
        description="Tool based on Pawlak's rough sets theory"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Info command
    i = sub.add_parser("info", help="Dataset summary and inconsistency analysis")
    i.add_argument("csv", help="Path to CSV file")
    i.add_argument("--decision", required=True, help="Name of the decision column")
    i.add_argument("--metrics", action="store_true", help="Show Pawlak metrics")

    # Introduce command
    j = sub.add_parser("introduce", help="Introduce inconsistency into dataset")
    j.add_argument("csv", help="Path to input CSV file")
    j.add_argument("--decision", required=True, help="Name of the decision column")
    j.add_argument(
        "--method",
        choices=list(INTRODUCERS.keys()),
        default="duplicate-flip",
        help="Method to introduce inconsistency"
    )
    j.add_argument("-k", type=int, default=1, help="Number of rows to duplicate (for duplicate-flip)")
    j.add_argument("--new-value", help="New decision value to use (for duplicate-flip)")
    j.add_argument("--attrs", nargs="*", help="Attributes to drop (for drop-attrs)")
    j.add_argument("-n", type=int, default=1, help="Number of attributes to drop (for drop-random)")
    j.add_argument("--out", default="data/with_inconsistency.csv", help="Output file path")

    return p


def cmd_info(args: argparse.Namespace) -> None:
    try:
        df = load_csv(args.csv)
        st = dataset_stats(df)
        print(f"File: {args.csv}\nColumns: {st['cols']}, Rows: {st['rows']}")
        
        inc = find_inconsistencies(df, args.decision)
        print(
            f"Inconsistency: classes={inc['inconsistent_class_count']}, "
            f"rows={inc['inconsistent_total_rows']}"
        )
        
        if args.metrics:
            per = alpha_rho_per_decision(df, args.decision)
            print("\nPawlak per decision:")
            for _, r in per.iterrows():
                print(
                    f"  {r['decision']}: alpha={r['alpha']:.4f}, rho={r['rho']:.4f} "
                    f"(L={r['L']}, U={r['U']}, boundary={r['boundary']})"
                )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_introduce(args: argparse.Namespace) -> None:
    try:
        df = load_csv(args.csv)
        method = INTRODUCERS[args.method]
        
        kwargs: Dict[str, Any] = {"decision_col": args.decision}
        if args.method == "duplicate-flip":
            kwargs["k"] = args.k
            if args.new_value:
                kwargs["new_value"] = args.new_value
        elif args.method == "drop-attrs":
            if args.attrs:
                kwargs["attrs"] = args.attrs
        elif args.method == "drop-random":
            kwargs["n"] = args.n
        
        df2 = method(df, **kwargs)
        df2.to_csv(args.out, index=False)
        print(f"Inconsistency introduced via {args.method} → saved: {args.out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)