import argparse
from utils import load_csv, dataset_stats, find_inconsistencies, introduce_inconsistency


def cmd_info(args):
    df = load_csv(args.csv)
    stats = dataset_stats(df)
    print(f"File: {args.csv}")
    print(f"Columns: {stats['cols']}, Rows: {stats['rows']}")

    if args.decision:
        inc = find_inconsistencies(df, args.decision)
        print(
            "Inconsistency: "
            f"classes={inc['inconsistent_class_count']}, "
            f"rows={inc['inconsistent_total_rows']}"
        )
        if inc["inconsistent_class_count"] > 0 and args.show_examples:
            print("\nExamples of inconsistent classes (up to 5):")
            for i, ex in enumerate(inc["examples"], 1):
                print(f"\n--- Class {i} ---")
                print(ex.to_string(index=False))


def cmd_introduce(args):
    df = load_csv(args.csv)
    df2 = introduce_inconsistency(
        df, args.decision, k=args.k, new_value=args.new_value
    )
    out = args.out or "data/with_inconsistency.csv"
    df2.to_csv(out, index=False)
    print(f"Inconsistency introduced → saved to: {out}")


def make_parser():
    p = argparse.ArgumentParser(
        prog="roughapp",
        description="Decision table analysis tool",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Dataset summary and inconsistency report")
    p_info.add_argument("csv", help="Path to the CSV file")
    p_info.add_argument("--decision", required=True, help="Decision column name (D)")
    p_info.add_argument(
        "--show-examples", action="store_true", help="Show examples of inconsistencies"
    )
    p_info.set_defaults(func=cmd_info)

    p_in = sub.add_parser("introduce", help="Introduce inconsistency (simulation)")
    p_in.add_argument("csv", help="Path to the CSV file")
    p_in.add_argument("--decision", required=True, help="Decision column (D)")
    p_in.add_argument("-k", type=int, default=1, help="How many conflicts to add (default: 1)")
    p_in.add_argument("--new-value", help="Force a specific decision value")
    p_in.add_argument("--out", help="Output CSV path",)
    p_in.set_defaults(func=cmd_introduce)

    return p


def main():
    args = make_parser().parse_args()
    # Dispatch to the selected subcommand handler:
    args.func(args)


if __name__ == "__main__":
    main()