from __future__ import annotations
import argparse
from utils import load_csv, dataset_stats, find_inconsistencies, introduce_inconsistency

def cmd_info(args):
    df = load_csv(args.csv)
    stats = dataset_stats(df)
    print(f"Plik: {args.csv}")
    print(f"Kolumny: {stats['cols']}, Wiersze: {stats['rows']}")
    if args.decision:
        inc = find_inconsistencies(df, args.decision)
        print(f"Niespójność: klasy={inc['inconsistent_class_count']}, wiersze={inc['inconsistent_total_rows']}")
        if inc["inconsistent_class_count"] > 0 and args.show_examples:
            print("\nPrzykłady niespójnych klas (max 5):")
            for i, ex in enumerate(inc["examples"], 1):
                print(f"\n--- Klasa {i} ---")
                print(ex.to_string(index=False))

def cmd_introduce(args):
    df = load_csv(args.csv)
    df2 = introduce_inconsistency(df, args.decision, k=args.k, new_value=args.new_value)
    out = args.out or "data/with_inconsistency.csv"
    df2.to_csv(out, index=False)
    print(f"🧪 Wprowadzono niespójność → zapisano: {out}")

def make_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="roughapp",
        description="Narzędzie do analizy tablic decyzyjnych (etap 1)"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # --- info ---
    p_info = sub.add_parser("info", help="Charakterystyka i niespójność")
    p_info.add_argument("csv", help="Ścieżka do pliku CSV")
    p_info.add_argument("--decision", required=True, help="Nazwa kolumny decyzji (D)")
    p_info.add_argument("--show-examples", action="store_true", help="Pokaż przykłady niespójności")
    p_info.set_defaults(func=cmd_info)

    # --- introduce ---
    p_in = sub.add_parser("introduce", help="Wprowadź niespójność (symulacja)")
    p_in.add_argument("csv", help="Ścieżka do pliku CSV")
    p_in.add_argument("--decision", required=True, help="Kolumna decyzji (D)")
    p_in.add_argument("-k", type=int, default=1, help="Ile konfliktów dodać (domyślnie 1)")
    p_in.add_argument("--new-value", help="Wymuś konkretną wartość decyzji")
    p_in.add_argument("--out", help="Plik wyjściowy CSV (domyślnie with_inconsistency.csv)")
    p_in.set_defaults(func=cmd_introduce)

    return p

def main():
    args = make_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()