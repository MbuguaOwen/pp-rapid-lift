import sys, pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/peek_info_bars.py <path-to-parquet>")
        sys.exit(1)
    p = sys.argv[1]
    df = pd.read_parquet(p)
    print(p, "rows:", len(df))
    print(df.head(3))
    print("index tz:", getattr(df.index, "tz", None))
    print("columns:", list(df.columns))

