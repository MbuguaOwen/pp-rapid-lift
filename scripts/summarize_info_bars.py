import argparse, os, glob, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/info_bars", help="Root of info bars")
p.add_argument("--symbol", required=True)
args = p.parse_args()

rows = []
for bt in ["dollar","volume","tick","imbalance"]:
    ptn = os.path.join(args.root, args.symbol, bt, "*.parquet")
    for f in sorted(glob.glob(ptn)):
        ym = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_parquet(f)
            rows.append({"month": ym, "bar_type": bt, "rows": len(df)})
        except Exception as e:
            rows.append({"month": ym, "bar_type": bt, "rows": -1})
df = pd.DataFrame(rows).sort_values(["month","bar_type"])
print(df.to_string(index=False))

