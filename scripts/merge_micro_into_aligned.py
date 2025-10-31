import os, glob, re
import numpy as np
import pandas as pd

ALIGNED_IN  = r"data/aligned/aligned_dollar_num_mh.parquet"     # has uid/fold + y_H120/240/480 + masks
MICRO_ROOT  = r"data/micro_windows"
ALIGNED_OUT = r"data/aligned/aligned_dollar_num_mh_feat.parquet"

def tmin(s):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.floor("min")

A = pd.read_parquet(ALIGNED_IN)
A["t_min"] = tmin(A["t_last"])
if "symbol" not in A.columns: A["symbol"] = "BTCUSDT"

parts = []
for p in glob.glob(os.path.join(MICRO_ROOT, "**", "*.parquet"), recursive=True):
    try:
        M = pd.read_parquet(p)
    except Exception:
        continue
    if "t_last" not in M.columns: 
        continue
    T = pd.DataFrame({"t_min": tmin(M["t_last"])})
    T["symbol"] = M["symbol"].astype(str) if "symbol" in M.columns else "BTCUSDT"
    EXCL = {"uid","row_id","fold","t_first","entry_ts","t0","t_last","ts","timestamp","time","symbol"}
    EXCL |= {c for c in M.columns if c.startswith("y_") or c.startswith("valid_")}
    num_cols = [c for c in M.select_dtypes(include=["number"]).columns if c not in EXCL]
    if not num_cols: 
        continue
    for c in num_cols:
        T[c] = pd.to_numeric(M[c], errors="coerce")
    parts.append(T)

if not parts:
    raise SystemExit("No numeric micro features found under data/micro_windows/**. Check Stage-3 outputs.")

F = (pd.concat(parts, ignore_index=True)
       .dropna(subset=["t_min"])
       .sort_values(["symbol","t_min"])
       .drop_duplicates(["symbol","t_min"], keep="last"))

DF = A.merge(F, on=["symbol","t_min"], how="left").drop(columns=["t_min"], errors="ignore")

protect = {"uid","fold"} | {c for c in DF.columns if c.startswith("y_") or c.startswith("valid_")}
num_all = DF.select_dtypes(include=["number"]).columns.tolist()
feat_cols = [c for c in num_all if c not in protect and c not in {"row_id"}]

DF[feat_cols] = DF[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).astype("float32")

tr = DF["fold"]==0
mu = DF.loc[tr, feat_cols].mean()
sd = DF.loc[tr, feat_cols].std().replace(0.0, 1.0)
DF[feat_cols] = (DF[feat_cols] - mu) / sd

DF["uid"]  = DF["uid"].astype("int64")
DF["fold"] = DF["fold"].astype("int8")
DF = DF.fillna(0.0)

DF.to_parquet(ALIGNED_OUT, index=False)
pd.DataFrame({"feature": feat_cols, "mu": mu.values, "sd": sd.values}).to_parquet(
    ALIGNED_OUT.replace(".parquet", "_feat_stats.parquet"), index=False
)

print("Wrote:", ALIGNED_OUT)
print("n_features:", len(feat_cols))
print("sample:", feat_cols[:20])
