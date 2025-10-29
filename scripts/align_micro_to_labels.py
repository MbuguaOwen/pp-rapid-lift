import argparse, os, yaml, logging, numpy as np, pandas as pd
from tqdm import tqdm
from src.eval.cv import time_series_folds, approx_uniqueness_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("align_micro_to_labels")

def parse_args():
    p = argparse.ArgumentParser("Stage 4c — Align micro windows to macro labels (no lookahead) + CV")
    p.add_argument("--labels-config", required=True)
    p.add_argument("--micro-config", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    Lcfg = yaml.safe_load(open(args.labels_config,"r"))
    Mcfg = yaml.safe_load(open(args.micro_config,"r"))

    labels_root = Lcfg["io"]["out_dir"]
    windows_root = Mcfg["io"]["out_root"]
    bar_type = Mcfg["io"]["bar_type"]
    join_H = int(Lcfg["align"]["join_horizon"])
    join_mode = Lcfg["align"]["join_mode"]
    regime_mode = Lcfg["align"]["regime_filter"]["mode"]
    min_votes = int(Lcfg["align"]["regime_filter"]["min_votes"])
    n_splits = int(Lcfg["cv"]["n_splits"])
    embargo_min = int(Lcfg["cv"]["embargo_minutes"])

    out_dir = os.path.join("data","aligned")
    all_rows = []

    for sym in args.symbols:
        for ym in tqdm(args.months, desc="align", unit="month"):
            lab_path = os.path.join(labels_root, sym, f"{ym}.parquet")
            win_dir  = os.path.join(windows_root, sym, bar_type, ym)
            idx_path = os.path.join(win_dir, "index.parquet")
            win_path = os.path.join(win_dir, "windows.npz")
            if not (os.path.exists(lab_path) and os.path.exists(idx_path) and os.path.exists(win_path)):
                LOG.warning("Missing inputs for %s %s", sym, ym); continue

            L = pd.read_parquet(lab_path).sort_index()
            idx = pd.read_parquet(idx_path)

            key = pd.to_datetime(idx["t_last"], utc=True)
            key_floor = key.dt.floor("1min") if join_mode == "floor_minute" else key
            idx["_join_key"] = key_floor

            y_col = f"y_H{join_H}"
            use_cols = [c for c in L.columns if c.startswith("y_")] + ["regime","vote_ratio","bull_votes","bear_votes"]
            Lj = L[use_cols].copy(); Lj["_join_key"] = Lj.index
            D = idx.merge(Lj, on="_join_key", how="left").drop(columns=["_join_key"])

            if regime_mode == "filter":
                D = D.loc[(D["regime"].abs() > 0) & (D[["bull_votes","bear_votes"]].max(axis=1) >= min_votes)]

            D = D.dropna(subset=[y_col])
            D[y_col] = D[y_col].astype(np.int8)

            F = time_series_folds(D["t_last"], n_splits=n_splits, embargo=embargo_min)
            D = D.join(F)

            D["w"] = approx_uniqueness_weights(D["t_last"], join_H)

            os.makedirs(os.path.join(out_dir, sym, bar_type, ym), exist_ok=True)
            D.to_parquet(os.path.join(out_dir, sym, bar_type, ym, "aligned.parquet"))
            all_rows.append(D[["row_id","t_first","t_last",y_col,"regime","vote_ratio","fold","is_train","is_valid","w"]])

    if all_rows:
        ALL = pd.concat(all_rows, ignore_index=True)
        ALL.to_parquet(os.path.join(out_dir, f"aligned_{bar_type}.parquet"))

if __name__ == "__main__":
    main()

