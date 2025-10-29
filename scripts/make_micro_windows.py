import argparse, os, logging, yaml, pandas as pd
from tqdm import tqdm
from src.datasets.micro_windows import build_windows_from_info, save_windows_npz
from src.features.micro_features import add_micro_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("make_micro_windows")

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 — Build micro feature windows from information bars.")
    p.add_argument("--config", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    bar_type = cfg["io"]["bar_type"]; info_root = cfg["io"]["info_root"]; out_root = cfg["io"]["out_root"]
    skip_existing = cfg["io"].get("skip_existing", True) and not args.force

    L = int(cfg["build"]["window_len"]); S = int(cfg["build"]["stride"])
    dtype = str(cfg["build"]["dtype"]); features = cfg["build"]["features"]
    ewm_spans = cfg["build"].get("ewm_spans", {}); norm_method = cfg["norm"]["method"]

    for sym in args.symbols:
        for ym in tqdm(args.months, desc=f"Micro windows {bar_type}", unit="month"):
            in_path = os.path.join(info_root, sym, bar_type, f"{ym}.parquet")
            out_dir = os.path.join(out_root, sym, bar_type, ym)
            win_path = os.path.join(out_dir, "windows.npz")
            if skip_existing and os.path.exists(win_path):
                LOG.info("Exists, skipping: %s", win_path); continue
            if not os.path.exists(in_path):
                LOG.warning("Missing info bars: %s", in_path); continue

            LOG.info("Loading %s", in_path)
            bars = pd.read_parquet(in_path).sort_index()

            LOG.info("Adding micro features")
            feats = add_micro_features(bars, ewm_spans)
            missing = [c for c in features if c not in feats.columns]
            if missing: raise RuntimeError(f"Missing features {missing}")
            feats = feats[features]; feats.index = bars.index

            LOG.info("Windowing (L=%d, S=%d)", L, S)
            X, idx = build_windows_from_info(feats, window_len=L, stride=S, norm_method=norm_method, dtype=dtype)
            if len(idx) == 0:
                LOG.warning("No windows for %s %s", sym, ym); continue

            LOG.info("Saving windows → %s", out_dir)
            save_windows_npz(out_dir, X, idx)

if __name__ == "__main__":
    main()

