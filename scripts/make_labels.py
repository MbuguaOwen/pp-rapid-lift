import argparse, os, yaml, logging, pandas as pd, numpy as np
from tqdm import tqdm
from src.labeling.triple_barrier import ewm_vol_sigma, tb_labels, scheme_map
from src.labeling.regime import vote_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("make_labels")

def parse_args():
    p = argparse.ArgumentParser("Stage 4b — Triple-barrier labels + Regime vote on 1-min bars")
    p.add_argument("--config", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config,"r"))
    root = cfg["io"]["timebar_root"]; out_root = cfg["io"]["out_dir"]
    skip = cfg["io"].get("skip_existing", True) and not args.force

    horizons = cfg["triple_barrier"]["horizons"]
    up_mult = float(cfg["triple_barrier"]["up_mult"]); dn_mult = float(cfg["triple_barrier"]["dn_mult"])
    vol_span = int(cfg["triple_barrier"]["vol_span"]); min_return = float(cfg["triple_barrier"].get("min_return", 0.0))
    scheme = cfg["triple_barrier"].get("label_scheme","onevsrest")

    spans = cfg["regime"]["ema_spans"]; slope_thr = float(cfg["regime"]["slope_thr"])
    k_of_n = int(cfg["regime"]["vote_k_of_n"])

    for sym in args.symbols:
        for ym in tqdm(args.months, desc="labels 1min", unit="month"):
            in_path = os.path.join(root, sym, f"{ym}.parquet")
            out_path = os.path.join(out_root, sym, f"{ym}.parquet")
            if skip and os.path.exists(out_path):
                LOG.info("Exists, skipping %s", out_path); continue
            if not os.path.exists(in_path):
                LOG.warning("Missing time bars: %s", in_path); continue

            bars = pd.read_parquet(in_path).sort_index()
            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low  = bars["low"].astype(float)
            log_close = np.log(close + 1e-12)

            sigma = ewm_vol_sigma(log_close.diff().fillna(0.0), vol_span)
            tb = tb_labels(log_close, high, low, horizons, up_mult, dn_mult, sigma, min_return)
            tb = scheme_map(tb, scheme)

            reg = vote_regime(close, spans, slope_thr, k_of_n)
            L = pd.concat([tb, reg], axis=1); L.index.name = "t_last"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            L.to_parquet(out_path)
            LOG.info("Wrote %s rows -> %s", len(L), out_path)

if __name__ == "__main__":
    main()

