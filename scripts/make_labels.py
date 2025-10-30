import argparse, os, yaml, logging, pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from src.labeling.triple_barrier import (
    ewm_vol_sigma,
    tb_labels,
    scheme_map,
    make_barriers,
    tb_from_barriers,
)
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

    # Backward-compatible config parsing
    horizons = cfg.get("horizons_min", cfg.get("triple_barrier", {}).get("horizons", []))
    if not horizons:
        raise ValueError("No horizons specified: set 'horizons_min' or 'triple_barrier.horizons'")
    scheme = cfg.get("triple_barrier", {}).get("label_scheme", "onevsrest")
    min_return = float(cfg.get("triple_barrier", {}).get("min_return", 0.0))

    # EWM path params (legacy)
    vol_span = int(cfg.get("vol_span", cfg.get("triple_barrier", {}).get("vol_span", 120)))
    up_mult_legacy = cfg.get("triple_barrier", {}).get("up_mult", None)
    dn_mult_legacy = cfg.get("triple_barrier", {}).get("dn_mult", None)

    # New barrier selection params
    vol_mode = str(cfg.get("vol_mode", "ewm_std")).lower()
    scale_with_sqrt_h = bool(cfg.get("scale_with_sqrt_h", True))
    k_pt_cfg = cfg.get("k_pt", None)
    k_sl_cfg = cfg.get("k_sl", None)
    if k_pt_cfg is None and up_mult_legacy is not None:
        k_pt_cfg = float(up_mult_legacy)
    if k_sl_cfg is None and dn_mult_legacy is not None:
        k_sl_cfg = float(dn_mult_legacy)
    vol_kwargs = {
        "atr_window": cfg.get("atr_window", cfg.get("triple_barrier", {}).get("atr_window", 200)),
        "atr_ma": cfg.get("atr_ma", cfg.get("triple_barrier", {}).get("atr_ma", "ema")),
        "vol_span": vol_span,
    }

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

            bars = pd.read_parquet(in_path)
            # Ensure UTC index
            if isinstance(bars.index, pd.DatetimeIndex):
                if bars.index.tz is None:
                    bars.index = bars.index.tz_localize("UTC")
                else:
                    bars.index = bars.index.tz_convert("UTC")
            bars = bars.sort_index()
            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low  = bars["low"].astype(float)
            log_close = np.log(close + 1e-12)

            if vol_mode == "atr":
                barriers = make_barriers(
                    df=bars,
                    horizons_min=list(map(int, horizons)),
                    vol_mode="atr",
                    scale_with_sqrt_h=scale_with_sqrt_h,
                    k_pt_map=k_pt_cfg if k_pt_cfg is not None else 1.0,
                    k_sl_map=k_sl_cfg if k_sl_cfg is not None else 1.0,
                    vol_kwargs=vol_kwargs,
                )
                tb = tb_from_barriers(high, low, barriers)
            else:
                sigma = ewm_vol_sigma(log_close.diff().fillna(0.0), vol_span)
                up_mult = float(k_pt_cfg if (k_pt_cfg is not None and not isinstance(k_pt_cfg, dict)) else (up_mult_legacy if up_mult_legacy is not None else 1.0))
                dn_mult = float(k_sl_cfg if (k_sl_cfg is not None and not isinstance(k_sl_cfg, dict)) else (dn_mult_legacy if dn_mult_legacy is not None else 1.0))
                tb = tb_labels(log_close, high, low, horizons, up_mult, dn_mult, sigma, min_return)
            tb = scheme_map(tb, scheme)

            reg = vote_regime(close, spans, slope_thr, k_of_n)
            L = pd.concat([tb, reg], axis=1); L.index.name = "t_last"

            # Tiny audit per horizon
            print(f"[audit] {sym} {ym}")
            for H in horizons:
                col = f"y_H{int(H)}"
                if col in L.columns:
                    vc = L[col].value_counts(dropna=False).sort_index()
                    print(f"  {col}: {dict(vc)}")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            L.to_parquet(out_path)
            LOG.info("Wrote %s rows -> %s", len(L), out_path)

if __name__ == "__main__":
    main()
