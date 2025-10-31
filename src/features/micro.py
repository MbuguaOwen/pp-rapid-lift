import numpy as np
import pandas as pd
from typing import Dict, List

def build_micro_features(bars: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    # Mirror of the implementation in scripts/make_micro_windows.py
    bars = bars.copy().sort_values("t_last").reset_index(drop=True)
    dtype = cfg.get("build", {}).get("dtype", "float32")

    o = pd.to_numeric(bars.get("open"), errors="coerce")
    h = pd.to_numeric(bars.get("high"), errors="coerce")
    l = pd.to_numeric(bars.get("low"), errors="coerce")
    c = pd.to_numeric(bars.get("close"), errors="coerce")
    v = pd.to_numeric(bars.get("volume"), errors="coerce")
    n = pd.to_numeric(bars.get("n_trades"), errors="coerce")
    bm = pd.to_numeric(bars.get("buy_maker_vol"), errors="coerce")

    lr = np.log(c / c.shift(1))
    feat = pd.DataFrame(index=bars.index)
    feat["logret_close"]   = lr
    feat["hl_range_norm"]  = (h - l) / (c.abs() + 1e-9)
    feat["oc_spread_norm"] = (c - o) / (c.abs() + 1e-9)
    feat["vol_per_trade"]  = (v / n.replace(0, np.nan)).fillna(0.0)
    feat["bm_share"]       = (bm / v.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.5) if "buy_maker_vol" in bars.columns else 0.5

    span = int(cfg.get("build", {}).get("ewm_spans", {}).get("logret", 16))
    feat[f"ewm_logret_{span}"]     = lr.ewm(span=span, adjust=False).mean()
    feat[f"ewm_abs_logret_{span}"] = lr.abs().ewm(span=span, adjust=False).mean()

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in feat.columns:
        feat[col] = np.clip(feat[col].astype(dtype), -8.0, 8.0)
    return feat
