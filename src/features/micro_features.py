from __future__ import annotations
import numpy as np, pandas as pd

def _safe_div(a, b, eps=1e-12):
    return a / np.maximum(b, eps)

def add_micro_features(df: pd.DataFrame, ewm_spans: dict) -> pd.DataFrame:
    """
    Input info bars columns:
      open, high, low, close, volume, n_trades, buy_maker_vol, t_first, t_last
    Returns same index with added feature columns (no inf; NaNs only at head).
    """
    out = df.copy()

    out["logret_close"] = np.log(out["close"]).diff()
    out["hl_range_norm"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_spread_norm"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    out["vol_per_trade"] = _safe_div(out["volume"], out["n_trades"].clip(lower=1))
    out["bm_share"] = _safe_div(out["buy_maker_vol"], out["volume"].abs().replace(0, np.nan))

    span = int(ewm_spans.get("logret", 16))
    lr = out["logret_close"].fillna(0.0)
    out["ewm_logret_16"] = lr.ewm(span=span, adjust=False).mean()
    out["ewm_abs_logret_16"] = lr.abs().ewm(span=span, adjust=False).mean()

    for c in ["logret_close","hl_range_norm","oc_spread_norm","vol_per_trade","bm_share","ewm_logret_16","ewm_abs_logret_16"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    return out

