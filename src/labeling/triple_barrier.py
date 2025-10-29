from __future__ import annotations
import numpy as np, pandas as pd

def ewm_vol_sigma(logret: pd.Series, span: int) -> pd.Series:
    return logret.ewm(span=span, adjust=False).std().fillna(method="bfill").fillna(0.0)

def tb_labels(log_close: pd.Series, high: pd.Series, low: pd.Series,
              horizons: list[int], up_mult: float, dn_mult: float,
              sigma: pd.Series, min_return: float=0.0) -> pd.DataFrame:
    """
    Approx TB: +1 if upper exceeded within horizon, -1 if lower, 0 otherwise.
    Uses rolling extremes with conservative tie handling.
    """
    close = np.exp(log_close)
    up_f = np.exp(up_mult * sigma)
    dn_f = np.exp(-dn_mult * sigma)
    out = pd.DataFrame(index=log_close.index)
    for H in horizons:
        up = close * up_f
        dn = close * dn_f
        hi = high.rolling(window=H, min_periods=1).max().shift(-H)
        lo = low.rolling(window=H, min_periods=1).min().shift(-H)

        up_hit = (hi >= up)
        dn_hit = (lo <= dn)
        y = np.where(up_hit & ~dn_hit, 1,
            np.where(dn_hit & ~up_hit, -1, 0)).astype(np.int8)

        if min_return > 0:
            small = (np.abs(sigma.values) < min_return)
            y[small] = 0

        out[f"y_H{H}"] = y
        out[f"hit_H{H}"] = np.where(y==1,"up", np.where(y==-1,"dn","none"))
        out[f"tth_H{H}"] = H
    return out

def scheme_map(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    if scheme == "onevsrest":
        return df
    elif scheme == "trinary":
        out = df.copy()
        for c in [c for c in df.columns if c.startswith("y_H")]:
            out[c] = df[c].map({-1:0, 0:1, 1:2}).astype(np.int8)
        return out
    else:
        return df

