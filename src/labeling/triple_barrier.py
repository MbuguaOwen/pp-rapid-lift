from __future__ import annotations
import numpy as np, pandas as pd

# === Volatility proxies and barrier builders ===

def ewm_vol_sigma(logret: pd.Series, span: int) -> pd.Series:
    """EWM standard deviation of log-returns (dimensionless).
    Uses bfill + fillna(0.0) to avoid deprecated fillna(method=...)."""
    return logret.ewm(span=span, adjust=False).std().bfill().fillna(0.0)

def _ewm_std_of_logret(close: pd.Series, span: int = 120) -> pd.Series:
    """Dimensionless per-bar volatility proxy via EWM std of log-returns."""
    logret = np.log(close).diff()
    vol = logret.ewm(span=span, adjust=False).std()
    vol = vol.bfill().fillna(0.0)
    return vol

def atr(df: pd.DataFrame, window: int = 200, ma: str = "ema") -> pd.Series:
    """
    Average True Range in PRICE units.
    df must have columns: open, high, low, close. UTC DatetimeIndex.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    if str(ma).lower() == "ema":
        out = tr.ewm(span=int(window), adjust=False).mean()
    else:  # "sma" or any other → simple
        out = tr.rolling(int(window), min_periods=max(1, int(window) // 5)).mean()

    return out.bfill().fillna(0.0)

def _barriers_from_returns(close: pd.Series, r_up: pd.Series, r_dn: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert return barriers to price space using exp(r)."""
    up = close * np.exp(r_up)
    dn = close * np.exp(-r_dn)
    return up, dn

def barriers_atr(
    close: pd.Series,
    atr_1m: pd.Series,
    H: int,
    k_pt: float,
    k_sl: float,
    scale_sqrt: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """ATR-based barriers in PRICE units on 1-min bars."""
    scale = np.sqrt(H) if scale_sqrt else 1.0
    up = close + (k_pt * atr_1m * scale)
    dn = close - (k_sl * atr_1m * scale)
    return up, dn

def make_barriers(
    df: pd.DataFrame,
    horizons_min: list[int],
    vol_mode: str,
    scale_with_sqrt_h: bool,
    k_pt_map,
    k_sl_map,
    vol_kwargs: dict | None = None,
) -> dict[int, dict[str, pd.Series]]:
    """
    Return a dict: {H: {"up": Series, "dn": Series}} using either ATR or EWM-std.
    k_pt_map / k_sl_map can be dict keyed by H or scalar; resolve per-H.
    """
    vol_kwargs = vol_kwargs or {}
    close = df["close"].astype(float)

    out: dict[int, dict[str, pd.Series]] = {}
    mode = str(vol_mode).lower()
    if mode == "atr":
        atr_win = int(vol_kwargs.get("atr_window", 200))
        atr_ma = str(vol_kwargs.get("atr_ma", "ema")).lower()
        vol = atr(df, window=atr_win, ma=atr_ma)  # price units

        for H in horizons_min:
            if isinstance(k_pt_map, dict):
                kpt = k_pt_map.get(H, None)
            else:
                kpt = k_pt_map
            if isinstance(k_sl_map, dict):
                ksl = k_sl_map.get(H, None)
            else:
                ksl = k_sl_map
            if kpt is None or ksl is None:
                raise ValueError(f"Missing k_pt/k_sl for horizon {H}")
            up, dn = barriers_atr(close, vol, H, float(kpt), float(ksl), scale_with_sqrt_h)
            out[H] = {"up": up, "dn": dn}

    else:
        # Fallback: EWM std of log returns (dimensionless). Convert to PRICE using close.
        span = int(vol_kwargs.get("vol_span", 120))
        sigma = _ewm_std_of_logret(close, span=span)  # unit: returns
        for H in horizons_min:
            if isinstance(k_pt_map, dict):
                kpt = k_pt_map.get(H, None)
            else:
                kpt = k_pt_map
            if isinstance(k_sl_map, dict):
                ksl = k_sl_map.get(H, None)
            else:
                ksl = k_sl_map
            if kpt is None or ksl is None:
                raise ValueError(f"Missing k_pt/k_sl for horizon {H}")
            scale = np.sqrt(H) if scale_with_sqrt_h else 1.0
            r_up = float(kpt) * sigma * scale
            r_dn = float(ksl) * sigma * scale
            up, dn = _barriers_from_returns(close, r_up, r_dn)
            out[H] = {"up": up, "dn": dn}

    return out

def tb_from_barriers(high: pd.Series, low: pd.Series, barriers: dict[int, dict[str, pd.Series]]) -> pd.DataFrame:
    """Resolve first-touch labels for each horizon against precomputed barriers.

    Returns columns per horizon: y_H*, hit_H*, tth_H* (tth is set to H as before).
    """
    idx = high.index
    out = pd.DataFrame(index=idx)
    for H, bd in barriers.items():
        up = bd["up"]
        dn = bd["dn"]
        hi = high.rolling(window=H, min_periods=1).max().shift(-H)
        lo = low.rolling(window=H, min_periods=1).min().shift(-H)

        up_hit = (hi >= up)
        dn_hit = (lo <= dn)
        y = np.where(up_hit & ~dn_hit, 1, np.where(dn_hit & ~up_hit, -1, 0)).astype(np.int8)

        out[f"y_H{H}"] = y
        out[f"hit_H{H}"] = np.where(y == 1, "up", np.where(y == -1, "dn", "none"))
        out[f"tth_H{H}"] = H
    return out

def tb_labels(log_close: pd.Series, high: pd.Series, low: pd.Series,
              horizons: list[int], up_mult: float, dn_mult: float,
              sigma: pd.Series, min_return: float=0.0) -> pd.DataFrame:
    """
    Approx TB: +1 if upper exceeded within horizon, -1 if lower, 0 otherwise.
    Uses rolling extremes with conservative tie handling.
    Backwards-compatible with prior implementation.
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
