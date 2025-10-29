from __future__ import annotations
import numpy as np, pandas as pd

def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def slopes(close: pd.Series, spans: list[int]) -> pd.DataFrame:
    out = {}
    for s in spans:
        e = ema(close, s)
        out[f"slope_{s}"] = (e - e.shift(1))
    return pd.DataFrame(out, index=close.index)

def vote_regime(close: pd.Series, spans: list[int], slope_thr: float, k_of_n: int) -> pd.DataFrame:
    sl = slopes(close, spans)
    votes = (sl > slope_thr).astype(int) - (sl < -slope_thr).astype(int)
    bull_votes = (votes == 1).sum(axis=1)
    bear_votes = (votes == -1).sum(axis=1)
    n = len(spans)
    regime = np.where(bull_votes >= k_of_n, 1, np.where(bear_votes >= k_of_n, -1, 0)).astype(np.int8)
    vote_ratio = (np.maximum(bull_votes, bear_votes) / n).astype(float)
    out = sl.copy()
    out["regime"] = regime
    out["vote_ratio"] = vote_ratio
    out["bull_votes"] = bull_votes
    out["bear_votes"] = bear_votes
    return out

