from __future__ import annotations
import numpy as np, pandas as pd

def time_series_folds(t: pd.Series, n_splits: int, embargo: int) -> pd.DataFrame:
    idx = t.sort_values().index
    n = len(idx)
    fold_sizes = [n // n_splits + (1 if i < n % n_splits else 0) for i in range(n_splits)]
    cuts = np.cumsum([0] + fold_sizes)
    rows = []
    for k in range(n_splits):
        v_start, v_end = cuts[k], cuts[k+1]
        v_idx = idx[v_start:v_end]
        tmin = max(0, v_start - embargo)
        tmax = min(n, v_end + embargo)
        trn_idx = idx[:tmin].append(idx[tmax:])
        rows.append(pd.DataFrame({"fold": k,
                                  "is_train": idx.isin(trn_idx),
                                  "is_valid": idx.isin(v_idx)}, index=idx))
    F = pd.concat(rows).groupby(level=0).first()
    return F

def approx_uniqueness_weights(events_t: pd.Series, horizon_bars: int) -> pd.Series:
    # Constant proxy (can be replaced with concurrency calc later)
    w = pd.Series(1.0, index=events_t.index)
    return w / w.mean()

