from __future__ import annotations
import os, numpy as np, pandas as pd
from typing import Tuple

FEATURE_ORDER = ["logret_close","hl_range_norm","oc_spread_norm","vol_per_trade","bm_share","ewm_logret_16","ewm_abs_logret_16"]

def build_windows_from_info(
    bars: pd.DataFrame,
    window_len: int,
    stride: int,
    norm_method: str = "zscore",
    dtype: str = "float32",
) -> Tuple[np.ndarray, pd.DataFrame]:
    feats = bars[FEATURE_ORDER].copy()
    arr = feats.to_numpy(dtype=np.float64)
    L = int(window_len); S = int(stride); C = arr.shape[1]
    N = max(0, (len(arr) - L) // S + 1)
    X = np.empty((N, L, C), dtype=np.float32 if dtype=="float32" else np.float64)
    t_first = []; t_last = []

    for i in range(N):
        start = i * S
        sl = arr[start:start+L]
        if norm_method == "zscore":
            mu = np.nanmean(sl, axis=0); sd = np.nanstd(sl, axis=0)
            sl = (sl - mu) / np.where(sd > 1e-8, sd, 1.0)
        elif norm_method == "robust":
            med = np.nanmedian(sl, axis=0)
            mad = np.nanmedian(np.abs(sl - med), axis=0)
            sl = (sl - med) / np.where(mad > 1e-8, 1.4826 * mad, 1.0)

        sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
        X[i] = sl.astype(X.dtype, copy=False)

        t_first.append(bars.index[start])
        t_last.append(bars.index[start+L-1])

    idx = pd.DataFrame({"row_id": np.arange(N, dtype=np.int64),
                        "t_first": pd.to_datetime(t_first, utc=True),
                        "t_last": pd.to_datetime(t_last, utc=True)})
    return X, idx

def save_windows_npz(out_dir: str, X, idx: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "windows.npz"), X=X)
    idx.to_parquet(os.path.join(out_dir, "index.parquet"))

