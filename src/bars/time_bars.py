from __future__ import annotations
import os, glob, logging
from typing import List, Dict
import numpy as np, pandas as pd

LOG = logging.getLogger("time_bars")

def _first_present(d: pd.DataFrame, cands: List[str]) -> str|None:
    cols = {str(c).strip().lower(): c for c in d.columns}
    for c in cands:
        k = str(c).strip().lower()
        if k in cols: return cols[k]
    return None

def _normalize_ticks(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        ts = out.index
    else:
        tcol = _first_present(out, schema["time_col_candidates"])
        if tcol is None: raise ValueError("Missing time column/index")
        ts = out[tcol]
    ts = pd.to_datetime(ts, utc=False, errors="coerce")
    ts = pd.DatetimeIndex(ts)
    if ts.tz is None: ts = ts.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else: ts = ts.tz_convert("UTC")
    out = out.loc[~ts.isna()].copy(); ts = ts[~ts.isna()]
    out.index = ts

    pcol = _first_present(out, schema["price_col_candidates"])
    qcol = _first_present(out, schema["qty_col_candidates"])
    bmcol = _first_present(out, schema["buyer_maker_col_candidates"])
    if pcol is None or qcol is None: raise ValueError("Missing price/qty columns")
    out["price"] = pd.to_numeric(out[pcol], errors="coerce")
    out["qty"] = pd.to_numeric(out[qcol], errors="coerce")
    if bmcol is not None:
        bm = out[bmcol]
        out["is_buyer_maker"] = bm.astype(bool) if bm.dtype.kind in "biu" else bm.astype(str).str.lower().isin(["1","true","t","yes","y"])
    else:
        out["is_buyer_maker"] = False
    out = out[(out["qty"]>0) & np.isfinite(out["price"])]
    return out[["price","qty","is_buyer_maker"]]

def build_time_bars(ticks: pd.DataFrame, freq: str="1min") -> pd.DataFrame:
    g = ticks.groupby(pd.Grouper(freq=freq, label="right", origin="start_day"))
    def ohlc(d: pd.DataFrame):
        if d.empty:
            return pd.Series({"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                              "volume": 0.0, "n_trades": 0, "buy_maker_vol": 0.0, "t_first": pd.NaT, "t_last": pd.NaT})
        return pd.Series({
            "open": d["price"].iloc[0],
            "high": float(d["price"].max()),
            "low": float(d["price"].min()),
            "close": d["price"].iloc[-1],
            "volume": float(d["qty"].sum()),
            "n_trades": int(len(d)),
            "buy_maker_vol": float(d.loc[d["is_buyer_maker"], "qty"].sum()),
            "t_first": d.index[0],
            "t_last": d.index[-1],
        })
    bars = g.apply(ohlc)
    bars.index = pd.DatetimeIndex(bars.index, name="t_last").tz_convert("UTC")
    bars = bars.dropna(subset=["open","close"])
    return bars

def load_ticks(pattern: str, schema: dict) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files: raise FileNotFoundError(pattern)
    parts = []
    for f in files:
        try:
            d = pd.read_parquet(f) if f.lower().endswith(".parquet") else pd.read_csv(f)
            parts.append(d)
        except Exception as e:
            LOG.error("read fail %s: %s", f, e)
    ticks = pd.concat(parts, ignore_index=False)
    return _normalize_ticks(ticks, schema)

def write_parquet(path: str, df: pd.DataFrame, meta: Dict[str,str]|None=None):
    import pyarrow as pa, pyarrow.parquet as pq, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pandas(df)
    md = {} if table.schema.metadata is None else dict(table.schema.metadata)
    for k,v in (meta or {}).items():
        md[str(k).encode()] = str(v).encode()
    table = table.replace_schema_metadata(md)
    tmp = path + ".__tmp__"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)

