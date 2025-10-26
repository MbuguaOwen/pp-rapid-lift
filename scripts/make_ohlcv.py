from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm

def load_cfg(p="configs/ingestion.yaml"):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _normalize_label(label: str) -> str:
    # accept old aliases like "1T" or "T"
    if label.lower() in ("t", "1t"):
        return "1min"
    return label

def _buy_maker_mask(df: pd.DataFrame) -> pd.Series:
    if "buy_maker" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["buy_maker"]
    if str(s.dtype) in ("bool", "boolean"):
        return s.fillna(False)
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() > 0.8:
        return sn.fillna(0).astype("int8").ne(0)
    sm = s.astype(str).str.lower().map(
        {"true": True, "t": True, "1": True, "false": False, "f": False, "0": False}
    )
    return sm.fillna(False)

def make_ohlcv(df: pd.DataFrame, label: str = "1min", add_buy: bool = True) -> pd.DataFrame:
    label = _normalize_label(label)

    # Use GroupBy with Grouper to avoid expensive inferred_freq computation
    g = pd.Grouper(freq=label, closed="left", label="left")

    # OHLC and volume/trades
    px = df.groupby(g)["price"].agg(open="first", high="max", low="min", close="last")
    vol = df.groupby(g)["qty"].sum().rename("volume")
    ntr = df.groupby(g)["price"].count().astype("int32").rename("n_trades")

    res = px.join([vol, ntr])

    if add_buy:
        mask = _buy_maker_mask(df).astype(bool)
        qty_buy = df["qty"].where(mask, other=0.0)
        buy_vol = qty_buy.groupby(g).sum().fillna(0.0).rename("buy_maker_vol")
        res = res.join(buy_vol)

    # Drop rows that are entirely empty (no ticks in that minute)
    res = res.dropna(how="all")

    # Invariant check ONLY on rows where all 4 OHLC fields exist
    ohlc_complete = res[["open","high","low","close"]].dropna(how="any")
    bad_low  = ohlc_complete["low"]  > ohlc_complete[["open","close","high"]].min(axis=1)
    bad_high = ohlc_complete["high"] < ohlc_complete[["open","close","low"]].max(axis=1)
    bad_idx = bad_low[bad_low].index.union(bad_high[bad_high].index)
    bad = res.loc[bad_idx, ["open","high","low","close","volume","n_trades"]]

    # Log offenders instead of raising; Stage 1 should complete deterministically
    if not bad.empty:
        outlog = Path("reports/qa") / "ohlcv_invariant_issues.csv"
        outlog.parent.mkdir(parents=True, exist_ok=True)
        # Append-friendly: include symbol/month later in main()
        bad.assign(_idx=bad.index.astype(str)).to_csv(outlog, mode="a", header=not outlog.exists(), index=False)

    return res

def main():
    cfg = load_cfg()
    out_root = Path(cfg["paths"]["ohlcv_1m_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = [(s, m) for s in cfg["symbols"] for m in cfg["months"]]
    pbar = tqdm(tasks, desc="Building 1m OHLCV", unit="file")

    for sym, month in pbar:
        pbar.set_postfix(symbol=sym, month=month)
        p = Path(cfg["paths"]["clean_dir"]) / sym / f"{month}.parquet"
        if not p.exists():
            tqdm.write(f"[SKIP] missing clean parquet: {p}")
            continue

        try:
            df = pd.read_parquet(p)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Clean parquet missing DatetimeIndex")
            df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC"))

            res = make_ohlcv(df, cfg["ohlcv"]["label"], cfg["ohlcv"]["add_buy_maker_vol"])

            out = out_root / sym / f"{month}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.to_parquet(out, compression=cfg["io"]["parquet_compression"])
            tqdm.write(f"[OK] {sym} {month}: {len(res):,} bars")
        except Exception as e:
            tqdm.write(f"[ERROR] {sym} {month} -> {type(e).__name__}: {e}")

    tqdm.write("[DONE] OHLCV build complete.")

if __name__ == "__main__":
    main()
