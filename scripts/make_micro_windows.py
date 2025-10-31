import os, glob, yaml, math
import numpy as np
import pandas as pd
from typing import Dict, List

# ---------- feature helpers ----------

def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return pd.DataFrame(out)

def build_micro_features(bars: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Build leak-free per-bar micro features from info bars/time bars.
    Expects columns: open, high, low, close, volume, n_trades, buy_maker_vol (optional).
    Returns a DataFrame aligned 1:1 with bars containing:
      - logret_close, hl_range_norm, oc_spread_norm, vol_per_trade, bm_share,
      - ewm_logret_16, ewm_abs_logret_16   (span controlled via cfg.build.ewm_spans.logret)
    """
    dtype = cfg.get("build", {}).get("dtype", "float32")
    bars = bars.copy().sort_values("t_last").reset_index(drop=True)

    N = _num(bars, ["open","high","low","close","volume","n_trades","buy_maker_vol"])
    for c in ["open","high","low","close","volume","n_trades","buy_maker_vol"]:
        if c not in N.columns: N[c] = np.nan
    o,h,l,c,v,n,bm = N["open"],N["high"],N["low"],N["close"],N["volume"],N["n_trades"],N["buy_maker_vol"]

    # Basic (past-only) transforms
    lr = np.log(c / c.shift(1))
    feat = pd.DataFrame(index=bars.index)
    feat["logret_close"]   = lr
    feat["hl_range_norm"]  = (h - l) / (c.abs() + 1e-9)
    feat["oc_spread_norm"] = (c - o) / (c.abs() + 1e-9)
    feat["vol_per_trade"]  = (v / n.replace(0, np.nan)).fillna(0.0)
    if "buy_maker_vol" in bars.columns:
        feat["bm_share"] = (bm / v.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    else:
        feat["bm_share"] = 0.5

    span = int(cfg.get("build", {}).get("ewm_spans", {}).get("logret", 16))
    feat[f"ewm_logret_{span}"]     = lr.ewm(span=span, adjust=False).mean()
    feat[f"ewm_abs_logret_{span}"] = lr.abs().ewm(span=span, adjust=False).mean()

    # Clean + dtype
    feat = feat.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    for col in feat.columns:
        feat[col] = np.clip(feat[col].astype(dtype), -8.0, 8.0)
    return feat

# ---------- IO & driver ----------

def _find_info_bar_files(info_root: str, bar_type: str, symbol: str, months: List[str]) -> List[str]:
    out = []
    for m in months:
        cands = [
            os.path.join(info_root, symbol, bar_type, f"{m}.parquet"),
            os.path.join(info_root, bar_type, symbol, f"{m}.parquet"),
            os.path.join(info_root, symbol, f"{m}_{bar_type}.parquet"),
            os.path.join(info_root, symbol, f"{m}.parquet"),
        ]
        hit = next((p for p in cands if os.path.exists(p)), None)
        if hit: out.append(hit)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Stage 3: Micro features from info bars")
    ap.add_argument("--config", required=True, help="configs/micro_windows.yaml")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--months",  nargs="+", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    io_cfg   = cfg.get("io", {})
    build    = cfg.get("build", {})
    out_root = io_cfg.get("out_dir", "data/micro_windows")
    info_root= io_cfg.get("info_root", "data/info_bars")
    bar_type = io_cfg.get("bar_type", "dollar")
    skip     = bool(io_cfg.get("skip_existing", True))
    dtype    = build.get("dtype", "float32")

    os.makedirs(out_root, exist_ok=True)
    print(f"[cfg] bar_type={bar_type}  info_root={info_root}  out_root={out_root}  skip_existing={skip}")
    print(f"[run] symbols={args.symbols} months={args.months}")

    for sym in args.symbols:
        files = _find_info_bar_files(info_root, bar_type, sym, args.months)
        if not files:
            print(f"[warn] no files for {sym} under {info_root}/{sym}/{bar_type}")
            continue
        out_sym_dir = os.path.join(out_root, sym)
        os.makedirs(out_sym_dir, exist_ok=True)
        for fp in files:
            m = os.path.splitext(os.path.basename(fp))[0]  # e.g., 2025-01
            out_path = os.path.join(out_sym_dir, f"{m}.parquet")
            if skip and os.path.exists(out_path):
                print(f"[skip] {out_path}")
                continue
            try:
                B = pd.read_parquet(fp)
            except Exception as e:
                print(f"[read-fail] {fp} -> {type(e).__name__}: {e}")
                continue

            # choose a time column OR index name
            known = ("t_last","timestamp","time","t0","entry_ts")
            tcol = next((c for c in known if c in B.columns), None)

            # if a DatetimeIndex exists and its name is one of the known time names,
            # make sure it does NOT collide with the column name
            if isinstance(B.index, pd.DatetimeIndex) and (B.index.name in known):
                if tcol is None:
                    # use the index as the time source -> materialize it as a column
                    tcol = B.index.name
                    B = B.reset_index()  # brings index out as a column named tcol
                else:
                    # we already have a time column AND an index with the same name -> drop the index
                    B = B.reset_index(drop=True)

            if tcol is None:
                print(f"[skip] {fp} (no recognizable time column or index)")
                continue

            # now tcol is an unambiguous COLUMN
            B = B.sort_values(tcol).reset_index(drop=True)
            B["t_last"] = _to_utc(B[tcol])
            B["symbol"] = sym

            F = build_micro_features(B, cfg)

            OUT = pd.DataFrame({"symbol": B["symbol"].astype(str), "t_last": B["t_last"]})
            for c in F.columns:
                OUT[c] = F[c].astype(dtype)
            OUT.to_parquet(out_path, index=False)
            print(f"[ok] {sym} {m} -> {out_path}  rows={len(OUT)}  feats={len(F.columns)}")

if __name__ == "__main__":
    main()
