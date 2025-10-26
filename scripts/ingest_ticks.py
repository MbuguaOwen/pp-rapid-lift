import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


# ---------- Config ----------

def load_cfg(path: str = "configs/ingestion.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- Utilities ----------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_raw_file(sym: str, month: str, cfg) -> Optional[Path]:
    raw_root = Path(cfg["paths"]["raw_dir"])
    tmpl = cfg["paths"].get("filename_template")

    cand_dirs = [raw_root / sym, raw_root]
    cand_names = []
    if tmpl:
        cand_names.append(tmpl.format(symbol=sym, month=month))
    cand_names += [
        f"{sym}-ticks-{month}.csv",
        f"{sym}_{month}.csv",
        f"{month}.csv",
    ]

    for d in cand_dirs:
        for name in cand_names:
            p = d / name
            if p.exists():
                return p
    return None


def _ci_lookup(cols: List[str], wanted: List[str]) -> Optional[str]:
    """case-insensitive column finder"""
    lower = {c.lower(): c for c in cols}
    for w in wanted:
        if w.lower() in lower:
            return lower[w.lower()]
    return None


def autodetect_headers(df: pd.DataFrame, rf: Dict) -> Tuple[str, str, Optional[str], str]:
    """
    Return (timestamp_col, price_col, side_col_opt, qty_col)
    Tries config first, falls back to common aliases.
    """
    aliases = {
        "timestamp": ["ts", "timestamp", "time", "date", "datetime", "event_time", "trade_time"],
        "price": ["price", "p", "last_price", "trade_price"],
        "qty": ["qty", "quantity", "size", "amount", "vol", "volume"],
        "side": ["is_buyer_maker", "buyer_is_maker", "isBuyerMaker", "taker_side", "is_buyer_makerr"]
    }

    cols = list(df.columns)

    tcol = rf.get("timestamp_field")
    if not tcol or tcol not in cols:
        tcol = _ci_lookup(cols, aliases["timestamp"])
    if not tcol:
        raise KeyError("No timestamp-like column found. Available columns: " + ", ".join(cols))

    pcol = rf.get("price_field")
    if not pcol or pcol not in cols:
        pcol = _ci_lookup(cols, aliases["price"])
    if not pcol:
        raise KeyError("No price-like column found. Available columns: " + ", ".join(cols))

    qcol = rf.get("qty_field")
    if not qcol or qcol not in cols:
        qcol = _ci_lookup(cols, aliases["qty"])
    if not qcol:
        raise KeyError("No qty-like column found. Available columns: " + ", ".join(cols))

    scol = rf.get("side_field")
    if not scol or scol not in cols:
        scol = _ci_lookup(cols, aliases["side"])  # optional, may remain None

    return tcol, pcol, scol, qcol


def parse_ts_auto(series: pd.Series, unit_cfg: str) -> pd.DatetimeIndex:
    """
    Parse timestamp series to UTC DatetimeIndex.
    unit_cfg: "s" | "ms" | "iso" | "auto"
    """
    if unit_cfg in ("s", "ms"):
        return pd.to_datetime(series, unit=unit_cfg, utc=True, errors="coerce")
    if unit_cfg == "iso":
        return pd.to_datetime(series, utc=True, errors="coerce")
    # auto
    s = pd.to_numeric(series, errors="coerce")
    numeric_ratio = s.notna().mean()
    if numeric_ratio > 0.8:
        # Heuristic: epoch seconds ~1e9, ms ~1e12
        med = s.median()
        if med > 1e11:   # definitely ms
            return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        else:
            return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    # treat as ISO-like
    return pd.to_datetime(series, utc=True, errors="coerce")


# ---------- Core ----------

def clean_one(csv_path: Path, cfg: dict, out_parquet: Path) -> int:
    rf, cl, io = cfg["raw_format"], cfg["cleaning"], cfg["io"]
    df = pd.read_csv(csv_path)

    # autodetect headers when needed
    if cl.get("autodetect_headers", True):
        tcol, pcol, scol, qcol = autodetect_headers(df, rf)
    else:
        tcol, pcol, scol, qcol = rf["timestamp_field"], rf["price_field"], rf.get("side_field"), rf["qty_field"]

    # parse
    ts = parse_ts_auto(df[tcol], rf.get("timestamp_unit", "auto"))
    price = pd.to_numeric(df[pcol], errors="coerce")
    qty = pd.to_numeric(df[qcol], errors="coerce")
    buy_maker = df[scol] if scol and (scol in df.columns) else pd.Series([None] * len(df), name="buy_maker")

    df = pd.DataFrame({"timestamp": ts, "price": price, "qty": qty, "buy_maker": buy_maker})

    # normalize buy_maker to nullable boolean early
    if "buy_maker" in df.columns:
        bm = df["buy_maker"]
        if str(bm.dtype) not in ("bool", "boolean"):
            num = pd.to_numeric(bm, errors="coerce")
            if num.notna().mean() > 0.8:
                df["buy_maker"] = num.fillna(0).astype("int8").ne(0).astype("boolean")
            else:
                df["buy_maker"] = bm.astype(str).str.lower().map(
                    {"true": True, "t": True, "1": True, "false": False, "f": False, "0": False}
                ).fillna(False).astype("boolean")

    # cleaning
    if cl.get("drop_na", True):
        df = df.dropna(subset=["timestamp", "price", "qty"])

    if cl.get("dedupe", True):
        df = df.drop_duplicates(subset=["timestamp", "price", "qty"])

    if cl.get("sort", True):
        df = df.sort_values("timestamp")

    df = df[(df["price"] > 0) & (df["qty"] >= cl.get("min_qty", 0.0))]

    # mild outlier guard
    clip_cfg = cl.get("clip_price", {})
    if clip_cfg.get("enabled", False) and len(df) > 1200:
        p = df["price"].astype(float)
        med = p.rolling(1000, min_periods=100).median()
        lo = clip_cfg.get("pct_low", 0.0)
        hi = clip_cfg.get("pct_high", 0.0)
        mask = ~(((p < med * (1 - lo)) | (p > med * (1 + hi))) & med.notna())
        df = df[mask]

    # finalize index
    df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("timestamp did not parse to DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # monotonic & unique
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, compression=io["parquet_compression"])

    # hard checks
    assert (df["price"] > 0).all()
    assert (df["qty"] >= 0).all()
    assert df.index.is_monotonic_increasing

    return len(df)


def main():
    cfg = load_cfg()
    qa_dir = Path(cfg["paths"]["qa_dir"]); qa_dir.mkdir(parents=True, exist_ok=True)

    catalog_rows = []
    error_rows = []

    tasks = [(sym, m) for sym in cfg["symbols"] for m in cfg["months"]]
    pbar = tqdm(tasks, desc="Ingesting ticks", unit="file")

    for sym, month in pbar:
        pbar.set_postfix(symbol=sym, month=month)
        raw = find_raw_file(sym, month, cfg)
        if raw is None:
            msg = f"Missing {cfg['paths']['raw_dir']}/* for {sym} {month}"
            tqdm.write("[WARN] " + msg)
            error_rows.append({
                "symbol": sym, "month": month, "file": None, "error": msg,
                "columns": None
            })
            continue

        try:
            out = Path(cfg["paths"]["clean_dir"]) / sym / f"{month}.parquet"
            nrows = clean_one(raw, cfg, out)
            catalog_rows.append({
                "symbol": sym, "month": month,
                "raw": str(raw), "raw_sha256": sha256_file(raw),
                "clean": str(out), "clean_sha256": sha256_file(out),
                "rows": int(nrows)
            })
            tqdm.write(f"[OK] {sym} {month}: {nrows:,} rows")
        except Exception as e:
            try:
                cols = list(pd.read_csv(raw, nrows=1).columns)
            except Exception:
                cols = None
            err_msg = f"{type(e).__name__}: {e}"
            tqdm.write(f"[ERROR] {sym} {month} -> {err_msg}")
            error_rows.append({
                "symbol": sym, "month": month, "file": str(raw),
                "error": err_msg, "columns": ";".join(cols) if cols else None
            })
            continue

    # write artifacts
    if catalog_rows:
        pd.DataFrame(catalog_rows).to_csv(qa_dir / "catalog_ticks.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(qa_dir / "ingest_errors.csv", index=False)

    tqdm.write(f"[DONE] Catalog: {qa_dir / 'catalog_ticks.csv'}")
    if error_rows:
        tqdm.write(f"[DONE] Errors:  {qa_dir / 'ingest_errors.csv'}  (inspect this for missing columns/files)")


if __name__ == "__main__":
    main()
