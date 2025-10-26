from __future__ import annotations
import os, glob, math
import logging
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np

# ---------- Logging ----------
LOG = logging.getLogger("info_bars")

def _to_utc(ts) -> pd.Timestamp:
    """
    Return a UTC tz-aware Timestamp for any timestamp-like input.
    Safe for naive or tz-aware datetimes.
    """
    if ts is None:
        return pd.NaT
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        try:
            return t.tz_localize("UTC")
        except Exception:
            # If already datetime64[ns] without tz and cannot localize, last resort:
            return pd.to_datetime(t, utc=True, errors="coerce")
    else:
        return t.tz_convert("UTC")

# ---------- Schema helpers ----------
def _first_present(d: pd.DataFrame, cands: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in d.columns}
    for c in cands:
        k = str(c).strip().lower()
        if k in cols:
            return cols[k]
    return None

def normalize_ticks(df: pd.DataFrame, cfg_schema: dict) -> pd.DataFrame:
    """
    Normalize ticks to columns: ts (UTC ns), price, qty, [side], [is_buyer_maker].
    Accepts:
      • DatetimeIndex as the timestamp  (preferred in our clean ticks)
      • or any time column from schema['time_col_candidates'].
    """
    log = LOG
    out = df.copy()

    # --- Resolve timestamp ---
    ts = None
    if isinstance(out.index, pd.DatetimeIndex):
        ts = out.index
    else:
        tcol = _first_present(out, cfg_schema.get("time_col_candidates", []))
        if tcol is not None:
            ts = out[tcol]
        elif "index" in out.columns:
            ts = out["index"]  # parquet sometimes stores previous index as a column
        else:
            raise ValueError(f"Missing required time index/column; present cols: {list(out.columns)[:20]}")

    # Parse/force tz to UTC
    if np.issubdtype(getattr(ts, "dtype", object), np.number):
        unit = cfg_schema.get("epoch_unit", "ms")
        if unit not in ("ms","s","ns"):
            unit = "ms"
        ts = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(ts, utc=True, errors="coerce")

    if ts.tz is None:
        # localize naive as UTC (clean ticks are already UTC timestamps)
        ts = ts.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.tz_convert("UTC")

    bad = ts.isna().sum()
    if bad:
        log.warning("Some timestamps failed to parse; dropping %d rows", int(bad))
        out = out.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]

    # --- Resolve price/qty ---
    pcol = _first_present(out, cfg_schema.get("price_col_candidates", []))
    qcol = _first_present(out, cfg_schema.get("qty_col_candidates", []))
    if pcol is None or qcol is None:
        missing = []
        if pcol is None: missing.append("price")
        if qcol is None: missing.append("qty")
        raise ValueError(f"Missing required columns: {missing}. Present: {list(out.columns)[:20]}")

    # --- Optional side / buyer_maker ---
    scol = _first_present(out, cfg_schema.get("side_col_candidates", []))
    bmcol = _first_present(out, cfg_schema.get("buyer_maker_col_candidates", []))

    # --- Build normalized frame ---
    norm = pd.DataFrame(index=pd.DatetimeIndex(ts.values, name="ts"))
    norm["price"] = pd.to_numeric(out[pcol], errors="coerce")
    norm["qty"] = pd.to_numeric(out[qcol], errors="coerce")

    if scol is not None:
        norm["side"] = out[scol].astype(str).str.upper()

    if bmcol is not None:
        # Handle 0/1, '0'/'1', True/False
        bm = out[bmcol]
        if bm.dtype.kind in "biu":      # bool/int/uint
            norm["is_buyer_maker"] = bm.astype(bool)
        else:
            # strings or mixed
            norm["is_buyer_maker"] = bm.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])

    # --- Clean & sort ---
    norm = norm.dropna(subset=["price","qty"])
    norm = norm[(norm["qty"] > 0) & np.isfinite(norm["price"])];
    norm = norm.sort_index(kind="mergesort").reset_index().rename(columns={"ts":"ts"})
    return norm

# ---------- Target computation ----------
def _one_minute_agg(ticks: pd.DataFrame) -> pd.DataFrame:
    g = ticks.set_index("ts").groupby(pd.Grouper(freq="1min", origin="start_day", label="right"))
    m = pd.DataFrame({
        "dollar": g.apply(lambda d: float((d["price"]*d["qty"]).sum())),
        "volume": g["qty"].sum(),
        "n": g["qty"].count().astype(float),
    })
    return m.fillna(0.0)

def compute_thresholds(ticks: pd.DataFrame, cfg_targets: dict) -> Dict[str, float]:
    mode = cfg_targets.get("mode", "ewma")
    if mode == "fixed":
        fx = cfg_targets.get("fixed", {}) or {}
        return {
            "dollar": float(fx.get("dollar_per_bar")) if fx.get("dollar_per_bar") else np.nan,
            "volume": float(fx.get("volume_per_bar")) if fx.get("volume_per_bar") else np.nan,
            "tick": float(fx.get("n_per_bar")) if fx.get("n_per_bar") else np.nan,
            "imbalance": np.nan,  # handled separately
        }

    # EWMA over 1-min aggregates → per-minute baseline
    one = _one_minute_agg(ticks)
    span = max(1, int(cfg_targets.get("ewma_span_minutes", 60)))
    bars_per_hour = max(1, int(cfg_targets.get("bars_per_hour", 60)))
    ew_dollar = one["dollar"].ewm(span=span, adjust=False).mean()
    ew_volume = one["volume"].ewm(span=span, adjust=False).mean()
    ew_n = one["n"].ewm(span=span, adjust=False).mean()

    # Per-bar target from per-minute baseline
    scale = 60.0 / float(bars_per_hour)
    thr = {
        "dollar": float(np.nanmedian(ew_dollar.values) * scale),
        "volume": float(np.nanmedian(ew_volume.values) * scale),
        "tick": float(np.nanmedian(ew_n.values) * scale),
        "imbalance": np.nan,  # dynamic; set later
    }
    # Fallbacks if zeros
    for k in ["dollar","volume","tick"]:
        if not np.isfinite(thr[k]) or thr[k] <= 0:
            # simple robust fallback from overall aggregates
            if k == "dollar":
                total = float((ticks["price"]*ticks["qty"]).sum())
            elif k == "volume":
                total = float(ticks["qty"].sum())
            else:
                total = float(len(ticks))
            bars_est = 60.0 * one.shape[0] / max(1.0, float(bars_per_hour))
            thr[k] = max(1.0, total / max(1.0, bars_est))
            LOG.warning("Fallback threshold for %s = %.4f", k, thr[k])
    return thr

# ---------- Sign + imbalance helpers ----------
def _signed_qty(row: pd.Series) -> float:
    if "side" in row:
        if row["side"] == "BUY":
            return float(row["qty"])
        elif row["side"] == "SELL":
            return -float(row["qty"])
    # If only is_buyer_maker is available:
    # Convention: if buyer is maker → seller is taker (aggressive SELL) → negative sign.
    if "is_buyer_maker" in row:
        return -float(row["qty"]) if bool(row["is_buyer_maker"]) else float(row["qty"])
    # default fallback: assume buy
    return float(row["qty"])

def _imbalance_threshold_stream(span_ticks: int, k: float):
    # Online EWM std tracker for signed imbalance magnitude
    alpha = 2.0 / (span_ticks + 1.0)
    mean, var = 0.0, 0.0
    def update(x_abs: float) -> float:
        nonlocal mean, var
        mean = (1 - alpha) * mean + alpha * x_abs
        # Welford-like EWM variance on |imbalance|
        diff = x_abs - mean
        var = (1 - alpha) * (var + alpha * diff * diff)
        std = math.sqrt(max(1e-12, var))
        return k * std
    return update

# ---------- Bar builders ----------
def _flush_bar(buf: dict) -> Optional[dict]:
    if buf["n_trades"] <= 0:
        return None
    # finalize OHLC
    return {
        "open": buf["open"],
        "high": buf["high"],
        "low": buf["low"],
        "close": buf["last_price"],
        "volume": float(buf["volume"]),
        "n_trades": int(buf["n_trades"]),
        "buy_maker_vol": float(buf["buy_maker_vol"]),
        "t_first": _to_utc(buf["t_first"]),
        "t_last": _to_utc(buf["t_last"]),
    }

def _init_buf() -> dict:
    return {
        "open": None, "high": -np.inf, "low": np.inf, "last_price": None,
        "volume": 0.0, "n_trades": 0, "buy_maker_vol": 0.0,
        "t_first": None, "t_last": None,
        "cum_dollar": 0.0, "cum_volume": 0.0, "cum_signed": 0.0,
    }

def build_bars_thresholded(ticks: pd.DataFrame, bar_type: str, thresholds: Dict[str, float], cfg: dict) -> pd.DataFrame:
    """Build bars by threshold crossing. bar_type in {'dollar','volume','tick','imbalance'}."""
    rows: List[dict] = []
    buf = _init_buf()

    thr_val = None
    if bar_type in ("dollar","volume","tick"):
        thr_val = float(thresholds[bar_type])
        if not np.isfinite(thr_val) or thr_val <= 0:
            raise ValueError(f"Invalid threshold for {bar_type}: {thr_val}")

    imb = cfg.get("imbalance", {}) or {}
    method = imb.get("method","ewm_std")
    span_ticks = int(imb.get("span_ticks", 2000))
    k = float(imb.get("k", 2.0))
    online_thresh = _imbalance_threshold_stream(span_ticks, k) if bar_type == "imbalance" and method == "ewm_std" else None

    signed_so_far_abs = 0.0  # for sqrtN method
    sqrtN_k = k

    for i, r in enumerate(ticks.itertuples(index=False), 1):
        price = float(r.price); qty = float(r.qty); ts = _to_utc(r.ts)
        side = getattr(r, "side", None)
        bm = bool(getattr(r, "is_buyer_maker", False))

        # initialize OHLC/time
        if buf["n_trades"] == 0:
            buf["open"] = price
            buf["t_first"] = ts

        buf["last_price"] = price
        buf["high"] = price if price > buf["high"] else buf["high"]
        buf["low"] = price if price < buf["low"] else buf["low"]
        buf["volume"] += qty
        buf["n_trades"] += 1
        if bm:
            buf["buy_maker_vol"] += qty
        buf["t_last"] = ts

        # accumulators for stopping rules
        buf["cum_dollar"] += price * qty
        buf["cum_volume"] += qty

        stop = False
        if bar_type == "tick":
            stop = (buf["n_trades"] >= thr_val)
        elif bar_type == "volume":
            stop = (buf["cum_volume"] >= thr_val)
        elif bar_type == "dollar":
            stop = (buf["cum_dollar"] >= thr_val)
        elif bar_type == "imbalance":
            s = _signed_qty(pd.Series({"qty": qty, "side": side, "is_buyer_maker": bm}))
            buf["cum_signed"] += s
            metric = abs(buf["cum_signed"])
            if method == "ewm_std":
                thr = online_thresh(abs(s))  # dynamic
                stop = metric >= thr
            else:
                # sqrtN rule on cumulative signed qty
                thr = sqrtN_k * math.sqrt(max(1, buf["n_trades"]))
                stop = metric >= thr
        else:
            raise ValueError(f"Unknown bar_type: {bar_type}")

        if stop:
            out = _flush_bar(buf)
            if out:
                rows.append(out)
            buf = _init_buf()

    # flush trailing partial bar
    out = _flush_bar(buf)
    if out:
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume","n_trades","buy_maker_vol","t_first","t_last"]).set_index(pd.DatetimeIndex([], name="t_last"))
    bars = pd.DataFrame(rows)
    # Ensure UTC tz-aware index using helper
    idx = pd.DatetimeIndex([_to_utc(x) for x in bars["t_last"].values], name="t_last")
    bars = bars.set_index(idx)
    # sanity
    bars.sort_index(inplace=True)
    return bars[["open","high","low","close","volume","n_trades","buy_maker_vol","t_first","t_last"]]

# ---------- IO helpers ----------
def write_parquet(path: str, df: pd.DataFrame, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use pyarrow metadata
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df)
        md = {k: str(v) for k, v in (meta or {}).items()}
        # merge metadata
        existing = table.schema.metadata or {}
        merged = existing.copy(); merged.update({k.encode(): v.encode() for k,v in md.items()})
        table = table.replace_schema_metadata(merged)
        tmp = path + ".__tmp__"
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except Exception as e:
        LOG.exception("pyarrow path failed, falling back to pandas parquet: %s", e)
        tmp = path + ".__tmp__"
        df.to_parquet(tmp, compression="snappy")
        os.replace(tmp, path)

def normalize_ticks(df: pd.DataFrame, cfg_schema: dict) -> pd.DataFrame:
    """
    Normalize ticks to columns: ts (UTC ns), price, qty, [side], [is_buyer_maker].

    Accepts either:
      • DatetimeIndex (preferred in our clean ticks), or
      • any time column from schema['time_col_candidates'].
    """
    from pandas.api import types as ptypes

    out = df.copy()

    # --- Resolve timestamp source (index or column) ---
    if isinstance(out.index, pd.DatetimeIndex):
        ts_raw = out.index
    else:
        tcol = _first_present(out, cfg_schema.get("time_col_candidates", []))
        if tcol is None:
            raise ValueError(f"Missing required time index/column; present cols: {list(out.columns)[:20]}")
        ts_raw = out[tcol]

    # --- Convert to a UTC DatetimeIndex safely (no NumPy subtype checks) ---
    # Numeric (epoch seconds/ms/ns) → direct unit conversion
    if ptypes.is_integer_dtype(ts_raw) or ptypes.is_float_dtype(ts_raw):
        unit = cfg_schema.get("epoch_unit", "ms")
        if unit not in ("ms", "s", "ns"):
            unit = "ms"
        idx = pd.to_datetime(ts_raw, unit=unit, utc=True, errors="coerce")
    else:
        # Strings / datetime-like (with or without tz)
        idx = pd.to_datetime(ts_raw, utc=False, errors="coerce")

    # Ensure DatetimeIndex
    idx = pd.DatetimeIndex(idx)

    # Localize/convert to UTC
    if idx.tz is None:
        idx = idx.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        idx = idx.tz_convert("UTC")

    # Drop any unparsable timestamps
    if idx.isna().any():
        mask = ~idx.isna()
        out = out.loc[mask].copy()
        idx = idx[mask]

    # --- Resolve required columns ---
    pcol = _first_present(out, cfg_schema.get("price_col_candidates", []))
    qcol = _first_present(out, cfg_schema.get("qty_col_candidates", []))
    if pcol is None or qcol is None:
        missing = []
        if pcol is None: missing.append("price")
        if qcol is None: missing.append("qty")
        raise ValueError(f"Missing required columns: {missing}. Present: {list(out.columns)[:20]}")

    # --- Optional columns ---
    scol = _first_present(out, cfg_schema.get("side_col_candidates", []))
    bmcol = _first_present(out, cfg_schema.get("buyer_maker_col_candidates", []))

    # --- Build normalized frame ---
    norm = pd.DataFrame(index=idx.rename("ts"))
    norm["price"] = pd.to_numeric(out[pcol], errors="coerce")
    norm["qty"] = pd.to_numeric(out[qcol], errors="coerce")

    if scol is not None:
        norm["side"] = out[scol].astype(str).str.upper()

    if bmcol is not None:
        bm = out[bmcol]
        if bm.dtype.kind in "biu":  # bool/int/uint
            norm["is_buyer_maker"] = bm.astype(bool)
        else:
            norm["is_buyer_maker"] = bm.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])

    # Clean and sort
    norm = norm.dropna(subset=["price","qty"])
    norm = norm[(norm["qty"] > 0) & np.isfinite(norm["price"])].copy()
    norm = norm.sort_index(kind="mergesort").reset_index().rename(columns={"ts": "ts"})
    return norm

def load_merge_ticks(patterns: List[str], cfg_schema: dict) -> pd.DataFrame:
    # Expand primary patterns plus safe fallbacks for common layouts
    primary = []
    for pat in patterns:
        primary.extend(glob.glob(pat))

    # If nothing matched, try common clean-tick fallbacks
    if not primary:
        # Expecting monthly parquet in data/ticks_clean/<SYM>/<YYYY-MM>.parquet
        # Extract sym & month from the formatted pattern if possible
        files = []
        for pat in patterns:
            # Collect sym & ym heuristically
            # ... pattern examples include data/ticks_clean/{sym}/{year_month}.parquet
            # We fallback to broader globs to assist
            parts = pat.replace("\\", "/").split("/")
            try:
                sym = parts[2]
            except Exception:
                sym = "*"
            ym = None
            for token in parts:
                # Accept tokens that either are exactly YYYY-MM or start with it (e.g., '2025-01.parquet')
                if len(token) >= 7 and token[:4].isdigit() and token[4] == "-" and token[5:7].isdigit():
                    ym = token[:7]
                    break
            if ym is None:
                ym = "*-*"
            files.extend(glob.glob(f"data/ticks_clean/{sym}/{ym}.parquet"))
            files.extend(glob.glob(f"data/ticks_clean/{sym}/{ym}-*.parquet"))
            files.extend(glob.glob(f"data/ticks_clean/{sym}/*{ym}*.parquet"))
            # Last-ditch: raw directory
            #  - Subfolder layout: data/ticks_raw/<SYM>/*<YYYY-MM>*
            files.extend(glob.glob(f"data/ticks_raw/{sym}/*{ym}*.parquet"))
            files.extend(glob.glob(f"data/ticks_raw/{sym}/*{ym}*.csv"))
            #  - Flat layout: data/ticks_raw/<SYM>-ticks-<YYYY-MM>[...].*
            files.extend(glob.glob(f"data/ticks_raw/{sym}-ticks-{ym}.parquet"))
            files.extend(glob.glob(f"data/ticks_raw/{sym}-ticks-{ym}.csv"))
            files.extend(glob.glob(f"data/ticks_raw/{sym}-ticks-{ym}-*.parquet"))
            files.extend(glob.glob(f"data/ticks_raw/{sym}-ticks-{ym}-*.csv"))
        primary = sorted({f for f in files})

    if not primary:
        raise FileNotFoundError(f"No tick files matched: {patterns}")

    parts = []
    for f in sorted(primary):
        try:
            if f.lower().endswith(".parquet"):
                d = pd.read_parquet(f)
            else:
                d = pd.read_csv(f)
            parts.append(d)
        except Exception as e:
            LOG.error("Failed to read %s: %s", f, e)

    if not parts:
        raise FileNotFoundError(f"All matched files failed to read: {primary}")

    # Preserve index if present (clean ticks store DatetimeIndex)
    try:
        ticks = pd.concat(parts, ignore_index=False)
    except Exception:
        ticks = pd.concat(parts, ignore_index=True)
    ticks = normalize_ticks(ticks, cfg_schema)
    return ticks

def build_one_month(sym: str, year_month: str, out_dir: str, cfg: dict) -> Dict[str, str]:
    """Return dict bar_type -> output path written (if created)."""
    io = cfg["io"]; schema = cfg["schema"]; targets_cfg = cfg["targets"]; imb_cfg = cfg.get("imbalance",{}) or {}
    tick_glob = io["tick_glob"].format(sym=sym, year_month=year_month)
    LOG.info("Loading ticks for %s %s: %s", sym, year_month, tick_glob)
    ticks = load_merge_ticks([tick_glob], schema)

    # compute thresholds
    thr = compute_thresholds(ticks, targets_cfg)

    # Respect optional runtime.only_bars filter
    runtime_cfg = cfg.get("runtime", {}) or {}
    only_bars = runtime_cfg.get("only_bars") or None
    only_set = set(only_bars) if only_bars else None

    # Build non-imbalance bars first, then imbalance with autotune
    written = {}
    ref_counts = {}

    # 1) Build non-imbalance first
    for bar_type in ["dollar","volume","tick"]:
        out_path = os.path.join(out_dir, sym, bar_type, f"{year_month}.parquet")
        # If user limited bar types, try to read counts for reference but don't build
        if only_set is not None and bar_type not in only_set:
            if os.path.exists(out_path):
                try:
                    ref_df = pd.read_parquet(out_path)
                    ref_counts[bar_type] = len(ref_df)
                except Exception:
                    pass
            continue
        if os.path.exists(out_path) and cfg["io"].get("skip_existing", True):
            LOG.info("Exists, skipping: %s", out_path)
            # still record a reference count if we can
            try:
                ref_df = pd.read_parquet(out_path)
                ref_counts[bar_type] = len(ref_df)
            except Exception:
                pass
            continue
        LOG.info("Building %s bars for %s %s", bar_type, sym, year_month)
        bars = build_bars_thresholded(ticks, bar_type, thr, cfg)
        write_parquet(out_path, bars, {
            "symbol": sym, "bar_type": bar_type, "year_month": year_month,
            "threshold_dollar": thr.get("dollar"),
            "threshold_volume": thr.get("volume"),
            "threshold_tick": thr.get("tick"),
            "imbalance_method": cfg.get("imbalance",{}).get("method","ewm_std"),
            "imbalance_span_ticks": cfg.get("imbalance",{}).get("span_ticks", 2000),
            "imbalance_k": cfg.get("imbalance",{}).get("k", 2.0),
            "note": "Stage 2 Information Bars; feature substrate only.",
        })
        LOG.info("Wrote %d rows → %s", len(bars), out_path)
        written[bar_type] = out_path
        ref_counts[bar_type] = len(bars)

    # 2) Build imbalance with autotune
    bar_type = "imbalance"
    out_path = os.path.join(out_dir, sym, bar_type, f"{year_month}.parquet")
    # If limited bar types and imbalance not requested, stop here
    if only_set is not None and bar_type not in only_set:
        return written
    if os.path.exists(out_path) and cfg["io"].get("skip_existing", True):
        LOG.info("Exists, skipping: %s", out_path)
        written[bar_type] = out_path
    else:
        imb_cfg_local = (cfg.get("imbalance") or {}).copy()
        autotune = (imb_cfg_local.get("autotune") or {})
        auto_enabled = bool(autotune.get("enabled", False))
        ref_key = str(autotune.get("reference", "tick"))
        max_mult = float(autotune.get("max_multiplier", 5.0))
        scale = float(autotune.get("scale_factor", 1.5))
        max_pass = int(autotune.get("max_passes", 4))

        k_base = float(imb_cfg_local.get("k", 2.0))
        k_curr = k_base

        LOG.info("Building %s bars for %s %s (k=%.4f)", bar_type, sym, year_month, k_curr)

        def _build_with_k(kval: float) -> pd.DataFrame:
            cfg_local = {**cfg}
            cfg_local["imbalance"] = {**(cfg.get("imbalance") or {}), "k": float(kval)}
            return build_bars_thresholded(ticks, bar_type, thr, cfg_local)

        bars = _build_with_k(k_curr)

        if auto_enabled and ref_key in ref_counts:
            ref_n = max(1, int(ref_counts[ref_key]))
            passes = 0
            while len(bars) > max_mult * ref_n and passes < max_pass:
                passes += 1
                k_curr *= scale
                LOG.warning(
                    "Imbalance rows %d exceed %.2fx %s rows (%d). Auto-raising k to %.4f and rebuilding...",
                    len(bars), max_mult, ref_key, ref_n, k_curr
                )
                bars = _build_with_k(k_curr)

        write_parquet(out_path, bars, {
            "symbol": sym, "bar_type": bar_type, "year_month": year_month,
            "threshold_dollar": thr.get("dollar"),
            "threshold_volume": thr.get("volume"),
            "threshold_tick": thr.get("tick"),
            "imbalance_method": cfg.get("imbalance",{}).get("method","ewm_std"),
            "imbalance_span_ticks": cfg.get("imbalance",{}).get("span_ticks", 2000),
            "imbalance_k": k_curr,
            "note": "Stage 2 Information Bars; feature substrate only. Auto-tuned k captured here.",
        })
        LOG.info("Wrote %d rows (imbalance, k=%.4f) → %s", len(bars), k_curr, out_path)
        written[bar_type] = out_path

    return written
    written = {}
    for bar_type in ["dollar","volume","tick","imbalance"]:
        out_path = os.path.join(out_dir, sym, bar_type, f"{year_month}.parquet")
        if os.path.exists(out_path) and cfg["io"].get("skip_existing", True):
            LOG.info("Exists, skipping: %s", out_path)
            continue

        LOG.info("Building %s bars for %s %s", bar_type, sym, year_month)
        bars = build_bars_thresholded(ticks, bar_type, thr, cfg)
        meta = {
            "symbol": sym,
            "bar_type": bar_type,
            "year_month": year_month,
            "threshold_dollar": thr.get("dollar"),
            "threshold_volume": thr.get("volume"),
            "threshold_tick": thr.get("tick"),
            "imbalance_method": imb_cfg.get("method","ewm_std"),
            "imbalance_span_ticks": imb_cfg.get("span_ticks", 2000),
            "imbalance_k": imb_cfg.get("k", 2.0),
            "note": "Stage 2 Information Bars; feature substrate only.",
        }
        write_parquet(out_path, bars, meta)
        LOG.info("Wrote %s rows → %s", len(bars), out_path)
        written[bar_type] = out_path
    return written
