#!/usr/bin/env python
import argparse, os, sys, logging
from pathlib import Path
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("qa_info_bars")

INFO_ROOT = Path("data/info_bars")

def _count_rows(p: Path) -> int:
    if not p.exists(): return 0
    # read a single light column for speed
    try:
        return len(pd.read_parquet(p, columns=["close"]))
    except Exception:
        return len(pd.read_parquet(p))

def _read_k_meta(p: Path):
    if pq is None or not p.exists():
        return None
    try:
        md = pq.read_metadata(str(p)).metadata or {}
        # keys are bytes in pyarrow metadata
        for k,v in md.items():
            if (k.decode() if isinstance(k, bytes) else k) == "imbalance_k":
                s = v.decode() if isinstance(v, bytes) else str(v)
                try:
                    return float(s)
                except Exception:
                    return s
    except Exception:
        return None
    return None

def gather(symbol: str, months: list[str], max_ratio: float):
    rows = []
    for ym in months:
        tick_p = INFO_ROOT / symbol / "tick" / f"{ym}.parquet"
        imb_p  = INFO_ROOT / symbol / "imbalance" / f"{ym}.parquet"
        t = _count_rows(tick_p)
        i = _count_rows(imb_p)
        k = _read_k_meta(imb_p)
        ratio = (i / t) if (t > 0) else float("inf")
        rows.append({"month": ym, "tick": t, "imbalance": i, "ratio": ratio, "k": k,
                     "tick_path": str(tick_p), "imb_path": str(imb_p)})
    df = pd.DataFrame(rows)
    df = df.sort_values("month")
    bad = df[(df["tick"] > 0) & (df["ratio"] > max_ratio)]
    return df, bad

def main():
    ap = argparse.ArgumentParser("QA for Stage 2 information bars")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--months", nargs="*", help="YYYY-MM ... (default: infer from tick dir)")
    ap.add_argument("--max-ratio", type=float, default=2.0, help="allowable imbalance/tick ratio")
    args = ap.parse_args()

    # infer months if not provided
    if not args.months:
        tick_dir = INFO_ROOT / args.symbol / "tick"
        months = sorted([p.stem for p in tick_dir.glob("*.parquet")])
    else:
        months = args.months

    if not months:
        LOG.error("No months found. Check %s", INFO_ROOT / args.symbol / "tick")
        sys.exit(2)

    df, bad = gather(args.symbol, months, args.max_ratio)
    # Pretty print
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df[["month","tick","imbalance","ratio","k"]])

    if len(bad):
        LOG.error("Failed QA: %d month(s) exceed max-ratio=%.2f", len(bad), args.max_ratio)
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print("\nOffenders:\n", bad[["month","tick","imbalance","ratio","k"]])
        sys.exit(1)
    else:
        LOG.info("QA pass: all months ratio ≤ %.2f", args.max_ratio)

if __name__ == "__main__":
    main()
