import argparse, os, yaml, logging
from tqdm import tqdm
from src.bars.time_bars import load_ticks, build_time_bars, write_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("make_time_bars")

def parse_args():
    p = argparse.ArgumentParser("Stage 4a — Build 1-min time bars from ticks (live parity)")
    p.add_argument("--config", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = cfg["io"]["out_dir"]
    skip = cfg["io"].get("skip_existing", True) and not args.force
    for sym in args.symbols:
        for ym in tqdm(args.months, desc=f"time bars 1min", unit="month"):
            tick_glob = cfg["io"]["tick_glob"].format(sym=sym, year_month=ym)
            out_path = os.path.join(out_dir, sym, f"{ym}.parquet")
            if skip and os.path.exists(out_path):
                LOG.info("Exists, skipping %s", out_path); continue
            LOG.info("Loading %s", tick_glob)
            ticks = load_ticks(tick_glob, cfg["schema"])
            LOG.info("Aggregating 1-min bars")
            bars = build_time_bars(ticks, cfg["bars"]["freq"])
            write_parquet(out_path, bars, {"symbol": sym, "freq": cfg["bars"]["freq"], "year_month": ym})
            LOG.info("Wrote %s rows -> %s", len(bars), out_path)

if __name__ == "__main__":
    main()

