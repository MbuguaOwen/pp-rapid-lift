import argparse, os, sys, logging
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime
from src.bars.info_bars import build_one_month

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 — Build Information Bars (dollar/volume/tick/imbalance).")
    p.add_argument("--config", required=True, help="Path to configs/info_bars.yaml")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols like BTCUSDT SOLUSDT ETHUSDT")
    p.add_argument("--months", nargs="+", required=True, help="YYYY-MM tokens, e.g., 2025-01 2025-02 ...")
    p.add_argument("--force", action="store_true", help="Override skip_existing and overwrite outputs")
    p.add_argument("--exclude-months", nargs="*", default=None, help="YYYY-MM tokens to skip (takes precedence over config blacklist)")
    p.add_argument("--only-bars", nargs="+", default=None, choices=["dollar","volume","tick","imbalance"], help="Limit processing to these bar types")
    return p.parse_args()

def main():
    setup_logging()
    log = logging.getLogger("make_info_bars")
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.force:
        cfg["io"]["skip_existing"] = False

    # Build blacklist from config and CLI override
    cfg_blacklist = set(cfg.get("io", {}).get("month_blacklist", []) or [])
    cli_blacklist = set(args.exclude_months or [])
    blacklist = cfg_blacklist.union(cli_blacklist)

    # Pass only-bars selection into runtime config for downstream use
    if args.only_bars:
        cfg.setdefault("runtime", {})["only_bars"] = list(args.only_bars)

    out_dir = cfg["io"]["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    total = len(args.symbols) * len(args.months)
    i = 0
    for sym in args.symbols:
        for ym in tqdm(args.months, desc=f"Building info bars", unit="month"):
            if ym in blacklist:
                log.warning("Skipping blacklisted month %s for %s", ym, sym)
                continue
            i += 1
            try:
                written = build_one_month(sym, ym, out_dir, cfg)
                if not written:
                    log.warning("No files created for %s %s (maybe all existed).", sym, ym)
            except FileNotFoundError as e:
                log.error("Missing ticks for %s %s: %s", sym, ym, e)
            except Exception as e:
                log.exception("Failed %s %s: %s", sym, ym, e)

if __name__ == "__main__":
    main()
