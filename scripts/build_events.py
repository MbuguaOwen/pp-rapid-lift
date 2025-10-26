from pathlib import Path
import pandas as pd, yaml
from tqdm import tqdm

def load_cfg(p="configs/ingestion.yaml"): 
    with open(p,"r") as f: return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    out_root = Path("data/events"); out_root.mkdir(parents=True, exist_ok=True)
    syms, months = cfg["symbols"], cfg["months"]
    bar_types = cfg["info_bars"]["build"]

    for sym in syms:
        for month in months:
            rows = []
            for btype in bar_types:
                p = Path(cfg["info_bars"]["out_dir"]) / sym / btype / f"{month}.parquet"
                if not p.exists(): 
                    tqdm.write(f"[SKIP] {p}"); continue
                bars = pd.read_parquet(p)
                bars = bars.reset_index().rename(columns={"t_last":"event_ts"})
                bars["bar_type"] = btype
                rows.append(bars[["event_ts","bar_type","t_first","event_ts"]].rename(columns={"event_ts":"t_last"}))

            if not rows: 
                continue
            E = pd.concat(rows, ignore_index=True)
            E = E.sort_values("t_last").drop_duplicates("t_last")  # one event per tick instant if clashes
            E = E.rename(columns={"t_last":"event_ts"})
            # attach a monotonic tick_id within month (ordinal rank)
            E["event_tick_id"] = E["event_ts"].rank(method="first").astype("int64")
            out = Path("data/events")/sym/f"{month}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            E.to_parquet(out)
            tqdm.write(f"[OK] events {sym} {month}: {len(E)} -> {out}")

if __name__=="__main__":
    main()

