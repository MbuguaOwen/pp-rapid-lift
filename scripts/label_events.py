from pathlib import Path
import pandas as pd, numpy as np, yaml
from tqdm import tqdm

def load_cfg(p="configs/ingestion.yaml"):
    with open(p,"r") as f: return yaml.safe_load(f)

def ewma_sigma_1m(ohlcv, span_min=60):
    r = np.log(ohlcv["close"]).diff()
    vol = r.ewm(span=span_min, adjust=False, min_periods=span_min//3).std()
    return vol  # per-minute vols aligned to minute index

def first_passage_label(ticks, event_ts, sigma, tp_mult=3.0, sl_mult=3.0, horizon_secs=1800):
    # ticks indexed by UTC DatetimeIndex; has 'price'
    # event price
    p0 = float(ticks.loc[:event_ts].iloc[-1]["price"])
    # use minute-vol sigma aligned to event minute (fallback to median)
    sig = float(sigma.reindex([event_ts.floor("min")]).ffill().iloc[0] if len(sigma) else 0.0)
    sig = max(sig, np.nanmedian(sigma.values)) if not np.isfinite(sig) else sig

    up = p0 * np.exp(+tp_mult * sig)
    dn = p0 * np.exp(-sl_mult * sig)

    # slice future ticks up to vertical barrier
    fut = ticks.loc[event_ts + pd.Timedelta(microseconds=1) : event_ts + pd.Timedelta(seconds=horizon_secs)]
    if fut.empty:
        return 0, event_ts, "vb", 0.0, 0.0

    hit_up = (fut["price"] >= up)
    hit_dn = (fut["price"] <= dn)

    t_up = hit_up.idxmax() if hit_up.any() else None
    t_dn = hit_dn.idxmax() if hit_dn.any() else None

    if t_up is not None and (t_dn is None or t_up <= t_dn):
        ret = np.log(float(fut.loc[t_up,"price"])/p0)
        return +1, t_up, "tp", ret, np.log(float(fut.iloc[-1]["price"])/p0)
    if t_dn is not None and (t_up is None or t_dn <= t_up):
        ret = np.log(float(fut.loc[t_dn,"price"])/p0)
        return -1, t_dn, "sl", ret, np.log(float(fut.iloc[-1]["price"])/p0)

    # vertical barrier
    ret_v = np.log(float(fut.iloc[-1]["price"])/p0)
    y = +1 if ret_v > 0 else (-1 if ret_v < 0 else 0)
    return y, fut.index[-1], "vb", ret_v, ret_v

def main():
    cfg = load_cfg()
    syms, months = cfg["symbols"], cfg["months"]

    for sym in syms:
        for month in months:
            p_events = Path("data/events")/sym/f"{month}.parquet"
            p_ticks  = Path(cfg["paths"]["clean_dir"]) / sym / f"{month}.parquet"
            p_1m     = Path(cfg["paths"]["ohlcv_1m_dir"]) / sym / f"{month}.parquet"
            if not (p_events.exists() and p_ticks.exists() and p_1m.exists()):
                tqdm.write(f"[SKIP] {sym} {month} inputs missing"); continue

                
            E = pd.read_parquet(p_events)
            ticks = pd.read_parquet(p_ticks)
            ohlcv = pd.read_parquet(p_1m)

            ticks.index = ticks.index.tz_convert("UTC") if ticks.index.tz is not None else ticks.index.tz_localize("UTC")
            sigma = ewma_sigma_1m(ohlcv, span_min=cfg.get("labels",{}).get("ewma_span_min", 60))

            tp_mult = cfg.get("labels",{}).get("tp_mult", 3.0)
            sl_mult = cfg.get("labels",{}).get("sl_mult", 3.0)
            horizon = cfg.get("labels",{}).get("horizon_secs", 1800)

            out_rows = []
            for ts in tqdm(E["event_ts"], desc=f"Labeling {sym} {month}", unit="ev"):
                y, t_hit, hit_type, ret_hit, ret_v = first_passage_label(
                    ticks, ts, sigma, tp_mult, sl_mult, horizon
                )
                out_rows.append({"event_ts": ts, "label": y, "t_hit": t_hit, "hit_type": hit_type,
                                 "ret_at_hit": ret_hit, "ret_at_vbar": ret_v})

            L = pd.DataFrame(out_rows).set_index("event_ts").sort_index()
            out = Path("data/labels")/sym/f"{month}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            L.to_parquet(out)
            tqdm.write(f"[OK] labels {sym} {month}: {len(L)} -> {out}")

if __name__ == "__main__":
    main()

