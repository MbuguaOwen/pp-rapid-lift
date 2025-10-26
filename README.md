# Stage 1 — Tick Ingestion & 1-Minute OHLCV

This stage standardizes raw tick CSVs (months 01–06) into:
- Clean tick Parquet (`data/ticks_clean/<SYMBOL>/<YYYY-MM>.parquet`)
- 1-min OHLCV Parquet (`data/ohlcv_1m/<SYMBOL>/<YYYY-MM>.parquet`)
- QA catalog with SHA256 hashes (`reports/qa/catalog_ticks.csv`)

## Install
```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Configure

Edit configs/ingestion.yaml if your CSV column names or timestamp units differ.

Run
python scripts/ingest_ticks.py
python scripts/make_ohlcv.py

Sanity Checks (Python REPL)
import pandas as pd
P = "data/ohlcv_1m/BTCUSDT/2025-03.parquet"
x = pd.read_parquet(P)
assert (x[['open','high','low','close','volume']].notna().sum(axis=1)>0).all()
assert (x['low']<=x[['open','close','high']].min(axis=1)).all()
assert (x['high']>=x[['open','close','low']].max(axis=1)).all()
print(x.index.tz)           # should be UTC
print(x.iloc[[0,-1]])

Month Seam Check
a = pd.read_parquet("data/ohlcv_1m/BTCUSDT/2025-03.parquet")
b = pd.read_parquet("data/ohlcv_1m/BTCUSDT/2025-04.parquet")
assert a.index.max() < b.index.min() or a.index.max() == b.index.min() - pd.Timedelta(minutes=1)

Definition of Done

All months 01–06 exist for BTCUSDT, SOLUSDT, ETHUSDT in clean and 1-min OHLCV form

All hard checks pass (UTC, monotonic, deduped, no NA in price/qty, OHLC invariants)

reports/qa/catalog_ticks.csv written with SHA256 for raw & clean files

# Stage 2 — Information Bars (Micro Context, Not Labels)

**Inputs (monthly clean ticks):** `data/ticks_clean/<SYMBOL>/<YYYY-MM>.parquet`  
**Common schema:** DatetimeIndex (UTC or naive), columns: `price`, `qty`, `buy_maker`  
**Outputs:** `data/info_bars/<SYMBOL>/<dollar|volume|tick|imbalance>/<YYYY-MM>.parquet`

**Bar columns:** `open, high, low, close, volume, n_trades, buy_maker_vol, t_first, t_last`  
**Index:** `t_last` (UTC), sorted.

**Why:** More stationary microstructure for motif/pattern discovery and micro features.  
These rows are not labeled.

## Run
```powershell
$env:PYTHONPATH = "$PWD"
python scripts\make_info_bars.py `
  --config "configs/info_bars.yaml" `
  --symbols BTCUSDT `
  --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07
```

Troubleshooting

- Error: Missing required columns: ['time']  
  Stage 2 now reads the DatetimeIndex when no time column exists. Make sure your Parquet preserved the index; our loader also falls back to columns in `schema.time_col_candidates`.

- Only price, qty, buy_maker present  
  That’s OK. `buy_maker` is auto-mapped to `is_buyer_maker`. If you have `side` (BUY/SELL), it takes precedence for imbalance sign.

- Index timezone  
  If your index is naive, we localize to UTC. If tz-aware, we convert to UTC.

- TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type  
  Fixed: Stage 2 now uses pandas dtype checks and converts any index/column to a UTC DatetimeIndex.

- ValueError: Cannot pass a datetime or Timestamp with tzinfo with the tz parameter  
  Fixed by normalizing timestamps via a helper that localizes/converts to UTC. We no longer call `pd.Timestamp(..., tz=...)` on tz-aware values.

### Guardrails

- Blacklist months in `configs/info_bars.yaml → io.month_blacklist: [...]` or pass `--exclude-months` to the CLI.
- Imbalance autotune: If imbalance rows > `max_multiplier ×` the reference bar (default `tick`), `k` is multiplied by `scale_factor` and rebuilt (up to `max_passes`). Final `k` is stored in Parquet metadata.

### Quick summary

```powershell
python scripts\summarize_info_bars.py --symbol BTCUSDT
```
