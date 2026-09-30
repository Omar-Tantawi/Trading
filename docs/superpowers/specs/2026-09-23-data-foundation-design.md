# Design: AI Trading Buddy — Sub-project 1: Data Foundation

Date: 2026-09-23
Status: Approved for planning
Milestone: 1 of 10 (see "Programme context" below)

## 1. Purpose

Build trustworthy market-data infrastructure for the AI Trading Buddy: a
complete, verified, gap-checked history of Binance market data plus a live
collector, stored so that every later subsystem (features, ML models,
backtesting, signals, risk, bot monitoring, bot advisor, LLM assistant,
dashboard) reads from one source of truth.

Nothing in this sub-project predicts, recommends or trades. Its single
success criterion: **downstream code can trust the data, and knows when it
cannot.**

### Success criteria

1. 1-minute candle history for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT from each
   symbol's listing date to the present is loaded and verified.
2. Higher timeframes (5m, 15m, 1h, 4h, 1d) are derived from the 1m data and
   provably agree with it.
3. A live collector keeps the database current and recovers from
   disconnections without silent gaps.
4. A data-quality engine produces a per-symbol PASS / WARN / FAIL report, and
   a FAIL blocks model training.
5. Every component is restartable at any moment without corrupting data.

### Non-goals for this sub-project

- No indicators, features, models, signals, risk logic, bots or UI.
- No order-book depth history (not available from Binance for past dates).
- No individual-trade history (tens of GB per symbol; not needed yet).
- No image upload, vision model or visual training library. **These were
  removed from the programme at the user's request** (original spec sections
  42-50, 88-90, 93, 96 and Milestone 10 are dropped).

## 2. Programme context

Agreed build order for the whole platform. Each sub-project gets its own
design, implementation plan and build cycle.

| # | Sub-project | Original spec sections |
|---|---|---|
| **1** | **Data foundation (this document)** | 5-8, 105 |
| 2 | Market intelligence: indicators, trend, volatility, volume, candles, S/R, patterns, regime. **Split (user-approved 2026-09-30):** 2a = features, indicators, trend, volatility, volume, candles, rules-based regime (`2026-09-30-market-intelligence-2a-design.md`); 2b = S/R, Market Profile, chart patterns | 9-17, 106 |
| 3 | Prediction ML: labels, baselines, XGBoost, walk-forward, calibration | 18-23, 30-33, 107 |
| 3.5 | **Minimal read-only dashboard** (user-approved 2026-09-23): candlestick charts of collected data plus model accuracy and calibration. Exists so the data and the model can be *seen* rather than read about in summaries. Small and deliberately unpolished; sub-project 9 builds the real thing | 85, 87 (partial) |
| 4 | Backtesting and financial evaluation | 28-29, 108 |
| 5 | Signals, risk, position sizing, portfolio | 24-27, 109 |
| 6 | Binance bot monitoring | 51-58, 110 |
| 7 | Binance bot advisor (suitability, capital, duration, stop rules, leverage) | 59-80, 111 |
| 8 | LLM trading assistant | 37-41, 91-92, 112 |
| 9 | Dashboard | 81-87, 104, 113 |
| 10 | Demo trading, shadow mode, controlled real deployment | 34-36, 115-117 |

## 3. Environment decisions

Decided with the user during brainstorming:

- **Runs on the user's Windows 11 PC now; may move to a VPS later.** Therefore
  all configuration is environment-driven; no hardcoded paths; database
  reached by URL.
- **Database: PostgreSQL 16 + TimescaleDB, in Docker** (`docker compose up`).
  Identical on a VPS. Docker Desktop must be installed (requires WSL2).
- **Python 3.13.3, already installed on the machine** (the `py` launcher
  defaults to it; the old 3.9 stays untouched and unused). Current
  pandas / XGBoost / LightGBM require >= 3.10, so 3.9 is not an option. A
  plain `venv` is used rather than `uv`, since no extra Python install is
  needed.
- **Data scope: candles only** (option A). Live best bid/ask is recorded from
  day one; live order-book depth recording is deferred to a later sub-project
  because order-flow features need months of recorded data before they are
  trainable.

## 4. Architecture

```text
            data.binance.vision                 Binance REST + WebSocket
          (monthly/daily zip archives)             (recent + live)
                    |                                     |
                    v                                     v
        +-----------------------+              +------------------------+
        | Historical Backfill   |              |    Live Collector      |
        | download -> verify    |              | closed 1m candles      |
        | SHA256 -> parse ->    |              | + best bid/ask         |
        | bulk COPY (resumable) |              | reconnect + REST gap   |
        +-----------+-----------+              |  backfill              |
                    |                          +-----------+------------+
                    |                                      |
                    v                                      v
        +-------------------------------------------------------------+
        |  PostgreSQL 16 + TimescaleDB (Docker)                       |
        |  candles_1m (hypertable, compressed after 7d)               |
        |    -> continuous aggregates: 5m / 15m / 1h / 4h / 1d        |
        |  book_ticker, symbols, ingestion_runs,                      |
        |  data_quality_reports                                       |
        +-----------------------------+-------------------------------+
                                      |
                                      v
                        +-----------------------------+
                        |     Data-Quality Engine     |
                        |  gaps, duplicates, ordering |
                        |  OHLC sanity, volume sanity |
                        +--------------+--------------+
                                       |
                                       v
                          PASS / WARN / FAIL report
                          (FAIL raises and blocks training)
```

### Components

Five units, each with one responsibility, a defined interface, and
independent tests.

1. **Config** (`data/config.py`)
   - Reads `.env` via `pydantic-settings`. Fields: `DATABASE_URL`,
     `SYMBOLS`, `BACKFILL_START`, `BINANCE_BASE_URL`, `BINANCE_WS_URL`,
     `DATA_CACHE_DIR`, `LOG_LEVEL`.
   - Depends on: nothing. Used by: everything.

2. **Storage layer** (`data/storage/`)
   - Owns the schema, migrations and all writes/reads for this sub-project.
   - Public interface: `upsert_candles(symbol, rows)`,
     `upsert_book_ticker(rows)`, `get_candles(symbol, timeframe, start, end)`,
     `last_candle_time(symbol)`, `record_run(...)`, `record_quality_report(...)`.
   - All writes idempotent (`ON CONFLICT DO UPDATE`, or `COPY` into a staging
     table then merge).

3. **Historical backfill** (`data/collectors/backfill.py`)
   - For each symbol and each month from listing to present: download
     `klines/<SYMBOL>/1m/<SYMBOL>-1m-<YYYY-MM>.zip` from data.binance.vision,
     download its `.CHECKSUM`, **verify SHA256**, unzip, parse CSV, bulk load.
   - The current (incomplete) month uses daily archives; the last archive-free
     days come from `GET /api/v3/klines` paginated at 1000 candles per call.
   - Resumable: consults `ingestion_runs` and `last_candle_time` and skips
     completed periods. Safe to re-run at any time.

4. **Live collector** (`data/collectors/live.py`)
   - Combined WebSocket stream: `<symbol>@kline_1m` and `<symbol>@bookTicker`
     for all configured symbols on one connection.
   - **Only writes a 1m candle when `k.x == true`** (candle closed), so no
     partial candle ever enters the database.
   - `book_ticker` messages are batched and flushed every N seconds.
   - On disconnect: exponential backoff reconnect, then REST-backfill every
     minute between `last_candle_time(symbol)` and now, before resuming.
   - Writes a heartbeat so staleness can be detected.

5. **Data-quality engine** (`data/quality/`)
   - Checks per symbol and timeframe (original spec section 8):
     missing timestamps / gaps; duplicate candles; non-monotonic ordering;
     impossible OHLC (`high < low`, `high < open|close`, `low > open|close`);
     negative or null volume; zero-volume runs; null prices; completeness %.
   - Produces a report (stored and printed) in the format from the original
     spec, with verdict PASS / WARN / FAIL.
   - Public interface: `run_quality_checks(symbol, timeframe) -> Report` and
     `assert_trainable(symbols)` which **raises** on FAIL. Sub-project 3 calls
     `assert_trainable` before any training run.

### CLI

`tb` (Typer), the single entry point:

- `tb db upgrade` — apply migrations.
- `tb symbols sync` — refresh symbol metadata (tick size, lot size, status)
  from Binance `exchangeInfo`.
- `tb backfill [--symbol S] [--from YYYY-MM-DD]` — historical load; resumable.
- `tb live` — run the live collector until stopped.
- `tb quality [--symbol S] [--timeframe 1m]` — run checks, print report, store it.
- `tb status` — last candle time, row counts, last quality verdict per symbol.

## 5. Database schema

Created by migration in this sub-project; later sub-projects add their own
tables (features, regimes, predictions, signals, bots, journal, models).

```sql
symbols(
  symbol text primary key,
  base_asset text, quote_asset text,
  tick_size numeric, step_size numeric,
  min_notional numeric,
  listed_at timestamptz,
  status text,            -- TRADING, BREAK, ...
  is_active boolean default true,
  updated_at timestamptz default now()
)

candles_1m(
  symbol text references symbols(symbol),
  open_time timestamptz not null,      -- UTC, candle OPEN
  close_time timestamptz not null,
  open numeric(20,8), high numeric(20,8),
  low numeric(20,8),  close numeric(20,8),
  volume numeric(30,8),                -- base asset
  quote_volume numeric(30,8),
  trade_count integer,
  taker_buy_base numeric(30,8),        -- buyer-initiated volume
  taker_buy_quote numeric(30,8),
  source text,                         -- 'archive' | 'rest' | 'ws'
  inserted_at timestamptz default now(),
  primary key (symbol, open_time)
)
-- TimescaleDB hypertable, chunk = 7 days, compression after 7 days,
-- segmentby symbol, orderby open_time desc.

book_ticker(
  symbol text, ts timestamptz,
  bid_price numeric(20,8), bid_qty numeric(30,8),
  ask_price numeric(20,8), ask_qty numeric(30,8),
  spread numeric(20,8),                -- ask - bid
  primary key (symbol, ts)
)  -- hypertable, chunk = 1 day

ingestion_runs(
  id bigserial primary key,
  component text,          -- 'backfill' | 'live' | 'rest_gapfill'
  symbol text,
  period_start timestamptz, period_end timestamptz,
  status text,             -- 'running' | 'success' | 'failed'
  rows_written bigint,
  error text,
  started_at timestamptz, finished_at timestamptz
)

data_quality_reports(
  id bigserial primary key,
  symbol text, timeframe text,
  checked_from timestamptz, checked_to timestamptz,
  total_candles bigint, duplicates bigint, invalid bigint,
  missing bigint, completeness_pct numeric(8,6),
  verdict text,            -- 'PASS' | 'WARN' | 'FAIL'
  details jsonb,
  created_at timestamptz default now()
)
```

**Higher timeframes** are TimescaleDB continuous aggregates over
`candles_1m` (`candles_5m`, `candles_15m`, `candles_1h`, `candles_4h`,
`candles_1d`), each aggregating: `first(open)`, `max(high)`, `min(low)`,
`last(close)`, `sum(volume)`, `sum(quote_volume)`, `sum(trade_count)`,
`sum(taker_buy_base)`, `sum(taker_buy_quote)`. Refresh policies keep them
current. This guarantees higher timeframes can never disagree with the 1m
truth. All timestamps are UTC, bucket-aligned to Binance's own boundaries.

## 6. Error handling and failure modes

| Failure | Behaviour |
|---|---|
| Binance HTTP error / timeout | Retry with exponential backoff and jitter, capped attempts, then mark the run failed |
| HTTP 429 / 418 (rate limit / ban) | Respect `Retry-After`, pause the component, log loudly; never hammer the endpoint |
| Checksum mismatch on an archive | Discard the file, mark that month failed, retry on next run; never load unverified data |
| WebSocket disconnect | Backoff reconnect, REST-backfill the gap, log the gap, then resume |
| Database unavailable | Live collector buffers in memory with a bounded queue and retries; on overflow it logs an explicit data-loss warning rather than failing silently |
| Duplicate / out-of-order WS messages | Idempotent upsert keyed on (symbol, open_time) makes them harmless |
| Stale feed (no candle for N minutes) | Heartbeat check surfaces a staleness warning in `tb status` |
| Process killed mid-load | Staging-table load plus `ingestion_runs` means a rerun resumes cleanly |

Principle: **fail safe and visibly.** The system prefers to stop and report
rather than to continue with data it cannot vouch for.

## 7. Testing strategy

- **Unit tests, no network**: archive CSV parsing, WebSocket message parsing,
  gap detection, quality rules, backoff logic — all against small committed
  fixture files.
- **Bad-data tests**: fixtures containing duplicates, `high < low`, negative
  volume, a missing hour, and out-of-order rows; each must be caught and must
  produce the correct verdict. `assert_trainable` must raise on FAIL.
- **Aggregation correctness**: load a known 1m fixture, then assert the
  database-built 1h candles equal 1h candles computed independently in pandas.
- **Idempotency**: loading the same month twice yields identical row counts
  and no duplicates.
- **Integration (opt-in, marked `@pytest.mark.network`, skipped by default)**:
  fetch a small real window from Binance and load it end to end.
- Tests run against a disposable Timescale container; CI-ready.

## 8. Project layout (this sub-project)

Repository root is `D:\Trading` (already a git repository); paths below are
relative to it.

```text
  docker-compose.yml          # timescaledb service + volume
  pyproject.toml              # uv / Python 3.12, deps, tb entry point
  .env.example
  data/
    config.py
    storage/    (schema.sql, migrations/, repository.py)
    collectors/ (backfill.py, live.py, binance_rest.py, binance_ws.py)
    quality/    (checks.py, report.py)
    cli.py
  tests/
    fixtures/
    unit/
    integration/
  docs/superpowers/specs/
```

Folders for later sub-projects (features, models, backtesting, signals, risk,
portfolio, bots, execution, journal, assistant, api, frontend) are created by
those sub-projects, not now.

## 9. Open items deliberately deferred

- Live order-book depth recorder (needs months of data before it is useful).
- Additional symbols beyond the initial four (adding one is a config change).
- Binance account and bot data ingestion — sub-project 6, needs API keys.
- Parquet export for fast ML training — sub-project 3, if training I/O
  becomes a bottleneck.
