# AI Trading Buddy — Data Foundation

Verified Binance market-data warehouse. Sub-project 1 of the platform;
see `docs/superpowers/specs/` for the design and `docs/superpowers/plans/`
for the implementation plan.

## First-time setup

1. Install Docker Desktop and confirm `docker info` works.
2. Copy `.env.example` to `.env`.
3. Start the database:

   ```bash
   docker compose up -d
   ```

   **Port note:** the database listens on host port **5433**, not the
   Postgres default of 5432. This machine also runs a native PostgreSQL 18
   Windows service bound to 5432, so `docker-compose.yml` remaps the
   `tb-timescaledb` container's 5432 to host 5433 to avoid a collision.
   `.env.example` and `.env` already point at `localhost:5433`; if you
   connect with `psql` or another client directly, remember the `-p 5433`.

4. Create the virtual environment and install:

   ```bash
   py -3.13 -m venv .venv
   ```

   ```bash
   .venv\Scripts\python.exe -m pip install -e ".[dev]"
   ```

5. Apply migrations:

   ```bash
   .venv\Scripts\tb.exe db upgrade
   ```

## Daily use

```bash
.venv\Scripts\tb.exe symbols sync
```

```bash
.venv\Scripts\tb.exe backfill
```

```bash
.venv\Scripts\tb.exe live
```

```bash
.venv\Scripts\tb.exe quality
```

```bash
.venv\Scripts\tb.exe status
```

The first `backfill` downloads years of 1-minute candles for four symbols
and takes a while. It is resumable: stop it with Ctrl+C and re-run.

### `tb backfill` options

```bash
.venv\Scripts\tb.exe backfill --symbol SOLUSDT
.venv\Scripts\tb.exe backfill --symbol SOLUSDT --from 2024-06-01
```

- `--symbol` restricts the run to one symbol; omitted, every configured
  symbol is backfilled.
- `--from YYYY-MM-DD` (UTC) moves the start of the monthly-archive loop
  forward to `max(--from, the symbol's own listing date)`, instead of
  always starting from the listing date. Useful for topping up a narrower
  window quickly (e.g. smoke-testing the pipeline) without waiting for
  full history. Omit it to backfill from listing.
- Each symbol's backfill is isolated: if one symbol fails, the others
  still run. The command prints and logs each failure and exits non-zero
  if any symbol failed, so a partial failure is never silent.

## Facts worth knowing

- All timestamps are UTC. All prices are `Decimal`, never `float`.
- Only **closed** candles are stored. A forming candle is never written.
- `candles_1m` is the only source of truth; 5m/15m/1h/4h/1d are derived by
  TimescaleDB and cannot disagree with it.
- Source precedence is `archive` > `rest` > `ws`; a lower source never
  overwrites a higher one.
- `assert_trainable()` in `data/quality/checks.py` is the gate every future
  training run must call first.

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

Network tests are skipped by default; run them deliberately with
`-m network`. One such test (`tests/integration/test_network_e2e.py`)
makes real HTTP requests to Binance's public market-data endpoints
(`api.binance.com`); no API key or account is needed.
