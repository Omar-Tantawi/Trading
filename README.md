# AI Trading Buddy — Data, Market Features, Prediction Models and Dashboard

Verified Binance market-data warehouse (sub-project 1), the market
features computed from it (sub-project 2a), prediction models measured
on them (sub-project 3) and a small read-only dashboard (sub-project 3.5); see `docs/superpowers/specs/`
for the designs and `docs/superpowers/plans/` for the implementation plans.

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
   `.env.example` and `.env` already point at `127.0.0.1:5433`; if you
   connect with `psql` or another client directly, remember the `-p 5433`.
   Use `127.0.0.1`, not `localhost`: the container listens on IPv4 only, and
   on Windows `localhost` tries IPv6 `::1` first, where nothing answers, so
   the connection hangs.

   **Network and password:** the port is published on `127.0.0.1` only,
   so the database is not reachable from other machines. The password
   defaults to `tb_local_dev`, which is fine on a single-user PC.

   **On a VPS, set a new password before the first start.** Docker-published
   ports bypass `ufw`, and Postgres fixes the password when the volume is
   first initialised; changing it in compose later does nothing. So, before
   the very first `docker compose up -d`, add `POSTGRES_PASSWORD=<strong
   value>` to `.env` and use the same value in `DATABASE_URL` and
   `TEST_DATABASE_URL`.

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

```bash
.venv\Scripts\tb.exe features build
```

```bash
.venv\Scripts\tb.exe analyze BTCUSDT
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

### Higher timeframes (`tb db refresh-aggregates`)

5m/15m/1h/4h/1d are TimescaleDB continuous aggregates over `candles_1m`.
Their scheduled refresh policies only reach back 3 days (5m) to 365 days
(1d), so older history reaches them only through an explicit refresh.
`tb backfill` does that automatically for every range it writes. Run it by
hand for history loaded before that existed, or after a backfill was
interrupted (Ctrl+C):

```bash
.venv\Scripts\tb.exe db refresh-aggregates
.venv\Scripts\tb.exe db refresh-aggregates --from 2024-01-01
```

`tb quality --timeframe 1h` (or any non-1m timeframe) takes its expected
range from `candles_1m`, so an aggregate that is missing history FAILs.

## Market features (sub-project 2a)

From the stored candles, `tb features build` computes a set of market
features (returns, EMAs, RSI, MACD, ATR, ADX, swing structure, volatility
and volume measures, regime labels) for 5m, 15m, 1h, 4h and 1d, and stores
them in the `features_*` tables. `tb analyze` shows the current state in
plain language.

```bash
.venv\Scripts\tb.exe features build
.venv\Scripts\tb.exe features build --symbol BTCUSDT --timeframe 1h
.venv\Scripts\tb.exe features build --rebuild
.venv\Scripts\tb.exe analyze BTCUSDT
.venv\Scripts\tb.exe analyze BTCUSDT --no-build
```

- **Built on demand.** Features exist only after you run `tb features build`
  (or `tb analyze`, which builds first). Nothing keeps them fresh yet; the
  live collector does not update them.
- **Incremental.** A build only computes the bars that are new since the
  last one, so re-running it is cheap and always safe. Only bars whose whole
  time span is already covered by stored 1m candles are built.
- **`--rebuild` recomputes everything from full history.** Run it after
  repairing holes in the candle data (for example by re-running
  `tb backfill`): bars that were already built over a hole are not
  recomputed automatically.
- **Quality gate.** A symbol whose 1m data fails `tb quality` is skipped,
  and the command exits non-zero. Each symbol and timeframe is isolated: one
  failure does not stop the others.
- **STALE.** `tb analyze` marks a timeframe **STALE** when its newest
  feature bar is older than 2 x the timeframe plus 5 minutes (for 1h: more
  than 2 hours 5 minutes). It means the features have not been built
  recently, not that anything is wrong. Run `tb features build`.
- `tb analyze` describes the market on each timeframe and counts how many
  are bullish, bearish or sideways. It never gives trading advice.
- `--no-build` prints what is already stored, without building. `SYMBOL` is
  case-insensitive. A symbol with no stored 1m candles exits with an error.
- When its build fails or is skipped (for example by the quality gate),
  `tb analyze` still prints the stored state, says it may be out of date,
  and exits 1.
- **Timing of a feature row.** A row keyed `open_time = t` describes the bar
  that opens at `t` and **closes** at `t + step` (for 1h, `t + 1 hour`): it
  uses that bar's close, so it is only known at `t + step`. Anything that
  joins features to later outcomes (sub-project 3's labels) must line up
  with `t + step`, not `t`, or it will use the future.

## Prediction models (sub-project 3)

Models that estimate, for each coin, the probability that the price ends
**up**, **down** or **flat** after 1, 4 or 24 hours. "Up" means the close
rises by more than half the coin's normal hourly range (ATR) scaled to the
horizon; "flat" is anything in between. These are **measurements, not
advice**: nothing here says to buy or sell, and "flat" is a normal outcome.

```bash
.venv\Scripts\tb.exe features build
.venv\Scripts\tb.exe ml evaluate
.venv\Scripts\tb.exe ml evaluate --horizon 4
.venv\Scripts\tb.exe ml runs
.venv\Scripts\tb.exe ml report 12
.venv\Scripts\tb.exe ml diagnose 12
.venv\Scripts\tb.exe ml train
.venv\Scripts\tb.exe predict BTCUSDT
```

- **`tb ml evaluate`** tests every model walk-forward: train on the past,
  predict the next six months it has never seen, move forward, repeat
  (from 2020). It prints a report and stores every prediction in the
  database (`ml_runs`, `ml_predictions`). It needs current features: if
  they are old it says "run `tb features build` first" and exits 1. A
  1m quality FAIL also stops it.
- **Skill** compares a model with simply predicting how often each outcome
  happened in the training data (the base rate). +1.0% means 1% lower log
  loss. The 95% interval shows how sure that is; if it includes 0, the model
  has not shown it knows anything the base rate does not.
- **`tb ml diagnose RUN_ID`** splits a run's skill in two: *size* (does
  the model know whether the price will move at all?) and *direction* (when
  it did move, does the model know which way?). It reads the stored
  predictions; nothing is retrained.
- **The holdout.** Data from 2025-10-01 on is kept out of every normal
  evaluation, so there is one final, honest test. `tb ml evaluate
  --holdout` uses it, and every use is counted: from the second time on, the
  report warns that the result is no longer unbiased. Use it once, at the
  end, deliberately.
- **`tb ml train`** fits the logistic regression and XGBoost on all history
  and saves them in `models\` (not committed to git). **`tb predict
  SYMBOL`** builds the newest features, then prints each model's
  probabilities with the skill it showed on unseen data next to them.
- Stored runs are never deleted automatically. To remove one (and its
  predictions): `DELETE FROM ml_runs WHERE run_id = 12;`.

## Dashboard (sub-project 3.5)

```bash
.venv\Scripts\tb.exe dashboard
```

Then open **http://127.0.0.1:8050** in your browser. Stop it with Ctrl+C.
`--port 8060` uses another port.

- **Chart:** candles for any coin and timeframe, with EMA 200 and RSI 14,
  and the newest bar's trend and volatility regime. Times are UTC
  (Binance shows your local time).
- **Models:** every `tb ml evaluate` run: the report, skill per test period,
  calibration charts, and the size-vs-direction split on request.
- **Data health:** newest candle per coin (STALE when the live collector is
  not running), newest feature bar and the latest quality verdict.
- **Read-only and local.** It only reads the database, listens on this PC
  only (127.0.0.1), and loads nothing from the internet (the TradingView
  Lightweight Charts library, Apache-2.0, is stored in
  `dashboard/static/vendor/`). The page shows what is stored: run
  `tb backfill` and `tb features build` first for fresh charts.

## Facts worth knowing

- All timestamps are UTC. All prices are `Decimal`, never `float`.
- Only **closed** candles are stored. A forming candle is never written.
- `candles_1m` is the only source of truth; 5m/15m/1h/4h/1d are derived
  from it by TimescaleDB (see `tb db refresh-aggregates` above).
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
