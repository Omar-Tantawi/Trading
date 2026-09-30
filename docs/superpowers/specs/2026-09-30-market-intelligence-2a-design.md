# Design: AI Trading Buddy — Sub-project 2a: Market Intelligence Features

Date: 2026-09-30
Status: Approved for planning
Milestone: 2a of the programme (see the data-foundation spec §2)
Source: vision document `docs/vision/2026-09-22-original-vision.md` §6, §9–14, §17, §32, §106

## 1. Purpose

Turn the verified candle history from sub-project 1 into
**machine-understandable market information**: a stored table of features
per coin and timeframe for the ML models of sub-project 3 to train on, and a
readable "current state" of each market that the user, the dashboard (3.5)
and the LLM assistant (8) can read.

Nothing in 2a predicts, recommends or trades. Indicators are information: an
RSI of 29 is a fact, not a BUY.

### Success criteria

1. `tb features build` computes every feature in §4 for BTCUSDT, ETHUSDT,
   SOLUSDT and BNBUSDT on 5m, 15m, 1h, 4h and 1d, and stores one row per
   closed bar in the database.
2. No feature uses information from after its own bar: a test proves this for
   every column (§7.2).
3. Rebuilding incrementally gives the same values as rebuilding from scratch
   (§5.3).
4. `tb analyze SYMBOL` prints the current state across all five timeframes in
   plain words, with a structured form the LLM can reuse later.
5. The N1 false alarm carried over from sub-project 1 is fixed (§8).

### Agreed with the user (2026-09-30)

- Sub-project 2 is split. **2a** (this spec): price features, indicators,
  trend, volatility, volume, candle shapes, rules-based regime, the N1 fix.
  **2b** (later): support/resistance, Market Profile
  (`docs/research/2026-09-23-market-profile-hypotheses.md`), chart patterns.
- Output: readable report **and** stored feature tables.
- Timeframes: 5m, 15m, 1h, 4h, 1d. No 1m features (too noisy; can be added
  later without redesign).
- Approach A: compute in Python (pandas/NumPy), store in TimescaleDB, in the
  same database as the candles.
- Indicator formulas are our own code, not a library, verified against
  hand-computed values in tests.

### Non-goals

- Support/resistance, Market Profile, chart patterns (2b).
- A trained regime model (sub-project 3 may replace the rules-based one).
- Features that update themselves continuously as candles arrive
  (sub-project 5, live signals). 2a builds on demand.
- Charts or any UI (sub-project 3.5).
- Anything that uses order-book or trade-level data.

## 2. Architecture

A new top-level package `features/`, beside `data/`, following the vision's
§103 layout:

| Module | Responsibility | Touches DB |
|---|---|---|
| `features/frame.py` | Load closed bars for (symbol, timeframe, range) as a pandas DataFrame. The **only** place Decimal becomes float (cast to `float8` in SQL). | yes |
| `features/price.py` | §4.1 | no |
| `features/indicators.py` | §4.2 | no |
| `features/trend.py` | §4.3 | no |
| `features/volatility.py` | §4.4 | no |
| `features/volume.py` | §4.5 | no |
| `features/candles.py` | §4.6 | no |
| `features/regime.py` | §4.7 | no |
| `features/pipeline.py` | Runs the modules in order into one wide DataFrame; applies warm-up blanking; owns `FEATURE_COLUMNS`, `WARMUP` and `FEATURE_SET` | no |
| `features/store.py` | Idempotent write and read of feature rows | yes |
| `features/build.py` | Orchestrates one (symbol, timeframe) build: quality gate, aggregate refresh, load window, compute, write | yes |
| `features/summary.py` | The newest stored rows → `MarketState` (structured) → rendered text | yes (read) |

The calculation modules take a DataFrame of bars (index `open_time`, UTC;
columns `open, high, low, close, volume, taker_buy_base`, all float) and return a DataFrame of feature columns on the same index. They
never read the database or the clock. This keeps every formula testable with
a few hand-made bars and no Docker.

The CLI (`data/cli.py`) gains `tb features build` and `tb analyze` (§6).
pandas and NumPy move from dev dependencies to runtime dependencies, and
`features*` is added to the setuptools package list.

## 3. Global rules

1. **No look-ahead.** A feature at bar *t* uses only bars whose `open_time`
   ≤ *t*, and only information those bars had once closed. A swing point is
   known only once its confirming bars have closed (§4.3).
2. **Scale-free for models.** Anything a model compares across coins or
   years is a ratio, percentage, log return or index (0–100 or 0–1). Price-
   level values (EMA, SMA, ATR, Bollinger bands, swing prices) are also
   stored, for the dashboard to draw, and are not meant as model inputs.
3. **Blank, never zero.** Until a feature has enough history (its warm-up,
   §4), or when its formula divides by zero, it is NULL. A NULL feature never
   blanks any other feature in the row.
4. **Closed, complete bars only.** A bar is built only when its whole time
   span is covered by stored 1m data (§5.2).
5. **Floats for features, Decimal for prices.** Stored candles stay exact
   Decimal (sub-project 1 rule). Features are float64 (`double precision`).
6. **Numbers are information.** No module outputs, and the summary never
   prints, BUY/SELL/advice words.

## 4. Feature catalogue

Notation: *n-bar* means the last n bars **including** the current one, unless
a feature says "previous n bars", which **excludes** the current bar. `ln` is
the natural log. Warm-up is counted in bars from the first bar of the
symbol's history on that timeframe; features are NULL for bar indices
0 … warm-up − 1.

### 4.1 Price (vision §9)

| Column | Definition | Warm-up |
|---|---|---|
| `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24` | ln(close_t / close_{t−n}) | n |
| `dist_high_20`, `dist_high_100` | close_t / max(high over n-bar) − 1 (≤ 0) | n − 1 |
| `dist_low_20`, `dist_low_100` | close_t / min(low over n-bar) − 1 (≥ 0) | n − 1 |
| `accel_6` | ret_6(t) − ret_6(t−6) | 12 |
| `range_ratio_20` | (high − low)_t / mean(high − low over previous 20 bars); NULL if that mean is 0 | 20 |

*Ruling:* the vision's "rolling return" is the n-bar returns above; it is not
stored twice.

### 4.2 Indicators (vision §10)

EMA: α = 2 / (n + 1), seeded with the first close, `ema_t = α·close_t +
(1 − α)·ema_{t−1}` (pandas `ewm(span=n, adjust=False)`). Wilder smoothing: the
same recursion with α = 1 / n. Recursive features have a warm-up of 3 × the
longest period in their chain, so the seed's influence has decayed.

| Column | Definition | Warm-up |
|---|---|---|
| `ema_9`, `ema_20`, `ema_50`, `ema_100`, `ema_200` | EMA of close (price level) | 3n |
| `dist_ema_9` … `dist_ema_200` | close / ema_n − 1 | 3n |
| `sma_20`, `sma_50`, `sma_200` | mean of close over n-bar (price level) | n − 1 |
| `dist_sma_20`, `dist_sma_50`, `dist_sma_200` | close / sma_n − 1 | n − 1 |
| `rsi_14` | Wilder RSI: gains/losses of close-to-close change, each Wilder-smoothed (n = 14); RSI = 100 − 100 / (1 + avg_gain / avg_loss); 100 if avg_loss = 0 and avg_gain > 0; 50 if both are 0 | 42 |
| `macd_pct`, `macd_signal_pct`, `macd_hist_pct` | macd = ema_12 − ema_26; signal = EMA₉ of macd; hist = macd − signal; each divided by close | macd 78, signal and hist 105 |
| `atr_14` | Wilder-smoothed true range; TR = max(high − low, \|high − close_{t−1}\|, \|low − close_{t−1}\|), and TR at bar 0 = high − low (price level) | 42 |
| `atr_pct` | atr_14 / close | 42 |
| `plus_di_14`, `minus_di_14` | up = high_t − high_{t−1}, down = low_{t−1} − low_t; +DM = up if up > down and up > 0 else 0; −DM mirror; each Wilder-smoothed; DI = 100 × smoothed DM / atr_14 (NULL if atr_14 = 0) | 42 |
| `adx_14` | DX = 100 × \|+DI − −DI\| / (+DI + −DI) (0 if the sum is 0); ADX = Wilder-smoothed DX | 84 |
| `bb_upper`, `bb_lower` | sma_20 ± 2 × population std (ddof = 0) of close over 20-bar (price level) | 19 |
| `bb_pct_b` | (close − bb_lower) / (bb_upper − bb_lower); NULL if the width is 0 | 19 |
| `bb_width` | (bb_upper − bb_lower) / sma_20 | 19 |
| `stoch_k_14` | 100 × (close − min low 14-bar) / (max high 14-bar − min low 14-bar); NULL if the range is 0 | 13 |
| `stoch_d_3` | mean of stoch_k_14 over 3-bar | 15 |
| `roc_10` | close_t / close_{t−10} − 1 | 10 |

### 4.3 Trend (vision §11)

**Swing points.** Bar *i* is a swing high if its high is strictly greater than
the highs of the 3 bars before it and the 3 bars after it (`SWING_K = 3`).
Swing lows mirror this on lows. A swing at *i* is **confirmed at bar i + 3**
and is invisible to every feature before that bar. This is the main
look-ahead trap in chart analysis; §7.2 tests it.

At each bar *t*, using swings confirmed at or before *t*: H2 and H1 are the
latest and previous swing-high prices; L2 and L1 the same for swing lows.

| Column | Definition | Warm-up |
|---|---|---|
| `swing_high` | H2 (price level); NULL until one swing high is confirmed | — |
| `swing_low` | L2 (price level); NULL until one swing low is confirmed | — |
| `dist_swing_high` | close / H2 − 1 | — |
| `dist_swing_low` | close / L2 − 1 | — |
| `swing_high_dir` | +1 if H2 > H1 (higher high), −1 if H2 < H1 (lower high), 0 if equal; NULL until two are confirmed | — |
| `swing_low_dir` | +1 if L2 > L1 (higher low), −1 if lower low, 0 if equal; NULL until two are confirmed | — |
| `structure` | +1 if both dirs are +1 (up: HH + HL); −1 if both are −1 (down: LH + LL); 0 otherwise; NULL if either is NULL | — |
| `ema_slope_20`, `ema_slope_50` | ema_n(t) / ema_n(t−5) − 1 | 3n + 5 |
| `trend_duration` | signed bar count of the current EMA 20 vs EMA 50 relation: +k if ema_20 > ema_50 for the last k bars including t, −k if below, 0 if equal | 150 |
| `trend_accel` | ema_slope_20(t) − ema_slope_20(t−5) | 70 |
| `bos` | break of structure at bar t: +1 if close_t > H2 and close_{t−1} ≤ H2 (the same confirmed H2); −1 if close_t < L2 and close_{t−1} ≥ L2; else 0 | — |
| `bars_since_bos` | bars since the most recent non-zero `bos`, signed by its direction (+k or −k; 0 on the bar itself); NULL if none yet | — |

Trend *strength* is `adx_14`; it is not stored twice. "—" means the feature
has no fixed warm-up: it is NULL until its inputs exist.

### 4.4 Volatility (vision §12)

| Column | Definition | Warm-up |
|---|---|---|
| `ret_std_20` | sample std (ddof = 1) of ret_1 over 20-bar | 20 |
| `hist_vol_20` | ret_std_20 × √(bars per year); bars per year = 525,600 / minutes per bar (crypto trades 24/7) | 20 |
| `range_pct` | (high − low) / close | 0 |
| `vol_pct_365d` | percentile (0–100) of atr_pct at t among all non-NULL atr_pct values with open_time in (t − 365 days, t]: 100 × count(values ≤ current) / count(values). NULL while the earliest non-NULL atr_pct is less than 30 days (`VOL_PCT_MIN_HISTORY`) older than t | time-based: see definition |

`bb_width` and `atr_pct` (§4.2) are also volatility features; they are not
stored twice.

### 4.5 Volume (vision §13)

| Column | Definition | Warm-up |
|---|---|---|
| `volume_ratio_20` | volume_t / mean(volume over previous 20 bars); NULL if that mean is 0 | 20 |
| `volume_change` | ln(volume_t / volume_{t−1}); NULL if either is 0 | 1 |
| `volume_z_20` | (volume_t − mean) / std (ddof = 1) over previous 20 bars; NULL if std = 0 | 20 |
| `buy_volume` | taker_buy_base (base-asset units) | 0 |
| `sell_volume` | volume − taker_buy_base | 0 |
| `buy_share` | taker_buy_base / volume (0–1); NULL if volume = 0 | 0 |

*Ruling:* the vision's "buy/sell ratio" is `buy_share / (1 − buy_share)`,
derivable, so not stored. `buy_share` is bounded, which suits models better.

### 4.6 Candle shape (vision §14)

With range = high − low; every fraction is NULL when range = 0.

| Column | Definition | Warm-up |
|---|---|---|
| `body_frac` | (close − open) / range (signed, −1 … 1) | 0 |
| `upper_wick_frac` | (high − max(open, close)) / range | 0 |
| `lower_wick_frac` | (min(open, close) − low) / range | 0 |
| `open_pos` | (open − low) / range (0–1) | 0 |
| `close_pos` | (close − low) / range (0–1) | 0 |
| `body_atr` | (close − open) / atr_14 | 42 |
| `size_atr` | range / atr_14 | 42 |

Invariant: \|body_frac\| + upper_wick_frac + lower_wick_frac = 1.

### 4.7 Rules-based regime (vision §17)

Two independent labels, because a market can be "sideways" and "extremely
volatile" at once. The vision's eight regimes are the combinations.

`trend_regime` (text), from adx_14 and the DIs:

| Condition | Label |
|---|---|
| adx_14 < 20 (`ADX_TREND_MIN`), or +DI = −DI | `sideways` |
| 20 ≤ adx_14 < 25 (`ADX_STRONG_MIN`) | `weak_bullish` if +DI > −DI, else `weak_bearish` |
| adx_14 ≥ 25 | `strong_bullish` if +DI > −DI, else `strong_bearish` |

`volatility_regime` (text), from vol_pct_365d:

| Condition | Label |
|---|---|
| < 20 | `low` |
| 20 – < 80 | `normal` |
| 80 – < 95 | `high` |
| ≥ 95 | `extreme` |

Both are NULL while their inputs are NULL. *Ruling:* the thresholds are the
conventional ADX readings (20 = trend starting, 25 = established trend) and
percentile bands. They are named constants, and sub-project 3 measures
whether they carry signal. `normal` is added because the vision lists only
low, high and extreme, and most of the time is none of those.

## 5. Storage and build

### 5.1 Tables

Migration `004_features.sql` creates five hypertables, `features_5m`,
`features_15m`, `features_1h`, `features_4h`, `features_1d`, with identical
columns:

- `symbol text NOT NULL REFERENCES symbols(symbol)`,
  `open_time timestamptz NOT NULL`, `PRIMARY KEY (symbol, open_time)`
- `feature_set smallint NOT NULL`: the `FEATURE_SET` version that produced
  the row
- every column of §4, in catalogue order: `double precision` for numbers,
  `smallint` for `swing_high_dir`, `swing_low_dir`, `structure` and `bos`,
  `integer` for `trend_duration` and `bars_since_bos`, `text` for the two
  regimes
- `computed_at timestamptz NOT NULL DEFAULT now()`

A compression policy segments by `symbol`, orders by `open_time DESC`, and
compresses after 30 days.

`FEATURE_COLUMNS` in `pipeline.py` is the single list of feature columns. A
DB test asserts the table columns equal it exactly, in order.

### 5.2 Which bars are built

For (symbol, timeframe, step):

1. **Quality gate.** Run the 1m quality check for the symbol. FAIL → skip this
   symbol, report it, exit non-zero at the end. WARN → proceed and log it.
2. **Completeness cutoff.** `last_1m` = the newest stored 1m `open_time`.
   Only buckets with `open_time + step ≤ last_1m + 1 minute` are eligible,
   i.e. buckets fully covered by stored 1m data. The bucket still forming,
   and one whose last minutes the live collector hasn't written yet, are
   never built.
3. **Refresh.** Call the existing `refresh_aggregates` over the range being
   built, up to the cutoff, so the continuous aggregate holds every eligible
   bucket. The build never depends on the hourly refresh policy.

*Ruling:* a bucket partly inside a genuine exchange outage is built from the
minutes that exist. Features are computed over the bars present, in order.
The quality gate already blocks unexplained holes longer than an hour, and
exchange outages are rare. Cost if wrong: a handful of bars in years of
history carry indicators computed across an outage.

### 5.3 Incremental builds

`last_built` = the newest stored `open_time` for (symbol, timeframe).

- **No rows yet, `--rebuild`, or any stored row with `feature_set` ≠
  `FEATURE_SET`:** full build. Load the whole history, compute, and upsert
  every row.
- **Otherwise:** load from `last_built − LOOKBACK` to the cutoff, compute,
  and upsert only rows with `open_time > last_built`.
  `LOOKBACK = max(2,100 bars, 400 days)` is chosen so that every recursive
  feature's seed has decayed below 1e-9 relative (EMA 200 needs about 2,070
  bars), the 365-day percentile window is complete, and swing and duration
  counters see their history. On 1d this covers the whole history.

**Contract:** an incremental build yields the same values as a full build:
exactly for integer and text columns, and for floats within
`numpy.isclose(a, b, rtol=1e-9, atol=1e-9)`, with NULL matching NULL.
§7.3 tests it.

Writes use the sub-project 1 pattern: COPY into a temporary staging table,
then `INSERT … ON CONFLICT (symbol, open_time) DO UPDATE`. Re-running a build
is always safe. Feature rows are derived data; the build never deletes them.

### 5.4 Performance targets

- Full build of all 4 symbols × 5 timeframes over full history: under 30
  minutes on this PC.
- `tb analyze` on data built within the last day: under 30 seconds.

## 6. Commands and the readable summary

### 6.1 `tb features build [--symbol S] [--timeframe TF] [--rebuild]`

Defaults: all configured symbols × all five timeframes. Each (symbol,
timeframe) is isolated like `tb backfill`: an error there rolls back and is
reported, and the others continue. Prints one line per pair: rows written,
the range built, and the time taken. Exits non-zero if any pair failed or was
skipped by the quality gate.

### 6.2 `tb analyze SYMBOL [--no-build]`

Runs an incremental build for SYMBOL on all five timeframes (skipped with
`--no-build`), then prints its current state. Per timeframe, from the newest
stored row:

- the bar's close time and its age; **STALE** if older than 2 × step plus 5
  minutes, since nothing keeps features fresh automatically yet
- trend regime, ADX, structure (up / down / mixed), trend duration in bars
- RSI
- volatility regime and its percentile
- volume ratio and buy share
- distance to EMA 200

Then one agreement line counting how many timeframes are bullish, bearish and
sideways by trend regime. This is how the vision's §6 idea (timeframes may
disagree) shows up.

`summary.py` builds a `MarketState` dataclass (one `TimeframeState` per
timeframe) that serialises to a plain dict: the structured form sub-project 8
will hand to the LLM. `render()` turns it into the printed report. Values are
rounded for display only; the structure keeps full precision.

The summary describes; it never recommends. A test asserts the rendered text
contains none of: buy, sell, long, short, enter, exit (as whole words).

## 7. Testing

### 7.1 Formulas (unit, no DB)

Each module is tested on a small hand-made bar series, with expected numbers
written into the test and derived by hand in a comment. Invariants too:

- EMA/SMA of a constant series equal that constant
- RSI of a strictly rising series is 100, of a falling one 0
- ATR of bars with constant range and no gaps equals that range
- ADX on a steady one-way trend rises above 25
- stochastic and percentiles stay within [0, 100]; `buy_share` within [0, 1]
- the candle-fraction invariant of §4.6
- every regime label for inputs on both sides of each threshold

### 7.2 No look-ahead (the most important test)

Prefix invariance, over **every** column in `FEATURE_COLUMNS`: on a synthetic
random-walk series of at least 3,000 bars, compute features on
bars[0 : k] and on the full series; row k − 1 must match (same tolerance as
§5.3), for at least 50 values of k spread across the series. A separate test
builds a known swing high at bar i and asserts `swing_high` is still NULL, or
still the previous value, at bars i … i + 2, and becomes that high at
i + 3.

### 7.3 Incremental equals full

At the pipeline level (no DB): compute the full series; compute again from a
window starting `LOOKBACK` before a cut point; the rows after the cut must
match (§5.3 tolerance). At the DB level: build, add candles, build
incrementally; the result equals a `--rebuild`.

### 7.4 Warm-up

For every column, NULL exactly on indices below its declared warm-up and
non-NULL at the warm-up index on a series with no zero ranges or zero
volumes. Every column in `FEATURE_COLUMNS` has a `WARMUP` entry.

### 7.5 Database and CLI (`-m db`)

- table columns equal `FEATURE_COLUMNS`
- a build is idempotent (second run writes 0 rows and changes nothing)
- the completeness cutoff excludes a partly covered final bucket
- a quality FAIL skips the symbol and exits non-zero
- a `feature_set` mismatch triggers a full rebuild
- `tb analyze` prints every timeframe and marks STALE correctly
- the N1 tests (§8)

### 7.6 Real-data verification (after the user's full backfill)

1. `tb db upgrade`, then `tb features build`. All 20 pairs succeed; record
   the time taken against §5.4.
2. For each pair: feature rows = eligible aggregate buckets.
3. `tb analyze BTCUSDT` reads sensibly.
4. **User's own check:** open Binance's BTCUSDT 1h chart with RSI(14) and
   EMA(200) and compare the last closed bar with `tb analyze`. Binance uses
   the same Wilder RSI; small differences in EMA 200 come only from where
   the history starts.

## 8. The N1 fix (carried over from sub-project 1)

**Problem:** `run_quality_checks` on 5m–1d expects the newest complete bucket.
But the continuous-aggregate policies materialize only every
`schedule_interval` (5m: 5 minutes, 15m: 15 minutes, 1h/4h/1d: 1 hour; each
with `end_offset` 1 minute). So a bucket that just closed reads as a hole:
a false WARN on 5m–1h, and a false FAIL (exit 1) on 4h/1d, for up to an hour
after each close while `tb live` runs.

**Fix:** for timeframes other than 1m, the checked range ends at the last
bucket whose end ≤ now − (`schedule_interval` + `end_offset` + 5 minutes of
grace), and never later than the existing end derived from 1m data. The
lags live in one mapping, `POLICY_LAG`, beside `STEP`.

**Tests:**

- A DB test asserts `POLICY_LAG` matches the policies actually registered in
  `timescaledb_information.jobs`, so the two can never drift apart.
- A test with a clock just after a 4h bucket closes, and that bucket not yet
  materialized, gets PASS, not FAIL.
- A bucket missing well beyond the lag still counts as missing.
