# Market Intelligence 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A feature engine that turns stored candles into 75 stored features per bar (5m–1d, four symbols), plus `tb features build`, `tb analyze`, and the N1 quality fix.

**Architecture:** Pure calculation modules in a new `features/` package (DataFrame of bars in, DataFrame of features out, no DB, no clock). A pipeline composes them and blanks warm-up rows. `store.py` and `frame.py` are the only DB touch points. `build.py` orchestrates incremental builds from the continuous aggregates. `summary.py` turns the newest rows into a readable state.

**Tech Stack:** Python 3.13, pandas, NumPy, psycopg 3, TimescaleDB 2.30.1 (PostgreSQL 16), typer, rich, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-market-intelligence-2a-design.md`. **Binding.** Every feature's exact definition and warm-up is in spec §4. Implementers read the spec sections their task names; this plan does not repeat the formulas.

## Global Constraints

- **Python 3.13.3 via `.venv\Scripts\python.exe`.** The machine's `python` is 3.9; never use it.
- **Stored prices stay `Decimal`.** Features are float64 (`double precision`). The Decimal → float conversion happens only in `features/frame.py`, by casting to `float8` in SQL (spec §3.5).
- **Every timestamp is timezone-aware UTC.** DataFrame indexes are `DatetimeIndex` with tz UTC, named `open_time`.
- **No look-ahead** (spec §3.1): a feature at bar t uses only bars with `open_time` ≤ t. Swing points count only from their confirmation bar, i + 3.
- **Blank, never zero** (spec §3.3): missing, warm-up and divide-by-zero values are NaN in pandas and NULL in the database, never `NaN` stored as a float and never ±inf.
- **Calculation modules never touch the database or the clock.**
- **Timeframes:** `("5m", "15m", "1h", "4h", "1d")`. Symbols come from settings (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT).
- **Tests run offline by default.** DB tests are marked `pytestmark = pytest.mark.db` and use the `db_conn` / `autocommit_conn` fixtures in `tests/conftest.py` (test DB on 127.0.0.1:5433).
- **Commit after every task:** conventional messages (`feat:`, `fix:`, `test:`, `docs:`), explicit `git add <paths>` (never `-A` or `.`), never `.env`. End every commit message with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No new dependencies** beyond moving pandas and NumPy to runtime dependencies.

## Review Focus

1. **NaN written as a float.** PostgreSQL `double precision` accepts `'NaN'`, so a naive COPY stores NaN instead of NULL; the summary then prints "nan" and SQL `IS NULL` misses it. Pinned in Task 7 (`test_nan_is_stored_as_null`).
2. **±inf from division.** A flat bar (high = low) or zero volume must yield NULL, not inf. Pinned in Task 5 (`test_no_infinities_on_flat_bars_and_zero_volume`).
3. **Very short history.** A newly listed coin, or 1d with 20 bars, must build with NULLs and no exception; zero bars must give an empty frame with every column. Pinned in Task 5 (`test_short_and_empty_history`).
4. **`tb analyze` with nothing to show.** An unknown symbol, or one with no candles, must print one clear line and exit 1, with no traceback. Pinned in Task 9 (`test_analyze_unknown_symbol_exits_1`).
5. **Lowercase input.** `tb analyze btcusdt` must work like `BTCUSDT`, matching `tb quality`. Pinned in Task 9 (`test_analyze_accepts_lowercase`).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | pandas and numpy become runtime deps; package list `["data*", "features*"]` | 1 |
| `features/__init__.py` | empty | 1 |
| `features/indicators.py` | spec §4.2 | 1 |
| `features/price.py` | spec §4.1 | 2 |
| `features/volume.py` | spec §4.5 | 2 |
| `features/candles.py` | spec §4.6 | 2 |
| `features/trend.py` | spec §4.3 | 3 |
| `features/volatility.py` | spec §4.4 | 4 |
| `features/regime.py` | spec §4.7 | 4 |
| `features/pipeline.py` | composition, `FEATURE_COLUMNS`, `WARMUP`, `FEATURE_SET`, lookback | 5 |
| `data/quality/checks.py` | N1: `POLICY_LAG`, `now` parameter | 6 |
| `data/storage/migrations/004_features.sql` | five feature hypertables | 7 |
| `features/store.py` | feature row writes and reads | 7 |
| `features/frame.py` | bar loading, completeness cutoff | 8 |
| `features/build.py` | quality gate, refresh, incremental build | 8 |
| `features/summary.py` | `MarketState`, rendering | 9 |
| `data/cli.py` | `tb features build`, `tb analyze` | 9 |
| `README.md` | operator docs for the new commands | 9 |
| `tests/conftest.py` | `make_bars` and `bars_from` fixtures (Task 1); `assert_features_match` fixture (Task 5); feature tables added to the `db_conn` cleanup (Task 7) | 1, 5, 7 |

Test files: `tests/unit/test_features_<module>.py` for calculation modules; `tests/integration/test_features_<area>_db.py` for DB work.

---

### Task 1: Package scaffold, test bar factories, indicators

**Files:**
- Modify: `pyproject.toml`
- Create: `features/__init__.py`, `features/indicators.py`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_features_indicators.py`

**Interfaces:**
- Produces (tests, `tests/conftest.py` fixtures that each return a function):
  - `make_bars(n: int = 3000, *, step: timedelta = timedelta(hours=1), seed: int = 0, start: datetime = datetime(2022, 1, 1, tzinfo=timezone.utc)) -> pd.DataFrame`: a seeded random walk. close_t = close_{t−1} · exp(N(0, 0.01)), starting at 100; open_t = close_{t−1} (bar 0: 100); high = max(open, close) · (1 + |N(0, 0.003)|); low = min(open, close) · (1 − |N(0, 0.003)|); volume = exp(N(3, 0.5)); taker_buy_base = volume · U(0.3, 0.7). No flat bars, no zero volume.
  - `bars_from(*, highs, lows, closes, opens=None, volumes=None, taker=None, step=timedelta(hours=1), start=<same default>) -> pd.DataFrame`: explicit bars. opens defaults to closes, volumes to 1.0, and taker to half the volume.
  - Both return columns `open, high, low, close, volume, taker_buy_base` (float64) on a UTC `DatetimeIndex` named `open_time`, spaced by `step`.
- Produces (`features/indicators.py`), each returning a float64 Series on the input's index:
  - `ema(s: pd.Series, n: int) -> pd.Series`: `s.ewm(span=n, adjust=False).mean()`; leading NaNs are skipped and the first valid value seeds it
  - `wilder(s: pd.Series, n: int) -> pd.Series`: same with `alpha=1/n`
  - `rsi(close: pd.Series, n: int = 14) -> pd.Series`
  - `true_range(high, low, close) -> pd.Series`; `atr(high, low, close, n: int = 14) -> pd.Series`
  - `dmi(high, low, close, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]` returning (plus_di, minus_di, adx)
  - `macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]` returning raw (macd, signal, hist)
  - `bollinger(close, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series]` returning (upper, lower)
  - `stochastic(high, low, close, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]` returning (k, d)
  - `compute(bars: pd.DataFrame) -> pd.DataFrame`: exactly the 32 spec §4.2 columns, in spec order, **no warm-up blanking** (Task 5 does that)

- [ ] **Step 1: Move `pandas>=2.2` from the dev extra into `dependencies`, add `numpy>=2.0`, set `[tool.setuptools.packages.find] include = ["data*", "features*"]`, create an empty `features/__init__.py`, then run `.venv\Scripts\python.exe -m pip install -e ".[dev]"`.**

- [ ] **Step 2: Add the `make_bars` and `bars_from` fixtures to `tests/conftest.py` as specified in Interfaces.**

- [ ] **Step 3: Write the failing tests** in `tests/unit/test_features_indicators.py`. Hand-worked values:

```python
def test_ema_hand_values():            # n=3 → α=0.5
    assert ema(pd.Series([1., 2., 3., 4.]), 3).tolist() == [1, 1.5, 2.25, 3.125]

def test_wilder_hand_values():         # n=2 → α=0.5
    assert wilder(pd.Series([2., 4., 4.]), 2).tolist() == [2, 3, 3.5]

def test_rsi_hand_values():            # changes +1, −1, +2; n=2
    r = rsi(pd.Series([10., 11., 10., 12.]), 2)
    assert np.isnan(r.iloc[0]) and r.iloc[1] == 100 and r.iloc[2] == 50
    assert r.iloc[3] == pytest.approx(83.3333333, rel=1e-6)   # RS = 1.25 / 0.25

def test_rsi_extremes():               # rising → 100, falling → 0, flat → 50 (after bar 0)

def test_true_range_and_atr_hand_values(bars_from):
    b = bars_from(highs=[10, 12, 11], lows=[8, 9, 7], closes=[9, 11, 8])
    assert true_range(b.high, b.low, b.close).tolist() == [2, 3, 4]
    assert atr(b.high, b.low, b.close, 2).tolist() == [2, 2.5, 3.25]

def test_dmi_hand_values(bars_from):   # same bars, n=2
    plus, minus, adx = dmi(b.high, b.low, b.close, 2)
    # +DM = [nan, 2, 0], −DM = [nan, 0, 2]; smoothed +DM [nan, 2, 1], −DM [nan, 0, 1]
    assert plus.iloc[1] == pytest.approx(80.0) and minus.iloc[1] == 0
    assert plus.iloc[2] == pytest.approx(100 / 3.25) == minus.iloc[2]
    assert adx.iloc[1] == pytest.approx(100.0) and adx.iloc[2] == pytest.approx(50.0)

def test_macd_of_constant_series_is_zero():

def test_bollinger_hand_values():      # n=3, k=2 on [1, 2, 3]: sma 2, population std √(2/3)
    up, lo = bollinger(pd.Series([1., 2., 3.]), 3, 2.0)
    assert up.iloc[2] == pytest.approx(2 + 2 * (2 / 3) ** 0.5)
    assert lo.iloc[2] == pytest.approx(2 - 2 * (2 / 3) ** 0.5)

def test_stochastic_bounds_and_flat_window(make_bars, bars_from):   # within [0, 100]; flat window → NaN

def test_ema_and_sma_of_constant_equal_constant():

def test_adx_rises_above_25_on_steady_trend(bars_from):   # 100 bars, each high/low/close +1

def test_compute_columns_in_spec_order(make_bars):
    assert list(compute(make_bars(300)).columns) == [
        "ema_9", "ema_20", "ema_50", "ema_100", "ema_200",
        "dist_ema_9", "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200",
        "sma_20", "sma_50", "sma_200", "dist_sma_20", "dist_sma_50", "dist_sma_200",
        "rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct", "atr_14", "atr_pct",
        "plus_di_14", "minus_di_14", "adx_14", "bb_upper", "bb_lower", "bb_pct_b",
        "bb_width", "stoch_k_14", "stoch_d_3", "roc_10"]
```

- [ ] **Step 4: Run `.venv\Scripts\python.exe -m pytest tests/unit/test_features_indicators.py -q`.** Expected: fails, because the module doesn't exist.

- [ ] **Step 5: Implement `features/indicators.py` to spec §4.2.** Divide-by-zero results (DI with ATR = 0, `bb_pct_b` with zero width, stochastic with a flat window) are NaN, never inf. DX is 0 when +DI + −DI = 0.

- [ ] **Step 6: Run the Step 4 command.** Expected: all pass. Then run `.venv\Scripts\python.exe -m pytest tests/unit -q`: all pass.

- [ ] **Step 7: Commit** `pyproject.toml features/__init__.py features/indicators.py tests/conftest.py tests/unit/test_features_indicators.py` with message `feat: features package with hand-verified technical indicators`.

---

### Task 2: Price, volume and candle-shape features

**Files:**
- Create: `features/price.py`, `features/volume.py`, `features/candles.py`
- Test: `tests/unit/test_features_price_volume_candles.py`

**Interfaces:**
- Consumes: `make_bars`, `bars_from` fixtures (Task 1).
- Produces:
  - `price.log_return(close, n) -> Series`; `price.dist_high(close, high, n) -> Series`; `price.dist_low(close, low, n) -> Series`; `price.ratio_to_previous_mean(s: Series, n: int) -> Series` = s_t / mean(s over the previous n bars, excluding t), NaN when the mean is 0. **Shared by `range_ratio_20` and `volume_ratio_20`.** `volume.py` imports it from `price.py`.
  - `price.compute(bars) -> DataFrame`: the 11 spec §4.1 columns in order `ret_1, ret_3, ret_6, ret_12, ret_24, dist_high_20, dist_high_100, dist_low_20, dist_low_100, accel_6, range_ratio_20`
  - `volume.compute(bars) -> DataFrame`: the 6 spec §4.5 columns in order `volume_ratio_20, volume_change, volume_z_20, buy_volume, sell_volume, buy_share`
  - `candles.compute(bars, atr_14: Series) -> DataFrame`: the 7 spec §4.6 columns in order `body_frac, upper_wick_frac, lower_wick_frac, open_pos, close_pos, body_atr, size_atr`

- [ ] **Step 1: Write the failing tests:**

```python
def test_log_return_hand_value(bars_from):   # closes 100 → 110: ret_1[1] = ln(1.1)

def test_dist_high_low_hand_values(bars_from):
    # highs [10, 12, 11], lows [8, 9, 7], closes [9, 11, 8], via dist_high / dist_low with n=3:
    # dist_high at bar 2 = 8/12 − 1; dist_low at bar 2 = 8/7 − 1

def test_ratio_to_previous_mean_excludes_current():
    s = pd.Series([1., 1., 1., 4.])
    assert ratio_to_previous_mean(s, 3).iloc[3] == 4.0 and np.isnan(ratio_to_previous_mean(s, 3).iloc[2])

def test_accel_6_is_change_in_ret_6(make_bars):

def test_volume_hand_values(bars_from):
    # volumes [4, 4, 8], taker [1, 1, 6]: buy_share [0.25, 0.25, 0.75]; sell_volume [3, 3, 2];
    # volume_change[2] = ln 2

def test_zero_volume_gives_null_share_and_change(bars_from):

def test_volume_z_is_null_when_previous_window_constant(bars_from):

def test_candle_hand_values(bars_from):      # o=10, h=14, l=8, c=12, atr=4
    c = candles.compute(b, atr_14=pd.Series([4.0], index=b.index)).iloc[0]
    assert c.body_frac == pytest.approx(1/3) and c.upper_wick_frac == pytest.approx(1/3)
    assert c.lower_wick_frac == pytest.approx(1/3) and c.open_pos == pytest.approx(1/3)
    assert c.close_pos == pytest.approx(2/3) and c.body_atr == 0.5 and c.size_atr == 1.5

def test_flat_candle_fractions_are_null(bars_from):          # high == low → all fractions NaN

def test_candle_fraction_invariant(make_bars):                # |body| + wicks == 1 on every bar

def test_columns_in_spec_order(make_bars):                    # all three compute() functions
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/unit/test_features_price_volume_candles.py -q`.** Expected: fails on the import.

- [ ] **Step 3: Implement the three modules to spec §4.1, §4.5 and §4.6.** Window features use `min_periods` equal to the window, so they are NaN until the window is full.

- [ ] **Step 4: Run the Step 2 command.** Expected: all pass.

- [ ] **Step 5: Commit** the three modules and the test file with message `feat: price, volume and candle-shape features`.

---

### Task 3: Trend structure

**Files:**
- Create: `features/trend.py`
- Test: `tests/unit/test_features_trend.py`

**Interfaces:**
- Consumes: `bars_from`, `make_bars`; `indicators.ema` (Task 1) in tests.
- Produces:
  - `SWING_K = 3`
  - `trend.compute(bars, ema_20: Series, ema_50: Series) -> DataFrame`: the 13 spec §4.3 columns in order `swing_high, swing_low, dist_swing_high, dist_swing_low, swing_high_dir, swing_low_dir, structure, ema_slope_20, ema_slope_50, trend_duration, trend_accel, bos, bars_since_bos`. The integer columns (`swing_high_dir, swing_low_dir, structure, trend_duration, bos, bars_since_bos`) use the pandas nullable `Int64` dtype; the others float64.
- Rulings on points the spec leaves open:
  - `bos` is NULL until both a swing high (H2) and a swing low (L2) are confirmed.
  - `trend_duration` is NULL where either EMA is NaN. The count starts at the first bar where both exist.
  - `bars_since_bos` counts bars since the latest non-zero `bos`, including across later zero bars, and resets on each new non-zero `bos`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_swing_high_only_visible_from_confirmation_bar(bars_from):
    highs = [1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 1, 1, 1]      # peak at i = 5
    t = trend.compute(bars_from(highs=highs, lows=[h - 1 for h in highs], closes=highs), ...)
    assert t.swing_high.iloc[5:8].isna().all()          # bars 5, 6, 7: not yet confirmed
    assert t.swing_high.iloc[8] == 9                    # confirmed at 5 + SWING_K

def test_tie_is_not_a_swing(bars_from):                  # equal neighbour high → no swing

def test_higher_highs_and_higher_lows_give_up_structure(bars_from):   # two rising peaks and troughs → dirs +1, structure +1

def test_lower_highs_and_lower_lows_give_down_structure(bars_from):

def test_mixed_structure_is_zero(bars_from):

def test_bos_fires_once_on_the_crossing_bar(bars_from):
    # close crosses above a confirmed H2 at bar j: bos[j] == 1, bos[j+1] == 0,
    # bars_since_bos[j] == 0, bars_since_bos[j+2] == 2

def test_trend_duration_hand_values(bars_from):
    ema20 = pd.Series([1., 2, 3, 1, 1]); ema50 = pd.Series([2., 1, 1, 2, 1])
    # → [-1, 1, 2, -1, 0]

def test_ema_slope_and_accel_hand_values(bars_from):    # given EMA series: slope_t = ema_t / ema_{t-5} − 1

def test_trend_prefix_invariance(make_bars):
    # for 30 cut points k: compute on bars[:k] vs all bars; row k−1 identical (NaN == NaN)
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/unit/test_features_trend.py -q`.** Expected: fails on the import.

- [ ] **Step 3: Implement `features/trend.py` to spec §4.3 and the rulings above.** It must handle 1,000,000 bars within the Task 5 performance test, so prefer vectorised NumPy or a single linear pass over nested Python loops.

- [ ] **Step 4: Run the Step 2 command.** Expected: all pass.

- [ ] **Step 5: Commit** with message `feat: trend structure features with confirmed swing points`.

---

### Task 4: Volatility and the rules-based regime

**Files:**
- Create: `features/volatility.py`, `features/regime.py`
- Test: `tests/unit/test_features_volatility_regime.py`

**Interfaces:**
- Produces:
  - `volatility.VOL_PCT_WINDOW = timedelta(days=365)`, `VOL_PCT_MIN_HISTORY = timedelta(days=30)`
  - `volatility.bars_per_year(step: timedelta) -> float` = 525,600 / minutes per bar
  - `volatility.compute(bars, ret_1: Series, atr_pct: Series, step: timedelta) -> DataFrame`: columns `ret_std_20, hist_vol_20, range_pct, vol_pct_365d`
  - `regime.ADX_TREND_MIN = 20.0`, `ADX_STRONG_MIN = 25.0`, `VOL_LOW_MAX = 20.0`, `VOL_NORMAL_MAX = 80.0`, `VOL_HIGH_MAX = 95.0`
  - `regime.compute(adx_14, plus_di_14, minus_di_14, vol_pct_365d) -> DataFrame`: columns `trend_regime, volatility_regime` (object dtype; labels exactly as spec §4.7; `None` when inputs are NaN)

- [ ] **Step 1: Write the failing tests:**

```python
def test_bars_per_year():               # 1h → 8760; 5m → 105120; 1d → 365

def test_hist_vol_is_std_times_root_bars_per_year(make_bars):

def test_vol_pct_hand_values(bars_from):
    # daily steps, atr_pct given directly: 40 strictly increasing values
    # → NaN while less than 30 days after the first value, 100.0 from day 30 on
    # strictly decreasing values from day 30 → 100 × 1 / count

def test_vol_pct_window_excludes_value_exactly_365_days_old(bars_from):
    # day 0 holds a huge value; at day 365 it is outside (t − 365 d, t]

def test_vol_pct_ignores_null_atr(bars_from):          # NaN atr_pct values are not counted

@pytest.mark.parametrize("adx,plus,minus,label", [
    (19.99, 30, 10, "sideways"), (20, 30, 10, "weak_bullish"), (20, 10, 30, "weak_bearish"),
    (24.99, 30, 10, "weak_bullish"), (25, 30, 10, "strong_bullish"), (25, 10, 30, "strong_bearish"),
    (40, 20, 20, "sideways"), (np.nan, 30, 10, None)])
def test_trend_regime_thresholds(adx, plus, minus, label):

@pytest.mark.parametrize("pct,label", [(0, "low"), (19.99, "low"), (20, "normal"), (79.99, "normal"),
    (80, "high"), (94.99, "high"), (95, "extreme"), (100, "extreme"), (np.nan, None)])
def test_volatility_regime_thresholds(pct, label):
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/unit/test_features_volatility_regime.py -q`.** Expected: fails on the import.

- [ ] **Step 3: Implement both modules to spec §4.4 and §4.7.** For `vol_pct_365d`, a time-based rolling window (`rolling("365D")`, closed right) with rank `method="max"` divided by the count of non-NaN values meets the definition and the performance target. Verify NaN handling against the Step 1 tests.

- [ ] **Step 4: Run the Step 2 command.** Expected: all pass.

- [ ] **Step 5: Commit** with message `feat: volatility features and rules-based market regime`.

---

### Task 5: The pipeline and its guarantees

**Files:**
- Create: `features/pipeline.py`
- Modify: `tests/conftest.py` (the `assert_features_match` fixture)
- Test: `tests/unit/test_features_pipeline.py`

**Interfaces:**
- Consumes: the `compute` functions of Tasks 1–4 and their signatures.
- Produces:
  - `FEATURE_SET: int = 1`
  - `FEATURE_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")`
  - `FEATURE_COLUMNS: tuple[str, ...]`: the 75 columns in this exact order (price, indicators, trend, volatility, volume, candles, regime):
    ```
    ret_1 ret_3 ret_6 ret_12 ret_24 dist_high_20 dist_high_100 dist_low_20 dist_low_100 accel_6 range_ratio_20
    ema_9 ema_20 ema_50 ema_100 ema_200 dist_ema_9 dist_ema_20 dist_ema_50 dist_ema_100 dist_ema_200
    sma_20 sma_50 sma_200 dist_sma_20 dist_sma_50 dist_sma_200 rsi_14 macd_pct macd_signal_pct macd_hist_pct
    atr_14 atr_pct plus_di_14 minus_di_14 adx_14 bb_upper bb_lower bb_pct_b bb_width stoch_k_14 stoch_d_3 roc_10
    swing_high swing_low dist_swing_high dist_swing_low swing_high_dir swing_low_dir structure
    ema_slope_20 ema_slope_50 trend_duration trend_accel bos bars_since_bos
    ret_std_20 hist_vol_20 range_pct vol_pct_365d
    volume_ratio_20 volume_change volume_z_20 buy_volume sell_volume buy_share
    body_frac upper_wick_frac lower_wick_frac open_pos close_pos body_atr size_atr
    trend_regime volatility_regime
    ```
  - `INT_COLUMNS = {"swing_high_dir", "swing_low_dir", "structure", "bos"}` (smallint) and `INT32_COLUMNS = {"trend_duration", "bars_since_bos"}` (integer); `TEXT_COLUMNS = {"trend_regime", "volatility_regime"}`; all others float.
  - `WARMUP: dict[str, int]`: every column's warm-up in bars, copied from spec §4. Columns marked "—" in the spec, and the time-based `vol_pct_365d` and `volatility_regime`, get 0 and belong to `DATA_DEPENDENT_WARMUP: frozenset[str]`.
  - `LOOKBACK_BARS = 2100`, `LOOKBACK_DAYS = 400`; `lookback_start(last_built: datetime, step: timedelta) -> datetime` = last_built − max(LOOKBACK_BARS × step, LOOKBACK_DAYS days)
  - `compute_features(bars: pd.DataFrame, step: timedelta) -> pd.DataFrame`: the columns of `FEATURE_COLUMNS` in order, on `bars.index`, with nullable integer dtypes for the integer columns
- Composition order and blanking. Each module's outputs are blanked (`WARMUP[col]` bars set to NaN) **before** they are passed to a later module, so dependants never see warm-up values:
  1. `indicators.compute`
  2. `price.compute`
  3. `volume.compute`
  4. `candles.compute(bars, atr_14)`
  5. `volatility.compute(bars, ret_1, atr_pct, step)`
  6. `trend.compute(bars, ema_20, ema_50)`
  7. `regime.compute(adx_14, plus_di_14, minus_di_14, vol_pct_365d)`
- Test fixture: `assert_features_match(a: DataFrame, b: DataFrame)`. Same index and columns; floats via `numpy.isclose(rtol=1e-9, atol=1e-9)` with NaN == NaN; integers and text exactly, with NA == NA.

- [ ] **Step 1: Write the failing tests:**

```python
def test_columns_types_and_warmup_table_complete(make_bars):
    f = compute_features(make_bars(500), timedelta(hours=1))
    assert tuple(f.columns) == FEATURE_COLUMNS and len(FEATURE_COLUMNS) == 75
    assert set(WARMUP) == set(FEATURE_COLUMNS)

def test_warmup_nulls_exactly(make_bars):
    # 1h random walk, 3000 bars; for every column not in DATA_DEPENDENT_WARMUP with WARMUP w > 0:
    # all NaN below index w, non-NaN at index w

def test_no_lookahead_prefix_invariance(make_bars, assert_features_match):
    # 1h, 3000 bars; 50 cut points k spread over [100, 3000]:
    # compute_features(bars[:k]).iloc[[-1]] matches compute_features(bars).iloc[[k-1]]

def test_incremental_window_equals_full(make_bars, assert_features_match):
    # for step in (1h: 12,000 bars; 1d: 2,500 bars): cut at bar n − 500;
    # window = bars[bars.index >= lookback_start(cut_time, step)];
    # rows after cut_time match between compute_features(window) and compute_features(full)

def test_short_and_empty_history(make_bars):
    # 10 bars: no exception, every column with WARMUP > 10 all NaN;
    # 0 bars: empty frame with exactly FEATURE_COLUMNS

def test_no_infinities_on_flat_bars_and_zero_volume(make_bars):
    # set 20 bars flat (high = low = open = close) and 20 bars to volume 0 → no ±inf anywhere

def test_lookback_start():
    # 5m → 400 days; 1d → 2100 days

def test_performance_300k_bars(make_bars):
    # compute_features on 300,000 5m bars finishes in under 60 s
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/unit/test_features_pipeline.py -q`.** Expected: fails on the import.

- [ ] **Step 3: Implement `features/pipeline.py`**, then add the `assert_features_match` fixture to `tests/conftest.py`. If a guarantee test fails, the bug is in a module: fix it there, and add a module-level test that pins it.

- [ ] **Step 4: Run the Step 2 command, then `.venv\Scripts\python.exe -m pytest tests/unit -q`.** Expected: all pass.

- [ ] **Step 5: Commit** with message `feat: feature pipeline with look-ahead, warm-up and incremental guarantees`.

---

### Task 6: N1 — no false alarms on freshly closed buckets

**Files:**
- Modify: `data/quality/checks.py`
- Test: `tests/integration/test_quality_policy_lag_db.py`

**Interfaces:**
- Produces:
  - `POLICY_LAG: dict[str, timedelta]` beside `STEP`, = schedule_interval + end_offset (1 minute) + 5 minutes of grace: `{"5m": 11 min, "15m": 21 min, "1h": 66 min, "4h": 66 min, "1d": 66 min}`
  - `run_quality_checks(conn, symbol, timeframe="1m", start=None, end=None, known_outages=None, now: datetime | None = None)`. `now` defaults to `datetime.now(timezone.utc)`. For timeframes other than 1m, the default end becomes `min(existing end from 1m data, _floor(now − POLICY_LAG[tf], step) − step)`, i.e. the last bucket whose end ≤ now − lag. Everything else is unchanged.

- [ ] **Step 1: Write the failing tests** (`pytestmark = pytest.mark.db`):

```python
def test_policy_lag_matches_registered_policies(db_conn):
    # read each continuous aggregate's refresh-policy schedule_interval and config end_offset from
    # timescaledb_information.jobs (joined to continuous_aggregates);
    # for tf in 5m..1d: POLICY_LAG[tf] == schedule_interval + end_offset + timedelta(minutes=5)

def test_freshly_closed_4h_bucket_is_not_a_hole(db_conn):
    # 1m candles from 2024-05-01 00:00 to 08:04; refresh aggregates only up to 04:00;
    # run_quality_checks(conn, "BTCUSDT", "4h", now=2024-05-01 08:05) → verdict PASS, missing 0

def test_bucket_missing_beyond_the_lag_still_counts(db_conn):
    # same data, never refreshed past 04:00, now = 2024-05-01 10:00 → the 04:00 bucket counts as missing
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/integration/test_quality_policy_lag_db.py -q`.** Expected: the false-alarm test fails, and `POLICY_LAG` is missing.

- [ ] **Step 3: Implement `POLICY_LAG` and the `now` parameter.** Reuse `data.storage.repository._floor`.

- [ ] **Step 4: Run the Step 2 command, then `.venv\Scripts\python.exe -m pytest -m db -q tests/integration/test_quality_db.py`.** Expected: all pass.

- [ ] **Step 5: Commit** with message `fix: quality checks wait for the refresh policy before calling a bucket missing`.

---

### Task 7: Feature tables and the store

**Files:**
- Create: `data/storage/migrations/004_features.sql`, `features/store.py`
- Modify: `tests/conftest.py` (`db_conn` also deletes from the five feature tables)
- Test: `tests/integration/test_features_store_db.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, `INT_COLUMNS`, `INT32_COLUMNS`, `TEXT_COLUMNS`, `FEATURE_SET`, `FEATURE_TIMEFRAMES` (Task 5).
- Produces:
  - Tables `features_5m`, `features_15m`, `features_1h`, `features_4h`, `features_1d`, per spec §5.1. Suggested migration shape: create `features_5m` in full, then `CREATE TABLE features_<tf> (LIKE features_5m INCLUDING ALL)` for the others, then add each foreign key explicitly (LIKE doesn't copy foreign keys), `create_hypertable` for each, the compression settings (`segmentby symbol`, `orderby open_time DESC`) and `add_compression_policy(…, INTERVAL '30 days')` for each.
  - `store.feature_table(timeframe: str) -> str` (ValueError for an unknown timeframe)
  - `store.upsert_features(conn, symbol: str, timeframe: str, frame: pd.DataFrame) -> int`: writes `symbol`, `open_time`, `feature_set = FEATURE_SET` and every `FEATURE_COLUMNS` value. Staging COPY + `INSERT … ON CONFLICT (symbol, open_time) DO UPDATE`; NaN/NA → NULL. Returns rows written. Does not commit (the caller owns the transaction, as in `data/storage/repository.py`).
  - `store.last_built(conn, symbol, timeframe) -> datetime | None`
  - `store.has_stale_feature_set(conn, symbol, timeframe) -> bool`: any row with `feature_set <> FEATURE_SET`
  - `store.read_features(conn, symbol, timeframe, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame`: `FEATURE_COLUMNS` on a UTC index named `open_time`, dtypes as `compute_features` produces, NULL → NaN/NA
  - `store.latest_features(conn, symbol, timeframe) -> dict | None`: the newest row as a dict including `open_time`

- [ ] **Step 1: Write the failing tests** (`pytestmark = pytest.mark.db`):

```python
def test_tables_match_feature_columns(db_conn):
    # for each timeframe: information_schema column names, in ordinal order, ==
    # ["symbol", "open_time", "feature_set", *FEATURE_COLUMNS, "computed_at"]; data types
    # smallint for INT_COLUMNS, integer for INT32_COLUMNS, text for TEXT_COLUMNS, double precision otherwise

def test_feature_tables_are_compressed_hypertables(db_conn):   # 5 hypertables, 5 compression jobs

def test_round_trip_preserves_values(db_conn, make_bars, assert_features_match):
    # compute_features on 300 1h bars → upsert → read_features matches the frame

def test_nan_is_stored_as_null(db_conn, make_bars):
    # after upsert, SELECT count(*) WHERE rsi_14 IS NULL equals the frame's NaN count (the 42 warm-up rows),
    # and SELECT count(*) WHERE rsi_14 = 'NaN'::float8 is 0

def test_upsert_is_idempotent_and_updates(db_conn, make_bars):
    # same frame twice → row count unchanged; changed value on 2nd upsert → stored value updated

def test_last_built_latest_and_stale_feature_set(db_conn, make_bars):
    # last_built == last index; latest_features()["open_time"] == last index;
    # UPDATE … SET feature_set = 0 on one row → has_stale_feature_set True
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/integration/test_features_store_db.py -q`.** Expected: fails; the tables don't exist.

- [ ] **Step 3: Write the migration, update the `db_conn` cleanup, and implement `features/store.py`.** The migration runs through the existing `run_migrations`. Remember TimescaleDB needs autocommit for policies (already true there).

- [ ] **Step 4: Run the Step 2 command, then `.venv\Scripts\python.exe -m pytest -m db -q`.** Expected: all pass, including the existing schema tests.

- [ ] **Step 5: Commit** with message `feat: feature hypertables and an idempotent feature store`.

---

### Task 8: Loading bars and building features

**Files:**
- Create: `features/frame.py`, `features/build.py`
- Test: `tests/integration/test_features_build_db.py`

**Interfaces:**
- Consumes: `compute_features`, `lookback_start`, `FEATURE_TIMEFRAMES` (Task 5); store functions (Task 7); `run_quality_checks` (Task 6); `refresh_aggregates`, `_floor` from `data.storage.repository`; `STEP` from `data.quality.checks`.
- Produces:
  - `frame.build_cutoff(conn, symbol, timeframe) -> datetime | None`: exclusive upper bound on bar `open_time` = `_floor(last_1m_open_time + 1 minute, step)`; None if the symbol has no 1m candles
  - `frame.load_bars(conn, symbol, timeframe, start: datetime | None, end: datetime) -> pd.DataFrame`: bars from `candles_<tf>` with `start ≤ open_time < end` (no lower bound when start is None). Columns are cast `::float8` in SQL, in the Task 1 shape.
  - ```python
    @dataclass
    class BuildResult:
        symbol: str
        timeframe: str
        rows_written: int
        start: datetime | None       # first bar written
        end: datetime | None         # last bar written
        full: bool                   # full rebuild?
        seconds: float
        skipped_reason: str | None = None
    ```
  - `build.build_features(conn, symbol, timeframe, *, rebuild: bool = False) -> BuildResult`: no quality gate (the caller gates). Steps:
    1. cutoff
    2. full if `rebuild`, if nothing is stored, or if `has_stale_feature_set`
    3. `conn.commit()` to end the read transaction, then `refresh_aggregates(conn, refresh_start, cutoff)`, where refresh_start is the start of the load window (full: the first 1m candle)
    4. load from `lookback_start(last_built, step)` (full: None) to cutoff
    5. `compute_features`
    6. upsert rows with `open_time > last_built` (full: all)
    7. `conn.commit()`
  - `build.build_symbol(conn, symbol, timeframes=FEATURE_TIMEFRAMES, *, rebuild=False) -> list[BuildResult]`: runs the 1m quality gate once. On FAIL, every result has `rows_written=0` and `skipped_reason="data quality FAIL"`. On WARN, it logs a warning and builds. Otherwise it calls `build_features` per timeframe.

- [ ] **Step 1: Write the failing tests** (`pytestmark = pytest.mark.db`; insert 1m candles with `upsert_candles`; helper-generated prices may be a simple walk):

```python
def test_build_cutoff_excludes_partial_bucket(db_conn):
    # 1m candles 10:00 … 10:58 → 5m cutoff 10:55 (bucket 10:50 eligible, 10:55 not); no candles → None

def test_load_bars_returns_floats_on_utc_index(db_conn):

def test_first_build_is_full_then_incremental_only_adds(db_conn):
    # 3 days of 1m → build 1h (full=True, rows = eligible buckets); add 6 more hours;
    # build again → full=False, rows_written == 6, start == first new bar

def test_second_build_without_new_data_writes_nothing(db_conn):

def test_incremental_equals_rebuild(db_conn, assert_features_match):
    # after the incremental build: read_features == compute_features over all bars (full recompute)

def test_stale_feature_set_forces_full_rebuild(db_conn):

def test_build_works_without_prior_aggregate_refresh(db_conn):
    # candles inserted, aggregates never refreshed → build writes rows

def test_quality_fail_skips_symbol(db_conn):
    # a 3-hour unexplained hole in 1m → build_symbol results all skipped, no feature rows
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/integration/test_features_build_db.py -q`.** Expected: fails on the import.

- [ ] **Step 3: Implement `features/frame.py` and `features/build.py`.**

- [ ] **Step 4: Run the Step 2 command, then `.venv\Scripts\python.exe -m pytest -m db -q`.** Expected: all pass.

- [ ] **Step 5: Commit** with message `feat: incremental feature builds from the continuous aggregates`.

---

### Task 9: Readable summary, `tb features build`, `tb analyze`, README

**Files:**
- Create: `features/summary.py`
- Modify: `data/cli.py`, `README.md`
- Test: `tests/unit/test_features_summary.py`, `tests/integration/test_features_cli_db.py`

**Interfaces:**
- Consumes: `latest_features` (Task 7); `build_symbol`, `BuildResult` (Task 8); `FEATURE_TIMEFRAMES` (Task 5); `STEP`.
- Produces:
  - ```python
    @dataclass
    class TimeframeState:
        timeframe: str
        built: bool                    # False: no feature rows yet; the rest are None
        bar_close: datetime | None     # open_time + step
        age: timedelta | None
        stale: bool
        trend_regime: str | None
        adx: float | None
        structure: str | None          # "up" | "down" | "mixed" | None
        trend_duration: int | None
        rsi: float | None
        volatility_regime: str | None
        vol_pct: float | None
        volume_ratio: float | None
        buy_share: float | None
        dist_ema_200: float | None

    @dataclass
    class MarketState:
        symbol: str
        as_of: datetime
        timeframes: list[TimeframeState]   # always all five, in FEATURE_TIMEFRAMES order
        def agreement(self) -> dict[str, int]   # {"bullish": n, "bearish": n, "sideways": n}
        def to_dict(self) -> dict               # JSON-serialisable: datetimes as ISO strings, timedeltas as seconds
        def render(self) -> str
    ```
  - `STALE_GRACE = timedelta(minutes=5)`; stale = age > 2 × step + STALE_GRACE
  - `summary.state_from_rows(symbol, rows: dict[str, dict | None], now: datetime) -> MarketState`: pure, no DB (for unit tests)
  - `summary.market_state(conn, symbol, now: datetime | None = None) -> MarketState`: reads `latest_features` per timeframe
  - CLI: a `features` typer group with `tb features build [--symbol S] [--timeframe TF] [--rebuild]` (per-symbol isolation with `conn.rollback()` on error, one line per result, exit 1 if any failed or was skipped) and `tb analyze SYMBOL [--no-build]` (uppercases SYMBOL; exits 1 with one line if the symbol has no 1m candles; otherwise builds unless `--no-build`, then prints `render()`)
- Display copy: volume shows as `volume x{ratio:.2f}, buyer share {share:.0%}`. The word "buyer" is fine; spec §6.2 forbids whole-word buy, sell, long, short, enter and exit.

- [ ] **Step 1: Write the failing tests:**

```python
# tests/unit/test_features_summary.py
def test_state_maps_rows_and_marks_missing_timeframes():   # rows for 1h only → 1h built, others built=False
def test_stale_threshold():   # 1h bar closed 2 h 4 min ago → not stale; 2 h 6 min → stale
def test_agreement_counts():  # weak_/strong_bullish → bullish, …, sideways; None not counted
def test_structure_labels():  # 1 → "up", −1 → "down", 0 → "mixed", NA → None
def test_to_dict_is_json_serialisable():
def test_render_has_every_timeframe_and_no_advice_words():
    # all five labels present, "STALE" shown for a stale frame, and
    # re.search(r"\b(buy|sell|long|short|enter|exit)\b", text, re.I) is None

# tests/integration/test_features_cli_db.py  (pytestmark = pytest.mark.db; CliRunner as in tests/unit/test_cli.py)
def test_features_build_prints_results_and_exit_codes(db_conn):
def test_analyze_builds_then_prints_state(db_conn):
def test_analyze_accepts_lowercase(db_conn):
def test_analyze_unknown_symbol_exits_1(db_conn):     # one line, no traceback
def test_analyze_no_build_reads_existing_rows(db_conn):
```

- [ ] **Step 2: Run `.venv\Scripts\python.exe -m pytest tests/unit/test_features_summary.py tests/integration/test_features_cli_db.py -q`.** Expected: fails on the import.

- [ ] **Step 3: Implement `features/summary.py` and the CLI commands.** Then add a "Market features (sub-project 2a)" section to `README.md`: what the commands do, that features are built on demand (nothing keeps them fresh automatically yet), `--rebuild`, and the STALE meaning.

- [ ] **Step 4: Run the Step 2 command, then the full suite `.venv\Scripts\python.exe -m pytest -q`.** Expected: all pass.

- [ ] **Step 5: Commit** with message `feat: tb features build and tb analyze with a readable market state`.

---

## After all tasks: real-data verification (controller)

Run after the user's full backfill has finished (spec §7.6):

1. `tb db upgrade` (applies 004)
2. `tb features build`, timed
3. Per pair, feature rows = aggregate buckets below the cutoff
4. `tb analyze BTCUSDT`
5. Hand the user the Binance RSI/EMA comparison check
