import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from features.pipeline import (
    DATA_DEPENDENT_WARMUP,
    FEATURE_COLUMNS,
    INT32_COLUMNS,
    INT_COLUMNS,
    LOOKBACK_BARS,
    LOOKBACK_DAYS,
    TEXT_COLUMNS,
    WARMUP,
    compute_features,
    lookback_start,
)

HOUR = timedelta(hours=1)


def test_columns_types_and_warmup_table_complete(make_bars):
    f = compute_features(make_bars(500), HOUR)
    assert tuple(f.columns) == FEATURE_COLUMNS and len(FEATURE_COLUMNS) == 75
    assert len(set(FEATURE_COLUMNS)) == 75
    assert set(WARMUP) == set(FEATURE_COLUMNS)
    assert DATA_DEPENDENT_WARMUP <= set(FEATURE_COLUMNS)
    assert all(WARMUP[c] == 0 for c in DATA_DEPENDENT_WARMUP)
    for col in FEATURE_COLUMNS:
        if col in INT_COLUMNS | INT32_COLUMNS:
            assert pd.api.types.is_integer_dtype(f[col]), col
        elif col in TEXT_COLUMNS:
            assert f[col].dtype == object, col
        else:
            assert f[col].dtype == "float64", col


def test_warmup_values_from_spec():
    # Spot checks against spec section 4 (the values are copied, not derived).
    expected = {
        "ret_1": 1, "ret_24": 24, "dist_high_20": 19, "dist_high_100": 99,
        "dist_low_100": 99, "accel_6": 12, "range_ratio_20": 20,
        "ema_9": 27, "ema_200": 600, "dist_ema_50": 150,
        "sma_20": 19, "dist_sma_200": 199, "rsi_14": 42,
        "macd_pct": 78, "macd_signal_pct": 105, "macd_hist_pct": 105,
        "atr_14": 42, "atr_pct": 42, "plus_di_14": 42, "adx_14": 84,
        "bb_upper": 19, "bb_width": 19, "stoch_k_14": 13, "stoch_d_3": 15,
        "roc_10": 10, "ema_slope_20": 65, "ema_slope_50": 155,
        "trend_duration": 150, "trend_accel": 70,
        "ret_std_20": 20, "hist_vol_20": 20, "range_pct": 0,
        "volume_ratio_20": 20, "volume_change": 1, "volume_z_20": 20,
        "buy_volume": 0, "body_frac": 0, "body_atr": 42, "size_atr": 42,
        "swing_high": 0, "bos": 0, "bars_since_bos": 0,
        "vol_pct_365d": 0, "volatility_regime": 0,
    }
    for col, w in expected.items():
        assert WARMUP[col] == w, col
    assert DATA_DEPENDENT_WARMUP >= {
        "swing_high", "swing_low", "dist_swing_high", "dist_swing_low",
        "swing_high_dir", "swing_low_dir", "structure", "bos", "bars_since_bos",
        "vol_pct_365d", "volatility_regime",
    }
    assert WARMUP["trend_regime"] == 84  # adx_14 is its slowest input


def test_warmup_nulls_exactly(make_bars):
    f = compute_features(make_bars(3000), HOUR)
    checked = 0
    for col in FEATURE_COLUMNS:
        w = WARMUP[col]
        if col in DATA_DEPENDENT_WARMUP or w == 0:
            continue
        assert f[col].iloc[:w].isna().all(), f"{col}: values before warm-up {w}"
        assert not pd.isna(f[col].iloc[w]), f"{col}: still null at warm-up {w}"
        checked += 1
    assert checked >= 50


def test_trend_uses_warmup_blanked_emas(make_bars):
    # The pipeline hands trend the *blanked* EMAs, so the EMA-derived trend
    # columns start counting from the first trustworthy EMA, not from bar 0.
    for seed in range(5):
        f = compute_features(make_bars(400, seed=seed), HOUR)
        gap = (f["ema_20"] - f["ema_50"]).to_numpy()
        expected, count = [], 0
        for value in gap:
            if np.isnan(value):
                expected.append(None)
                count = 0
                continue
            sign = int(np.sign(value))
            count = count + sign if count != 0 and np.sign(count) == sign else sign
            expected.append(count)
        actual = [None if pd.isna(v) else int(v) for v in f["trend_duration"]]
        assert actual == expected
        assert abs(f["trend_duration"].iloc[WARMUP["trend_duration"]]) == 1
        slope = f["ema_20"] / f["ema_20"].shift(5) - 1
        pd.testing.assert_series_equal(
            f["ema_slope_20"], slope, check_names=False, rtol=1e-12)


def test_vol_percentile_starts_after_atr_pct_warmup(make_bars):
    # atr_pct reaches the percentile already blanked (42 bars), and the
    # percentile needs 30 more days of history: 42 + 720 bars on 1h. Feeding it
    # the raw atr_pct would start it 42 bars earlier, at 720.
    f = compute_features(make_bars(1000), HOUR)
    first = WARMUP["atr_pct"] + 30 * 24
    assert first == 762
    for col in ("vol_pct_365d", "volatility_regime"):
        assert f[col].iloc[:first].isna().all(), col
        assert not pd.isna(f[col].iloc[first]), col


def test_no_lookahead_prefix_invariance(make_bars, assert_features_match):
    bars = make_bars(3000)
    full = compute_features(bars, HOUR)
    for k in np.linspace(100, 3000, 50).astype(int):
        prefix = compute_features(bars.iloc[:k], HOUR)
        assert_features_match(prefix.iloc[[-1]], full.iloc[[k - 1]])


def test_incremental_window_equals_full(make_bars, assert_features_match):
    for step, n in ((HOUR, 12_000), (timedelta(days=1), 2_500)):
        bars = make_bars(n, step=step)
        cut_time = bars.index[n - 500]
        window = bars[bars.index >= lookback_start(cut_time.to_pydatetime(), step)]
        assert 0 < len(window) <= len(bars)
        from_window = compute_features(window, step)
        from_full = compute_features(bars, step)
        after = from_full.index[from_full.index > cut_time]
        assert len(after) == 499
        assert_features_match(from_window.loc[after], from_full.loc[after])


def _btc_like_bars(n, step, seed=0):
    """A walk at BTC price levels (30,000 to 60,000 on a 0.01 tick) with calm
    and volatile stretches: 0.01 % to 0.5 % per bar, each 50 to 2,000 bars
    long. make_bars' price of about 100 hides float drift in rolling
    statistics that shows at these levels."""
    rng = np.random.default_rng(seed)
    sigma = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(50, 2000))
        sigma[i:i + length] = rng.choice([0.0001, 0.0002, 0.001, 0.003, 0.005])
        i += length
    lo, hi = np.log(30_000), np.log(60_000)
    walk = np.mod(np.cumsum(rng.normal(0.0, sigma)), 2 * (hi - lo))
    close = np.round(np.exp(lo + np.minimum(walk, 2 * (hi - lo) - walk)), 2)
    open_ = np.concatenate(([close[0]], close[:-1]))
    wick = np.abs(rng.normal(0.0, sigma / 2))
    volume = np.round(np.exp(rng.normal(np.log(800), 0.6, n)), 3)
    index = pd.DatetimeIndex(
        [datetime(2018, 1, 1, tzinfo=timezone.utc) + i * step for i in range(n)],
        name="open_time")
    return pd.DataFrame(
        {"open": open_,
         "high": np.round(np.maximum(open_, close) * (1 + wick), 2),
         "low": np.round(np.minimum(open_, close) * (1 - wick), 2),
         "close": close, "volume": volume,
         "taker_buy_base": np.round(volume * rng.uniform(0.3, 0.7, n), 3)},
        index=index)


def test_incremental_window_equals_full_at_btc_prices(assert_features_match):
    # About 6.8 years of 1h bars. A rolling statistic whose value depends on
    # how many bars came before would make the window and the full history
    # disagree here (spec section 5.3).
    n = 60_000
    bars = _btc_like_bars(n, HOUR)
    assert bars["close"].between(30_000, 60_000).all()
    cut_time = bars.index[n - 500]
    window = bars[bars.index >= lookback_start(cut_time.to_pydatetime(), HOUR)]
    assert len(window) < len(bars)
    from_window = compute_features(window, HOUR)
    from_full = compute_features(bars, HOUR)
    after = from_full.index[from_full.index > cut_time]
    assert len(after) == 499
    assert_features_match(from_window.loc[after], from_full.loc[after])


def test_short_and_empty_history(make_bars):
    short = compute_features(make_bars(10), HOUR)
    assert tuple(short.columns) == FEATURE_COLUMNS and len(short) == 10
    for col in FEATURE_COLUMNS:
        if WARMUP[col] > 10:
            assert short[col].isna().all(), col

    empty = compute_features(make_bars(5).iloc[:0], HOUR)
    assert len(empty) == 0 and tuple(empty.columns) == FEATURE_COLUMNS


def test_no_infinities_on_flat_bars_and_zero_volume(make_bars):
    bars = make_bars(600)
    for col in ("open", "high", "low", "close"):
        bars.iloc[300:320, bars.columns.get_loc(col)] = float(bars["close"].iloc[300])
    for col in ("volume", "taker_buy_base"):
        bars.iloc[400:420, bars.columns.get_loc(col)] = 0.0
    f = compute_features(bars, HOUR)
    floats = f.select_dtypes("float64")
    assert not np.isinf(floats.to_numpy()).any()


def test_lookback_start():
    last = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (LOOKBACK_BARS, LOOKBACK_DAYS) == (2100, 400)
    # max(400 days, 365 days + 2,100 bars)
    # 5m: 365 d + 7.3 d; 15m: 365 d + 21.9 d; both under 400 days
    assert last - lookback_start(last, timedelta(minutes=5)) == timedelta(days=400)
    assert last - lookback_start(last, timedelta(minutes=15)) == timedelta(days=400)
    # 1h: 365 d + 87.5 d; 4h: 365 d + 350 d; 1d: 365 d + 2,100 d
    assert last - lookback_start(last, HOUR) == timedelta(days=365, hours=2100)
    assert last - lookback_start(last, timedelta(hours=4)) == timedelta(days=715)
    assert last - lookback_start(last, timedelta(days=1)) == timedelta(days=2465)


def test_performance_300k_bars(make_bars):
    bars = make_bars(300_000, step=timedelta(minutes=5))
    started = time.perf_counter()
    f = compute_features(bars, timedelta(minutes=5))
    elapsed = time.perf_counter() - started
    assert len(f) == 300_000
    assert elapsed < 60, f"took {elapsed:.1f}s"
