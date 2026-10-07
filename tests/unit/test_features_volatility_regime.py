"""Hand-verified tests for volatility features and the regime (spec 4.4, 4.7)."""

import math
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from features import regime, volatility
from features.price import log_return

DAY = timedelta(days=1)


def _vol_pct(bars_from, atr_values):
    """vol_pct_365d for daily bars whose atr_pct is given directly."""
    n = len(atr_values)
    ones = np.ones(n)
    bars = bars_from(highs=ones, lows=ones, closes=ones, step=DAY)
    atr_pct = pd.Series(atr_values, index=bars.index, dtype="float64")
    ret_1 = pd.Series(np.nan, index=bars.index)
    return volatility.compute(bars, ret_1, atr_pct, DAY)["vol_pct_365d"]


def test_bars_per_year():
    assert volatility.bars_per_year(timedelta(hours=1)) == 8760
    assert volatility.bars_per_year(timedelta(minutes=5)) == 105120
    assert volatility.bars_per_year(timedelta(days=1)) == 365


def test_constants():
    assert volatility.VOL_PCT_WINDOW == timedelta(days=365)
    assert volatility.VOL_PCT_MIN_HISTORY == timedelta(days=30)


def test_hist_vol_is_std_times_root_bars_per_year(make_bars):
    bars = make_bars(200)
    ret_1 = log_return(bars["close"], 1)
    atr_pct = pd.Series(1.0, index=bars.index)
    out = volatility.compute(bars, ret_1, atr_pct, timedelta(hours=1))
    # hand value at bar 50: sample std (ddof=1) of ret_1[31..50]
    window = ret_1.iloc[31:51].to_numpy()
    expected_std = float(np.std(window, ddof=1))
    assert out["ret_std_20"].iloc[50] == pytest.approx(expected_std, rel=1e-12)
    assert out["hist_vol_20"].iloc[50] == pytest.approx(
        expected_std * math.sqrt(8760), rel=1e-12)
    # warm-up: ret_1[0] is NaN, so the first full window ends at bar 20
    assert out["ret_std_20"].iloc[:20].isna().all()
    assert out["hist_vol_20"].iloc[:20].isna().all()
    assert out["ret_std_20"].iloc[20:].notna().all()


def test_range_pct(bars_from):
    bars = bars_from(highs=[12.0, 5.0], lows=[8.0, 5.0], closes=[10.0, 0.0])
    nan = pd.Series(np.nan, index=bars.index)
    out = volatility.compute(bars, nan, nan, timedelta(hours=1))
    assert out["range_pct"].iloc[0] == pytest.approx(0.4)
    assert np.isnan(out["range_pct"].iloc[1])  # close 0 -> blank, not inf


def test_vol_pct_hand_values(bars_from):
    # 40 strictly increasing values: every value is the max of its window
    up = _vol_pct(bars_from, np.arange(1.0, 41.0))
    assert up.iloc[:30].isna().all()          # days 0..29: < 30 days of history
    assert (up.iloc[30:] == 100.0).all()      # day 30 on: 100 * n / n
    # strictly decreasing: value at day d is the smallest of d + 1 values
    down = _vol_pct(bars_from, np.arange(40.0, 0.0, -1.0))
    assert down.iloc[:30].isna().all()
    for d in (30, 31, 39):
        assert down.iloc[d] == pytest.approx(100.0 * 1 / (d + 1))


def test_vol_pct_ties_count_as_at_or_below(bars_from):
    # 31 values, all equal: every one is <= the current, so 100
    out = _vol_pct(bars_from, np.full(31, 2.0))
    assert out.iloc[30] == 100.0


def test_vol_pct_window_excludes_value_exactly_365_days_old(bars_from):
    values = np.ones(367)
    values[0] = 1000.0                # huge value on day 0
    values[365] = 1.0
    out = _vol_pct(bars_from, values)
    # day 364: day 0 is inside (t - 365d, t]; current 1.0 ranks below it
    assert out.iloc[364] == pytest.approx(100.0 * 364 / 365)
    # day 365: day 0 is exactly 365 days old, so outside; 365 ones remain
    assert out.iloc[365] == 100.0


def test_vol_pct_ignores_null_atr(bars_from):
    # leading NaNs (warm-up): history clock starts at the first non-NaN value
    values = np.concatenate([np.full(10, np.nan), np.arange(1.0, 41.0)])
    out = _vol_pct(bars_from, values)
    assert out.iloc[:40].isna().all()         # first value at day 10; ready day 40
    assert (out.iloc[40:] == 100.0).all()
    # NaN inside the window is not counted in the denominator
    values = np.arange(1.0, 41.0)
    values[5] = np.nan
    values[20] = np.nan
    values[35] = 0.5              # smallest of the 34 non-NaN values in days 0..35
    out = _vol_pct(bars_from, values)
    assert np.isnan(out.iloc[5]) and np.isnan(out.iloc[20])
    assert out.iloc[35] == pytest.approx(100.0 * 1 / 34)


def test_vol_pct_stays_blank_when_current_atr_is_blank(bars_from):
    values = np.arange(1.0, 41.0)
    values[35] = np.nan
    out = _vol_pct(bars_from, values)
    assert np.isnan(out.iloc[35])
    assert out.iloc[36] == 100.0


def test_vol_pct_is_fast_on_a_five_minute_year():
    n = 300_000
    step = timedelta(minutes=5)
    idx = pd.date_range("2022-01-01", periods=n, freq="5min", tz="UTC",
                        name="open_time")
    rng = np.random.default_rng(1)
    atr = pd.Series(np.exp(rng.normal(0, 0.3, n)), index=idx)
    atr.iloc[:42] = np.nan
    ones = pd.Series(1.0, index=idx)
    bars = pd.DataFrame({"high": ones, "low": ones, "close": ones})
    t0 = time.perf_counter()
    out = volatility.compute(bars, ones * np.nan, atr, step)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0
    assert out["vol_pct_365d"].iloc[:8640 + 42].isna().all()
    assert out["vol_pct_365d"].iloc[-1] >= 0.0


def _assert_label(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert type(actual) is str and actual == expected


@pytest.mark.parametrize("adx,plus,minus,label", [
    (19.99, 30, 10, "sideways"), (20, 30, 10, "weak_bullish"),
    (20, 10, 30, "weak_bearish"),
    (24.99, 30, 10, "weak_bullish"), (25, 30, 10, "strong_bullish"),
    (25, 10, 30, "strong_bearish"),
    (40, 20, 20, "sideways"), (np.nan, 30, 10, None),
    (30, np.nan, 10, None), (30, 10, np.nan, None)])
def test_trend_regime_thresholds(adx, plus, minus, label):
    out = regime.compute(pd.Series([adx]), pd.Series([plus]), pd.Series([minus]),
                         pd.Series([50.0]))
    _assert_label(out["trend_regime"].iloc[0], label)


@pytest.mark.parametrize("pct,label", [
    (0, "low"), (19.99, "low"), (20, "normal"), (79.99, "normal"),
    (80, "high"), (94.99, "high"), (95, "extreme"), (100, "extreme"),
    (np.nan, None)])
def test_volatility_regime_thresholds(pct, label):
    nan = pd.Series([np.nan])
    out = regime.compute(nan, nan, nan, pd.Series([pct]))
    _assert_label(out["volatility_regime"].iloc[0], label)


def test_regime_columns_are_object_dtype_and_independent():
    idx = pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC")
    out = regime.compute(
        pd.Series([np.nan, 30.0, 10.0], index=idx),
        pd.Series([np.nan, 30.0, 10.0], index=idx),
        pd.Series([np.nan, 10.0, 30.0], index=idx),
        pd.Series([50.0, np.nan, 99.0], index=idx))
    assert list(out.columns) == ["trend_regime", "volatility_regime"]
    assert out["trend_regime"].dtype == object
    assert out["volatility_regime"].dtype == object
    assert out.index.equals(idx)
    assert list(out["trend_regime"]) == [None, "strong_bullish", "sideways"]
    assert list(out["volatility_regime"]) == ["normal", None, "extreme"]


def test_regime_constants():
    assert (regime.ADX_TREND_MIN, regime.ADX_STRONG_MIN) == (20.0, 25.0)
    assert (regime.VOL_LOW_MAX, regime.VOL_NORMAL_MAX, regime.VOL_HIGH_MAX) == (
        20.0, 80.0, 95.0)


def test_vol_pct_matches_brute_force_definition():
    n = 1700                                   # 6h bars: 425 days, window rolls
    idx = pd.date_range("2022-01-01", periods=n, freq="6h", tz="UTC",
                        name="open_time")
    rng = np.random.default_rng(7)
    values = np.round(rng.normal(5, 1, n), 1)  # rounding forces ties
    values[rng.random(n) < 0.1] = np.nan
    values[:20] = np.nan
    atr = pd.Series(values, index=idx)
    got = volatility.vol_percentile(atr)
    first = atr.first_valid_index()
    for i in (0, 100, 200, 300, 699, 1000, 1400, 1699):
        t = idx[i]
        in_window = atr[(idx > t - timedelta(days=365)) & (idx <= t)].dropna()
        if np.isnan(atr.iloc[i]) or t - first < timedelta(days=30):
            assert np.isnan(got.iloc[i])
        else:
            expected = 100.0 * (in_window <= atr.iloc[i]).sum() / len(in_window)
