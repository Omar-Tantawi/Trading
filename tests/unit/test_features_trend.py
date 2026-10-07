"""Hand-verified tests for trend structure features (spec section 4.3)."""

import numpy as np
import pandas as pd
import pytest

from features import trend
from features.indicators import ema
from features.trend import SWING_K


def _zigzag(pivots, *, half_range=0.5):
    """Closes linearly interpolated through (index, price) pivots; highs and
    lows sit half_range either side of the close."""
    xs, ys = zip(*pivots)
    closes = np.interp(np.arange(xs[0], xs[-1] + 1), xs, ys)
    return {"closes": closes, "highs": closes + half_range,
            "lows": closes - half_range}


def _run(bars_from, path, ema_20=None, ema_50=None):
    bars = bars_from(**path)
    close = bars["close"]
    return trend.compute(
        bars,
        ema(close, 20) if ema_20 is None else ema_20,
        ema(close, 50) if ema_50 is None else ema_50,
    )


def _float(s):
    return s.to_numpy(dtype="float64", na_value=np.nan)


# --- swing points ----------------------------------------------------------

def test_swing_k_is_three():
    assert SWING_K == 3


def test_columns_order_and_dtypes(bars_from):
    t = _run(bars_from, _zigzag([(0, 10), (4, 14), (8, 11), (12, 16)]))
    assert list(t.columns) == [
        "swing_high", "swing_low", "dist_swing_high", "dist_swing_low",
        "swing_high_dir", "swing_low_dir", "structure", "ema_slope_20",
        "ema_slope_50", "trend_duration", "trend_accel", "bos",
        "bars_since_bos",
    ]
    for col in ("swing_high_dir", "swing_low_dir", "structure",
                "trend_duration", "bos", "bars_since_bos"):
        assert str(t[col].dtype) == "Int64", col
    for col in ("swing_high", "swing_low", "dist_swing_high",
                "dist_swing_low", "ema_slope_20", "ema_slope_50",
                "trend_accel"):
        assert t[col].dtype == np.float64, col


def test_swing_high_only_visible_from_confirmation_bar(bars_from):
    highs = [1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 1, 1, 1]      # peak at i = 5
    t = _run(bars_from, dict(highs=highs, lows=[h - 1 for h in highs],
                             closes=highs))
    assert t.swing_high.iloc[:8].isna().all()            # bars 0-7 incl. 5, 6, 7
    assert t.swing_high.iloc[8] == 9                     # confirmed at 5 + SWING_K
    assert (t.swing_high.iloc[8:] == 9).all()
    assert np.isnan(t.dist_swing_high.iloc[7])
    assert t.dist_swing_high.iloc[8] == pytest.approx(3 / 9 - 1)    # close 3 / 9 - 1


def test_swing_low_only_visible_from_confirmation_bar(bars_from):
    lows = [9, 8, 7, 6, 5, 1, 5, 6, 7, 8, 9, 9, 9]       # trough at i = 5
    t = _run(bars_from, dict(highs=[x + 1 for x in lows], lows=lows,
                             closes=lows))
    assert t.swing_low.iloc[:8].isna().all()
    assert t.swing_low.iloc[8] == 1
    assert t.dist_swing_low.iloc[8] == pytest.approx(7 / 1 - 1)     # close 7 / 1 - 1
    assert t.swing_high.isna().all()                     # no swing high exists


def test_tie_is_not_a_swing(bars_from):
    highs = [1, 2, 3, 9, 9, 3, 2, 1, 1, 1]               # bars 3 and 4 tie
    t = _run(bars_from, dict(highs=highs, lows=[h - 1 for h in highs],
                             closes=highs))
    assert t.swing_high.isna().all()
    # A tie three bars away still blocks: bar 3 == bar 6.
    highs = [1, 2, 3, 9, 4, 5, 9, 2, 1, 1]
    t = _run(bars_from, dict(highs=highs, lows=[h - 1 for h in highs],
                             closes=highs))
    assert t.swing_high.isna().all()


def test_short_series_has_no_swings(bars_from):
    t = _run(bars_from, dict(highs=[1, 2, 3], lows=[0, 1, 2], closes=[1, 2, 3]))
    assert t.swing_high.isna().all() and t.swing_low.isna().all()
    assert t.bos.isna().all()


# --- structure -------------------------------------------------------------

def test_higher_highs_and_higher_lows_give_up_structure(bars_from):
    # swing highs at 4, 12, 20 (14.5, 16.5, 18.5); swing lows at 8, 16, 24
    # (10.5, 12.5, 14.5); confirmed at +3.
    t = _run(bars_from, _zigzag([(0, 10), (4, 14), (8, 11), (12, 16),
                                 (16, 13), (20, 18), (24, 15), (28, 20)]))
    assert t.swing_high_dir.iloc[14] is pd.NA            # only one high confirmed (7)
    assert t.swing_high_dir.iloc[15] == 1                # 16.5 > 14.5 at bar 15
    assert t.swing_low_dir.iloc[18] is pd.NA             # lows confirmed at 11, 19
    assert t.swing_low_dir.iloc[19] == 1                 # 12.5 > 10.5
    assert t.structure.iloc[18] is pd.NA
    assert t.structure.iloc[19] == 1
    assert t.structure.iloc[27] == 1                     # L2 = 14.5 > 12.5
    assert t.swing_high.iloc[27] == 18.5 and t.swing_low.iloc[27] == 14.5


def test_lower_highs_and_lower_lows_give_down_structure(bars_from):
    # swing lows at 4, 12, 20 (15.5, 13.5, 11.5); swing highs at 8, 16, 24
    # (19.5, 17.5, 15.5).
    t = _run(bars_from, _zigzag([(0, 20), (4, 16), (8, 19), (12, 14),
                                 (16, 17), (20, 12), (24, 15), (28, 10)]))
    assert t.structure.iloc[18] is pd.NA
    assert t.swing_high_dir.iloc[19] == -1               # 17.5 < 19.5
    assert t.swing_low_dir.iloc[19] == -1                # 13.5 < 15.5
    assert t.structure.iloc[19] == -1
    assert t.structure.iloc[27] == -1


def test_mixed_structure_is_zero(bars_from):
    # swing highs rise (14.5, 16.5, 18.5), swing lows fall (10.5, 9.5, 8.5).
    t = _run(bars_from, _zigzag([(0, 12), (4, 14), (8, 11), (12, 16),
                                 (16, 10), (20, 18), (24, 9), (28, 19)]))
    assert t.swing_high_dir.iloc[27] == 1
    assert t.swing_low_dir.iloc[27] == -1
    assert t.structure.iloc[27] == 0


def test_equal_swing_prices_give_direction_zero(bars_from):
    # Swing highs at 4, 12, 20 all at 14.5; swing lows at 8, 16 both at 10.5.
    t = _run(bars_from, _zigzag([(0, 10), (4, 14), (8, 11), (12, 14),
                                 (16, 11), (20, 14), (24, 10)]))
    assert t.swing_high_dir.iloc[15] == 0
    assert t.swing_low_dir.iloc[19] == 0
    assert t.structure.iloc[19] == 0


# --- break of structure ----------------------------------------------------

def test_bos_fires_once_on_the_crossing_bar(bars_from):
    # Swing high at 4 (high 14.5, confirmed 7), swing low at 8 (low 10.5,
    # confirmed 11), then +0.5 per bar: close 14.5 at bar 15 (not above H2),
    # 15.0 at bar 16 (crosses).
    t = _run(bars_from, _zigzag([(0, 10), (4, 14), (8, 11), (24, 19)]))
    assert t.bos.iloc[10] is pd.NA                       # no swing low yet
    assert t.bos.iloc[11] == 0
    assert t.bos.iloc[15] == 0                           # close == H2: not above
    assert t.bos.iloc[16] == 1
    assert t.bos.iloc[17] == 0                           # fires once
    assert t.bos.iloc[24] == 0
    assert t.bars_since_bos.iloc[15] is pd.NA
    assert t.bars_since_bos.iloc[16] == 0
    assert t.bars_since_bos.iloc[17] == 1
    assert t.bars_since_bos.iloc[18] == 2


def test_bos_down_is_negative(bars_from):
    # Swing low at 4 (15.5, confirmed 7), swing high at 8 (19.5, confirmed 11),
    # then -0.5 per bar: close 15.5 at bar 15, 15.0 at bar 16 (crosses).
    t = _run(bars_from, _zigzag([(0, 20), (4, 16), (8, 19), (24, 11)]))
    assert t.bos.iloc[15] == 0
    assert t.bos.iloc[16] == -1
    assert t.bos.iloc[17] == 0
    assert t.bars_since_bos.iloc[16] == 0
    assert t.bars_since_bos.iloc[18] == -2               # signed by direction


def test_bars_since_bos_counts_through_zero_bars_and_resets():
    bos = np.array([np.nan, 0, 1, 0, 0, -1, 0, 0, 1])
    got = trend._bars_since_nonzero(bos)
    expected = np.array([np.nan, np.nan, 0, 1, 2, 0, -1, -2, 0])
    np.testing.assert_array_equal(got, expected)


# --- EMA-based columns -----------------------------------------------------

def test_trend_duration_hand_values(bars_from):
    closes = [1, 2, 3, 4, 5]
    bars = bars_from(highs=closes, lows=closes, closes=closes)
    ema20 = pd.Series([1., 2, 3, 1, 1], index=bars.index)
    ema50 = pd.Series([2., 1, 1, 2, 1], index=bars.index)
    t = trend.compute(bars, ema20, ema50)
    # below(-1), above(+1), above(+2), below(-1), equal(0)
    assert t.trend_duration.tolist() == [-1, 1, 2, -1, 0]


def test_trend_duration_starts_at_first_bar_with_both_emas(bars_from):
    closes = [1.0] * 8
    bars = bars_from(highs=closes, lows=closes, closes=closes)
    ema20 = pd.Series([np.nan, np.nan, 3, 3, 3, 1, 1, 1.], index=bars.index)
    ema50 = pd.Series([np.nan, np.nan, np.nan, 2, 2, 2, 2, 2.], index=bars.index)
    t = trend.compute(bars, ema20, ema50)
    assert t.trend_duration.iloc[:3].isna().all()
    assert t.trend_duration.tolist()[3:] == [1, 2, -1, -2, -3]


def test_ema_slope_and_accel_hand_values(bars_from):
    closes = [float(i) for i in range(1, 16)]
    bars = bars_from(highs=closes, lows=closes, closes=closes)
    ema20 = pd.Series(closes, index=bars.index)               # 1, 2, ..., 15
    ema50 = pd.Series([np.nan] * 3 + [2 * c for c in closes[3:]], index=bars.index)
    t = trend.compute(bars, ema20, ema50)
    assert t.ema_slope_20.iloc[:5].isna().all()
    assert t.ema_slope_20.iloc[5] == pytest.approx(6 / 1 - 1)        # 5
    assert t.ema_slope_20.iloc[10] == pytest.approx(11 / 6 - 1)      # 0.8333
    # first valid slope_50 needs ema50 at t and t - 5: t = 8 -> 18 / 8 - 1
    assert t.ema_slope_50.iloc[:8].isna().all()
    assert t.ema_slope_50.iloc[8] == pytest.approx(18 / 8 - 1)       # 1.25
    # accel = slope_20(t) - slope_20(t - 5): first valid at t = 10
    assert t.trend_accel.iloc[:10].isna().all()
    assert t.trend_accel.iloc[10] == pytest.approx((11 / 6 - 1) - (6 / 1 - 1))


def test_zero_ema_gives_nan_slope_not_inf(bars_from):
    closes = [1.0] * 8
    bars = bars_from(highs=closes, lows=closes, closes=closes)
    zero = pd.Series([0.0, 1, 1, 1, 1, 1, 1, 1], index=bars.index)
    t = trend.compute(bars, zero, zero)
    assert np.isnan(t.ema_slope_20.iloc[5])
    assert not np.isinf(t.ema_slope_20.to_numpy()).any()


# --- no look-ahead ---------------------------------------------------------

def test_trend_prefix_invariance(make_bars):
    bars = make_bars(n=600, seed=3)
    ema20 = ema(bars["close"], 20).where(np.arange(len(bars)) >= 60)
    ema50 = ema(bars["close"], 50).where(np.arange(len(bars)) >= 150)
    full = trend.compute(bars, ema20, ema50)
    assert full.swing_high.notna().any() and (full.bos.abs() > 0).any()
    for k in np.linspace(20, len(bars), 30).astype(int):
        part = trend.compute(bars.iloc[:k], ema20.iloc[:k], ema50.iloc[:k])
        for col in trend.COLUMNS:
            a = _float(part[col])[k - 1]
            b = _float(full[col])[k - 1]
            assert np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), (col, k)
