"""Hand-verified tests for price, volume and candle-shape features
(spec sections 4.1, 4.5, 4.6)."""

import math

import numpy as np
import pandas as pd
import pytest

from features import candles, price, volume
from features.price import ratio_to_previous_mean

PRICE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "dist_high_20", "dist_high_100", "dist_low_20", "dist_low_100",
    "accel_6", "range_ratio_20",
]
VOLUME_COLUMNS = [
    "volume_ratio_20", "volume_change", "volume_z_20",
    "buy_volume", "sell_volume", "buy_share",
]
CANDLE_COLUMNS = [
    "body_frac", "upper_wick_frac", "lower_wick_frac",
    "open_pos", "close_pos", "body_atr", "size_atr",
]


# --- price -----------------------------------------------------------------

def test_log_return_hand_value(bars_from):
    b = bars_from(highs=[101, 111], lows=[99, 99], closes=[100, 110])
    r = price.log_return(b["close"], 1)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(math.log(1.1))


def test_log_return_warm_up_is_n(make_bars):
    b = make_bars(40)
    r = price.log_return(b["close"], 6)
    assert r.iloc[:6].isna().all() and r.iloc[6:].notna().all()


def test_dist_high_low_hand_values(bars_from):
    b = bars_from(highs=[10, 12, 11], lows=[8, 9, 7], closes=[9, 11, 8])
    dh = price.dist_high(b["close"], b["high"], 3)
    dl = price.dist_low(b["close"], b["low"], 3)
    assert dh.iloc[:2].isna().all() and dl.iloc[:2].isna().all()
    assert dh.iloc[2] == pytest.approx(8 / 12 - 1)
    assert dl.iloc[2] == pytest.approx(8 / 7 - 1)


def test_dist_high_window_includes_current_bar(bars_from):
    # the current bar sets the window max, so the distance is exactly 0
    b = bars_from(highs=[10, 10, 20], lows=[5, 5, 5], closes=[8, 8, 20])
    assert price.dist_high(b["close"], b["high"], 3).iloc[2] == pytest.approx(0.0)


def test_ratio_to_previous_mean_excludes_current():
    s = pd.Series([1., 1., 1., 4.])
    out = ratio_to_previous_mean(s, 3)
    assert out.iloc[3] == 4.0 and np.isnan(out.iloc[2])


def test_ratio_to_previous_mean_is_nan_when_mean_is_zero():
    s = pd.Series([0., 0., 0., 5.])
    out = ratio_to_previous_mean(s, 3)
    assert np.isnan(out.iloc[3])
    assert not np.isinf(out).any()


def test_accel_6_is_change_in_ret_6(make_bars):
    b = make_bars(60)
    out = price.compute(b)
    ret6 = out["ret_6"]
    expected = ret6 - ret6.shift(6)
    pd.testing.assert_series_equal(out["accel_6"], expected, check_names=False)
    assert out["accel_6"].iloc[:12].isna().all()
    assert out["accel_6"].iloc[12:].notna().all()


def test_range_ratio_20_hand_value(bars_from):
    # ranges: twenty bars of 2, then one of 6 -> 6 / mean(previous 20) = 3
    n = 21
    highs = [11.0] * 20 + [13.0]
    lows = [9.0] * 20 + [7.0]
    b = bars_from(highs=highs, lows=lows, closes=[10.0] * n)
    out = price.compute(b)["range_ratio_20"]
    assert out.iloc[:20].isna().all()
    assert out.iloc[20] == pytest.approx(3.0)


def test_price_warm_ups_match_spec(make_bars):
    out = price.compute(make_bars(150))
    idx = out.index
    expected = {
        "ret_1": 1, "ret_3": 3, "ret_6": 6, "ret_12": 12, "ret_24": 24,
        "dist_high_20": 19, "dist_high_100": 99,
        "dist_low_20": 19, "dist_low_100": 99,
        "accel_6": 12, "range_ratio_20": 20,
    }
    for col, pos in expected.items():
        assert out[col].first_valid_index() == idx[pos], col


# --- volume ----------------------------------------------------------------

def test_volume_hand_values(bars_from):
    b = bars_from(highs=[2, 2, 2], lows=[1, 1, 1], closes=[1.5, 1.5, 1.5],
                  volumes=[4, 4, 8], taker=[1, 1, 6])
    out = volume.compute(b)
    assert out["buy_share"].tolist() == [0.25, 0.25, 0.75]
    assert out["sell_volume"].tolist() == [3, 3, 2]
    assert out["buy_volume"].tolist() == [1, 1, 6]
    assert np.isnan(out["volume_change"].iloc[0])
    assert out["volume_change"].iloc[1] == 0.0
    assert out["volume_change"].iloc[2] == pytest.approx(math.log(2))


def test_volume_ratio_and_z_hand_values(bars_from):
    # previous 20 volumes: ten 1s and ten 3s -> mean 2, sample std sqrt(20/19)
    vols = [1.0] * 10 + [3.0] * 10 + [5.0]
    n = len(vols)
    b = bars_from(highs=[2.0] * n, lows=[1.0] * n, closes=[1.5] * n, volumes=vols)
    out = volume.compute(b)
    assert out["volume_ratio_20"].iloc[:20].isna().all()
    assert out["volume_z_20"].iloc[:20].isna().all()
    assert out["volume_ratio_20"].iloc[20] == pytest.approx(5.0 / 2.0)
    assert out["volume_z_20"].iloc[20] == pytest.approx(
        (5.0 - 2.0) / math.sqrt(20 / 19))


def test_zero_volume_gives_null_share_and_change(bars_from):
    b = bars_from(highs=[2, 2, 2], lows=[1, 1, 1], closes=[1.5, 1.5, 1.5],
                  volumes=[5, 0, 5], taker=[2, 0, 2])
    out = volume.compute(b)
    assert np.isnan(out["buy_share"].iloc[1])
    assert np.isnan(out["volume_change"].iloc[1])
    assert np.isnan(out["volume_change"].iloc[2])
    assert out["buy_share"].iloc[0] == pytest.approx(0.4)
    assert out["buy_volume"].iloc[1] == 0.0 and out["sell_volume"].iloc[1] == 0.0
    assert not np.isinf(out.to_numpy()).any()


def test_volume_z_is_null_when_previous_window_constant(bars_from):
    n = 25
    b = bars_from(highs=[2.0] * n, lows=[1.0] * n, closes=[1.5] * n,
                  volumes=[3.7] * n)
    out = volume.compute(b)
    assert out["volume_z_20"].isna().all()
    assert not np.isinf(out.to_numpy()).any()


def test_volume_ratio_is_null_when_previous_mean_is_zero(bars_from):
    vols = [0.0] * 20 + [4.0]
    n = len(vols)
    b = bars_from(highs=[2.0] * n, lows=[1.0] * n, closes=[1.5] * n,
                  volumes=vols, taker=[0.0] * n)
    out = volume.compute(b)
    assert np.isnan(out["volume_ratio_20"].iloc[20])


# --- candle shape ----------------------------------------------------------

def test_candle_hand_values(bars_from):
    b = bars_from(opens=[10], highs=[14], lows=[8], closes=[12])
    c = candles.compute(b, atr_14=pd.Series([4.0], index=b.index)).iloc[0]
    assert c.body_frac == pytest.approx(1 / 3)
    assert c.upper_wick_frac == pytest.approx(1 / 3)
    assert c.lower_wick_frac == pytest.approx(1 / 3)
    assert c.open_pos == pytest.approx(1 / 3)
    assert c.close_pos == pytest.approx(2 / 3)
    assert c.body_atr == 0.5 and c.size_atr == 1.5


def test_bearish_candle_has_negative_body(bars_from):
    # o=13, h=14, l=8, c=9: range 6, body -4/6, upper 1/6, lower 1/6
    b = bars_from(opens=[13], highs=[14], lows=[8], closes=[9])
    c = candles.compute(b, atr_14=pd.Series([2.0], index=b.index)).iloc[0]
    assert c.body_frac == pytest.approx(-4 / 6)
    assert c.upper_wick_frac == pytest.approx(1 / 6)
    assert c.lower_wick_frac == pytest.approx(1 / 6)
    assert c.body_atr == pytest.approx(-2.0) and c.size_atr == pytest.approx(3.0)


def test_flat_candle_fractions_are_null(bars_from):
    b = bars_from(opens=[10], highs=[10], lows=[10], closes=[10])
    out = candles.compute(b, atr_14=pd.Series([1.0], index=b.index))
    for col in ("body_frac", "upper_wick_frac", "lower_wick_frac",
                "open_pos", "close_pos"):
        assert np.isnan(out[col].iloc[0]), col
    assert out["size_atr"].iloc[0] == 0.0
    assert not np.isinf(out.to_numpy()).any()


def test_zero_or_missing_atr_gives_null_atr_features(bars_from):
    b = bars_from(opens=[10, 10], highs=[14, 14], lows=[8, 8], closes=[12, 12])
    out = candles.compute(b, atr_14=pd.Series([0.0, np.nan], index=b.index))
    assert out[["body_atr", "size_atr"]].isna().all().all()
    assert out["body_frac"].notna().all()


def test_candle_fraction_invariant(make_bars):
    b = make_bars(500)
    atr = pd.Series(1.0, index=b.index)
    out = candles.compute(b, atr_14=atr)
    total = out["body_frac"].abs() + out["upper_wick_frac"] + out["lower_wick_frac"]
    assert out["body_frac"].notna().all()
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-12)


# --- shared ----------------------------------------------------------------

def test_columns_in_spec_order(make_bars):
    b = make_bars(200)
    atr = pd.Series(1.0, index=b.index)
    assert list(price.compute(b).columns) == PRICE_COLUMNS
    assert list(volume.compute(b).columns) == VOLUME_COLUMNS
    assert list(candles.compute(b, atr_14=atr).columns) == CANDLE_COLUMNS


def test_outputs_are_float64_on_the_input_index(make_bars):
    b = make_bars(200)
    atr = pd.Series(1.0, index=b.index)
    for out in (price.compute(b), volume.compute(b),
                candles.compute(b, atr_14=atr)):
        assert out.index.equals(b.index)
        assert (out.dtypes == "float64").all()
