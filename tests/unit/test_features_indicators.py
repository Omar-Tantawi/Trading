import numpy as np
import pandas as pd
import pytest

from features.indicators import (
    atr,
    bollinger,
    compute,
    dmi,
    ema,
    macd,
    rsi,
    stochastic,
    true_range,
    wilder,
)

SPEC_COLUMNS = [
    "ema_9", "ema_20", "ema_50", "ema_100", "ema_200",
    "dist_ema_9", "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200",
    "sma_20", "sma_50", "sma_200", "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct", "atr_14", "atr_pct",
    "plus_di_14", "minus_di_14", "adx_14", "bb_upper", "bb_lower", "bb_pct_b",
    "bb_width", "stoch_k_14", "stoch_d_3", "roc_10",
]


def test_bar_factories_shape(make_bars, bars_from):
    b = make_bars(50, seed=3)
    assert list(b.columns) == ["open", "high", "low", "close", "volume", "taker_buy_base"]
    assert b.index.name == "open_time" and str(b.index.tz) == "UTC"
    assert (b.dtypes == "float64").all()
    assert (b.index[1:] - b.index[:-1] == pd.Timedelta(hours=1)).all()
    assert b.open.iloc[0] == 100.0 and (b.high > b.low).all() and (b.volume > 0).all()
    assert b.open.iloc[1:].tolist() == b.close.iloc[:-1].tolist()
    assert make_bars(50, seed=3).equals(b)
    e = bars_from(highs=[2, 3], lows=[1, 1], closes=[1.5, 2])
    assert e.open.tolist() == [1.5, 2] and e.volume.tolist() == [1, 1]
    assert e.taker_buy_base.tolist() == [0.5, 0.5]
    assert e.index.name == "open_time" and str(e.index.tz) == "UTC"


def test_ema_hand_values():            # n=3 -> alpha=0.5
    assert ema(pd.Series([1., 2., 3., 4.]), 3).tolist() == [1, 1.5, 2.25, 3.125]


def test_ema_skips_leading_nans():
    out = ema(pd.Series([np.nan, np.nan, 1., 2., 3.]), 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [1, 1.5, 2.25]


def test_wilder_hand_values():         # n=2 -> alpha=0.5
    assert wilder(pd.Series([2., 4., 4.]), 2).tolist() == [2, 3, 3.5]


def test_rsi_hand_values():            # changes +1, -1, +2; n=2
    r = rsi(pd.Series([10., 11., 10., 12.]), 2)
    assert np.isnan(r.iloc[0]) and r.iloc[1] == 100 and r.iloc[2] == 50
    assert r.iloc[3] == pytest.approx(83.3333333, rel=1e-6)   # RS = 1.25 / 0.25


def test_rsi_extremes():
    up = rsi(pd.Series(np.arange(1., 60.)))
    down = rsi(pd.Series(np.arange(60., 1., -1.)))
    flat = rsi(pd.Series(np.full(40, 7.0)))
    assert np.isnan(up.iloc[0]) and (up.iloc[1:] == 100).all()
    assert np.isnan(down.iloc[0]) and (down.iloc[1:] == 0).all()
    assert np.isnan(flat.iloc[0]) and (flat.iloc[1:] == 50).all()


def test_true_range_and_atr_hand_values(bars_from):
    b = bars_from(highs=[10, 12, 11], lows=[8, 9, 7], closes=[9, 11, 8])
    # TR0 = 10-8 = 2; TR1 = max(3, |12-9|, |9-9|) = 3; TR2 = max(4, |11-11|, |7-11|) = 4
    assert true_range(b.high, b.low, b.close).tolist() == [2, 3, 4]
    assert atr(b.high, b.low, b.close, 2).tolist() == [2, 2.5, 3.25]


def test_atr_of_constant_range_without_gaps_is_that_range(bars_from):
    c = np.arange(100., 140.)
    b = bars_from(highs=c + 1, lows=c - 1, closes=c)   # close stays inside next bar's range
    assert atr(b.high, b.low, b.close, 14).tolist() == pytest.approx([2.0] * 40)


def test_dmi_hand_values(bars_from):   # same bars, n=2
    b = bars_from(highs=[10, 12, 11], lows=[8, 9, 7], closes=[9, 11, 8])
    plus, minus, adx = dmi(b.high, b.low, b.close, 2)
    # +DM = [nan, 2, 0], -DM = [nan, 0, 2]; smoothed +DM [nan, 2, 1], -DM [nan, 0, 1]
    # ATR = [2, 2.5, 3.25]
    assert np.isnan(plus.iloc[0]) and np.isnan(minus.iloc[0]) and np.isnan(adx.iloc[0])
    assert plus.iloc[1] == pytest.approx(80.0) and minus.iloc[1] == 0
    assert plus.iloc[2] == pytest.approx(100 / 3.25) == minus.iloc[2]
    # DX = [nan, 100, 0]; ADX = wilder(DX, 2) = [nan, 100, 50]
    assert adx.iloc[1] == pytest.approx(100.0) and adx.iloc[2] == pytest.approx(50.0)


def test_dmi_zero_atr_is_nan_not_inf(bars_from):
    b = bars_from(highs=[5.0] * 30, lows=[5.0] * 30, closes=[5.0] * 30)
    plus, minus, adx = dmi(b.high, b.low, b.close)
    for s in (plus, minus, adx):
        assert s.isna().all()


def test_dx_is_zero_when_di_sum_is_zero(bars_from):
    # Inside bars after the first: no directional movement, but ATR > 0.
    b = bars_from(highs=[10, 9, 9, 9], lows=[8, 8.5, 8.5, 8.5], closes=[9, 8.8, 8.8, 8.8])
    plus, minus, adx = dmi(b.high, b.low, b.close, 2)
    assert plus.iloc[1:].tolist() == [0, 0, 0] and minus.iloc[1:].tolist() == [0, 0, 0]
    assert adx.iloc[1:].tolist() == [0, 0, 0]


def test_macd_of_constant_series_is_zero():
    m, s, h = macd(pd.Series(np.full(100, 42.0)))
    for x in (m, s, h):
        assert (x == 0).all()


def test_macd_relations():
    close = pd.Series(np.sin(np.arange(200) / 7.0) + 5)
    m, s, h = macd(close)
    assert m.tolist() == (ema(close, 12) - ema(close, 26)).tolist()
    assert s.tolist() == ema(m, 9).tolist()
    assert h.tolist() == pytest.approx((m - s).tolist())


def test_bollinger_hand_values():      # n=3, k=2 on [1, 2, 3]: sma 2, population std sqrt(2/3)
    up, lo = bollinger(pd.Series([1., 2., 3.]), 3, 2.0)
    assert up.iloc[:2].isna().all() and lo.iloc[:2].isna().all()
    assert up.iloc[2] == pytest.approx(2 + 2 * (2 / 3) ** 0.5)
    assert lo.iloc[2] == pytest.approx(2 - 2 * (2 / 3) ** 0.5)


def test_bollinger_flat_window_at_btc_price_has_zero_width():
    # A flat window after a varied history: the band width must be exactly 0
    # (so bb_pct_b is blank), not a rounding residue of the history or of the
    # window's mean.
    varied = [63_000.0 + 97.13 * (i % 7) for i in range(30)]
    close = pd.Series(varied + [63_696.17] * 20)
    up, lo = bollinger(close, 20, 2.0)
    assert up.iloc[-1] == lo.iloc[-1]


def test_stochastic_bounds_and_flat_window(make_bars, bars_from):
    b = make_bars(400, seed=5)
    k, d = stochastic(b.high, b.low, b.close)
    assert k.iloc[:13].isna().all() and k.iloc[13:].between(0, 100).all()
    assert d.iloc[:15].isna().all() and d.iloc[15:].between(0, 100).all()
    flat = bars_from(highs=[5.0] * 30, lows=[5.0] * 30, closes=[5.0] * 30)
    fk, fd = stochastic(flat.high, flat.low, flat.close)
    assert fk.isna().all() and fd.isna().all()


def test_stochastic_hand_values(bars_from):
    b = bars_from(highs=[10, 12, 11, 14], lows=[8, 9, 7, 10], closes=[9, 11, 8, 13])
    k, d = stochastic(b.high, b.low, b.close, 3, 2)
    # bar 2: window lows min 7, highs max 12 -> 100*(8-7)/5 = 20
    # bar 3: lows min 7, highs max 14 -> 100*(13-7)/7
    assert k.iloc[2] == pytest.approx(20.0) and k.iloc[3] == pytest.approx(600 / 7)
    assert d.iloc[3] == pytest.approx((20 + 600 / 7) / 2)
    assert d.iloc[:3].isna().all()


def test_ema_and_sma_of_constant_equal_constant(bars_from):
    b = bars_from(highs=[10.0] * 300, lows=[10.0] * 300, closes=[10.0] * 300)
    out = compute(b)
    for n in (9, 20, 50, 100, 200):
        assert (out[f"ema_{n}"] == 10.0).all()
    for n in (20, 50, 200):
        assert (out[f"sma_{n}"].iloc[n - 1:] == 10.0).all()
    assert (out["bb_upper"].iloc[19:] == 10.0).all()


def test_adx_rises_above_25_on_steady_trend(bars_from):
    c = np.arange(1., 101.)
    b = bars_from(highs=c + 0.5, lows=c - 0.5, closes=c)
    plus, minus, adx = dmi(b.high, b.low, b.close)
    assert adx.iloc[-1] > 25
    assert plus.iloc[-1] > 0 and minus.iloc[-1] == 0


def test_compute_columns_in_spec_order(make_bars):
    assert list(compute(make_bars(300)).columns) == SPEC_COLUMNS


def test_compute_types_index_and_no_inf(make_bars, bars_from):
    b = make_bars(400)
    out = compute(b)
    assert out.index.equals(b.index)
    assert (out.dtypes == "float64").all()
    assert not np.isinf(out.to_numpy()).any()
    # No warm-up blanking in this module: only natural rolling gaps.
    assert out["ema_200"].notna().all() and out["rsi_14"].iloc[1:].notna().all()
    flat = bars_from(highs=[10.0] * 300, lows=[10.0] * 300, closes=[10.0] * 300)
    fo = compute(flat)
    assert not np.isinf(fo.to_numpy()).any()
    assert fo["bb_pct_b"].isna().all() and fo["stoch_k_14"].isna().all()
    assert fo["plus_di_14"].isna().all() and fo["atr_14"].iloc[-1] == 0


def test_compute_formulas_against_definitions(make_bars):
    b = make_bars(300, seed=9)
    out = compute(b)
    close = b.close
    assert out["dist_ema_20"].tolist() == pytest.approx((close / out["ema_20"] - 1).tolist())
    assert out["dist_sma_50"].dropna().tolist() == pytest.approx(
        (close / out["sma_50"] - 1).dropna().tolist())
    assert out["atr_pct"].tolist() == pytest.approx((out["atr_14"] / close).tolist())
    assert out["roc_10"].iloc[10:].tolist() == pytest.approx(
        (close / close.shift(10) - 1).iloc[10:].tolist())
    assert out["roc_10"].iloc[:10].isna().all()
    width = out["bb_upper"] - out["bb_lower"]
    assert out["bb_pct_b"].dropna().tolist() == pytest.approx(
        ((close - out["bb_lower"]) / width).dropna().tolist())
    assert out["bb_width"].dropna().tolist() == pytest.approx(
        (width / out["sma_20"]).dropna().tolist())
    m, s, h = macd(close)
    assert out["macd_pct"].tolist() == pytest.approx((m / close).tolist())
    assert out["macd_signal_pct"].tolist() == pytest.approx((s / close).tolist())
    assert out["macd_hist_pct"].tolist() == pytest.approx((h / close).tolist())
    assert out["rsi_14"].iloc[1:].between(0, 100).all()
