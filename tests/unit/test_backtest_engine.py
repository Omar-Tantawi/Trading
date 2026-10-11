"""Strategies over days (spec sections 3-4)."""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import START, STRATEGIES, CoinData, run_strategy

FEE, SLIP = 0.001, 0.0005


def _data(days=20, seed=0, drop_day=None, p_wild=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 1440, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(idx))))
    open_ = np.concatenate([[100.0], close[:-1]])
    m = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0002,
                      "low": np.minimum(open_, close) * 0.9998, "close": close}, index=idx)
    if drop_day is not None:
        day = idx[0] + pd.Timedelta(days=drop_day)
        m = m.drop(m.index[(m.index >= day) & (m.index < day + pd.Timedelta(minutes=30))])
    starts = pd.date_range("2024-01-01", periods=days, freq="1D", tz="UTC")
    pw = p_wild if p_wild is not None else rng.uniform(0.05, 0.5, days)
    d = pd.DataFrame({"atr_pct": 0.004, "p_wild": pw}, index=starts)
    return CoinData(minutes=m, days=d)


def test_strategy_names():
    assert STRATEGIES == ("cash", "hold", "grid_always", "grid_vol3_25",
                          "grid_vol3_30", "grid_vol3_35", "grid_random_30")


def test_cash_stays_flat():
    r = run_strategy(_data(), "cash", FEE, SLIP)
    assert (r.equity == START).all() and not r.active.any()


def test_hold_follows_price():
    data = _data()
    r = run_strategy(data, "hold", FEE, SLIP)
    coin = START / (100.0 * (1 + SLIP) * (1 + FEE))
    last_close = data.minutes["close"].iloc[-1]
    assert r.equity.iloc[-1] == pytest.approx(coin * last_close)


def test_pause_keeps_equity_except_exit_cost():
    data = _data(p_wild=np.array([0.1] * 5 + [0.9] * 15))
    r = run_strategy(data, "grid_vol3_30", FEE, SLIP)
    assert r.active.iloc[:5].all() and not r.active.iloc[5:].any()
    assert r.equity.iloc[6:].nunique() == 1          # flat once out of the market


def test_incomplete_day_is_cash_for_grids_and_counted():
    data = _data(drop_day=3)
    r = run_strategy(data, "grid_always", FEE, SLIP)
    assert not r.active.iloc[3] and r.active.iloc[2] and r.skipped_days == 1


def test_no_look_ahead_prefix():
    full = _data(days=20)
    cut = CoinData(minutes=full.minutes[full.minutes.index < pd.Timestamp("2024-01-11", tz="UTC")],
                   days=full.days.iloc[:10])
    for name in ("grid_always", "grid_vol3_30", "hold"):
        a = run_strategy(full, name, FEE, SLIP).equity.iloc[:10]
        b = run_strategy(cut, name, FEE, SLIP).equity
        pd.testing.assert_series_equal(a, b)


def test_random_control_pauses_as_often_as_vol3_30():
    data = _data(days=60)
    v = run_strategy(data, "grid_vol3_30", FEE, SLIP)
    r = run_strategy(data, "grid_random_30", FEE, SLIP)
    assert (~v.active).sum() == (~r.active).sum()
    assert not (v.active == r.active).all()


def test_coin_without_predictions_gives_empty_results():
    empty = CoinData(minutes=pd.DataFrame(columns=["open", "high", "low", "close"],
                                          index=pd.DatetimeIndex([], tz="UTC")),
                     days=pd.DataFrame(columns=["atr_pct", "p_wild"],
                                       index=pd.DatetimeIndex([], tz="UTC")))
    for name in STRATEGIES:
        r = run_strategy(empty, name, FEE, SLIP)
        assert r.equity.empty and r.skipped_days == 0


def test_day_with_grid_step_below_fees_is_not_traded():
    data = _data(days=5)
    data.days["atr_pct"] = 0.00005          # step far below 2 x fee
    r = run_strategy(data, "grid_always", FEE, SLIP)
    assert not r.active.any() and (r.equity == START).all()
