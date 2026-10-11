"""One day of a spot grid (sub-project 4 spec section 2)."""
import numpy as np
import pytest

from backtest.grid import N_GRIDS, levels, path, run_day

FEE = 0.001
LV = np.array([90.0, 95.0, 100.0, 105.0, 110.0])     # 4 grids, for hand checks


def test_levels():
    lv = levels(100.0, 0.01)                          # W = 3 * 0.01 * sqrt(24)
    w = 3 * 0.01 * 24 ** 0.5
    assert len(lv) == N_GRIDS + 1
    assert lv[0] == pytest.approx(100 * (1 - w)) and lv[-1] == pytest.approx(100 * (1 + w))


def test_path_order():
    p = path(np.array([10.0, 10.0]), np.array([12.0, 12.0]),
             np.array([9.0, 9.0]), np.array([11.0, 9.5]))
    assert p.tolist() == [10, 9, 12, 11, 10, 12, 9, 9.5]


def test_one_oscillation_is_one_round_trip():
    # start at 101: grids 0-1 (90-95, 95-100) wait to buy, grid 2 (100-105)
    # straddles -> waits to buy at 100, grid 3 (105-110) holds coin
    r = run_day(np.array([101.0, 100.0, 105.0, 101.0]), LV, cash=1000.0, coin=0.0,
                fee=FEE, slippage=0.0)
    a = 0.995 * 1000 / 4
    q2, q3 = a / 100.0, a / 105.0
    assert r.round_trips == 1
    # grid 3 keeps the coin bought at 101: it sells only at its top level, 110
    assert r.coin == pytest.approx(q3)
    start_cost = q3 * 101 * (1 + FEE)
    cash = 1000 - start_cost - q2 * 100 * (1 + FEE) + q2 * 105 * (1 - FEE)
    assert r.cash == pytest.approx(cash)
    assert r.fees == pytest.approx(FEE * (q3 * 101 + q2 * 100 + q2 * 105))


def test_full_fall_ends_in_coin_and_full_rise_in_cash():
    down = run_day(np.array([101.0, 89.0]), LV, cash=1000.0, coin=0.0, fee=FEE, slippage=0.0)
    a = 0.995 * 1000 / 4
    assert down.coin == pytest.approx(sum(a / l for l in LV[:-1]))
    assert down.cash >= 0
    up = run_day(np.array([101.0, 111.0]), LV, cash=1000.0, coin=0.0, fee=FEE, slippage=0.0)
    assert up.coin == pytest.approx(0.0) and up.round_trips == 0


def test_gap_fills_every_crossed_grid_once():
    r = run_day(np.array([101.0, 91.0]), LV, cash=1000.0, coin=0.0, fee=FEE, slippage=0.0)
    assert r.fills == 1 + 2          # start buy for grid 3, then buys at 100 and 95 (91 > 90)
    a = 0.995 * 1000 / 4
    assert r.coin == pytest.approx(a / 105 + a / 100 + a / 95)


def test_rebalance_trades_only_the_difference():
    a = 0.995 * 1000 / 4
    have = a / 105.0                  # exactly grid 3's quantity; equity stays 1,000
    r = run_day(np.array([101.0]), LV, cash=1000.0 - have * 101, coin=have,
                fee=FEE, slippage=0.0)
    assert r.fills == 0 and r.fees == 0.0 and r.coin == pytest.approx(have)


def test_cash_never_negative_over_random_days():
    rng = np.random.default_rng(0)
    cash, coin = 1000.0, 0.0
    for _ in range(300):
        p0 = 100 * np.exp(rng.normal(0, 0.05))
        x = p0 * np.exp(np.cumsum(rng.normal(0, 0.004, 400)))
        x[0] = p0
        r = run_day(x, levels(p0, 0.006), cash=cash, coin=coin, fee=FEE, slippage=0.0005)
        assert r.cash >= -1e-9
        cash, coin = r.cash, r.coin
