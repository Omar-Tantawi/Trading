"""Financial metrics (spec section 5)."""
import numpy as np
import pandas as pd
import pytest

from backtest.metrics import diff_ci, summary

DAYS = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")


def test_summary_hand_series():
    equity = pd.Series([1000, 1100, 990, 1089, 1089, 1200.0], index=DAYS)
    active = pd.Series([True, True, True, False, True, True], index=DAYS)
    s = summary(equity, active, round_trips=7, fees=3.0, start=1000.0)
    assert s["total_return"] == pytest.approx(0.2)
    assert s["max_drawdown"] == pytest.approx(990 / 1100 - 1)
    assert s["active_days"] == 5 and s["round_trips"] == 7
    assert s["fees_pct"] == pytest.approx(0.003)
    # active-day returns: 0 (day 1 vs start), +10%, -10%, (inactive), 0, +10.19%
    assert s["win_share"] == pytest.approx(2 / 5)
    assert s["longest_losing_streak"] == 1
    assert s["sharpe"] > 0
    assert s["annual_return"] == pytest.approx(1.2 ** (365 / 6) - 1)


def test_flat_series_has_zero_sharpe_and_drawdown():
    s = summary(pd.Series(1000.0, index=DAYS), pd.Series(False, index=DAYS), 0, 0.0, 1000.0)
    assert s["sharpe"] == 0.0 and s["max_drawdown"] == 0.0 and s["active_days"] == 0


def test_diff_ci_reproducible_and_contains_mean():
    rng = np.random.default_rng(1)
    days = pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")
    a = pd.Series(rng.normal(0.001, 0.01, 400), index=days)
    b = pd.Series(rng.normal(0.0, 0.01, 400), index=days)
    m, lo, hi = diff_ci(a, b)
    assert (m, lo, hi) == diff_ci(a, b)
    assert lo < m < hi
    assert m == pytest.approx((a - b).mean() * 1e4)
