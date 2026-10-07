"""Metrics (spec section 6)."""
import math

import numpy as np
import pandas as pd
import pytest

from ml.metrics import (
    accuracy, bootstrap_skill_ci, brier, ece, log_loss, reliability, skill,
)


def test_log_loss_hand():
    assert log_loss(np.array([[.2, .3, .5]]), np.array([2])) == pytest.approx(-math.log(.5))


def test_brier_hand():
    assert brier(np.array([[.2, .3, .5]]), np.array([2])) == pytest.approx(.38)


def test_accuracy():
    p = np.array([[.6, .3, .1], [.1, .2, .7], [.3, .4, .3]])
    assert accuracy(p, np.array([0, 0, 1])) == pytest.approx(2 / 3)


def _sample(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > p.cumsum(1)).sum(1)
    return p, y


def test_ece_perfectly_calibrated_is_small():
    p, y = _sample()
    assert ece(p, y) < 0.01
    bins = reliability(p, y)
    assert sum(b["n"] for b in bins) == 3 * len(y)


def test_ece_overconfident_is_large():
    p, y = _sample()
    hot = p ** 3 / (p ** 3).sum(1, keepdims=True)
    assert ece(hot, y) > 0.05


def test_skill_zero_for_identical():
    assert skill(0.9, 0.9) == 0.0
    assert skill(0.45, 0.9) == pytest.approx(0.5)


def test_bootstrap_reproducible_and_contains_point():
    p, y = _sample(20_000)
    base = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
    tau = pd.date_range("2024-01-01", periods=len(y), freq="1h", tz="UTC")
    a = bootstrap_skill_ci(p, base, y, tau)
    b = bootstrap_skill_ci(p, base, y, tau)
    assert a == b
    point = skill(log_loss(p, y), log_loss(base, y))
    assert a[0] < point < a[1]
