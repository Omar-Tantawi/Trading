"""Baselines, ML models and temperature calibration (spec section 5)."""
import numpy as np
import pandas as pd
import pytest

from ml.baselines import BaseRate, RsiZone
from ml.models import apply_temperature, fit_temperature, make_models


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _sample(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3))
    p = _softmax(logits)
    y = (rng.random(n)[:, None] > p.cumsum(1)).sum(1)
    return logits, p, y


def test_temperature_recovers_known_T():
    logits, p, y = _sample()
    assert fit_temperature(_softmax(logits * 2), y) == pytest.approx(2.0, abs=0.2)
    assert fit_temperature(p, y) == pytest.approx(1.0, abs=0.1)
    assert np.allclose(apply_temperature(p, 1.0), p)


def test_base_rate_is_class_share():
    X = pd.DataFrame({"a": [0.0] * 4})
    m = BaseRate().fit(X, np.array([0, 0, 1, 2]), X.iloc[:0], np.array([], int))
    assert np.allclose(m.predict_proba(X[:2]), [[.5, .25, .25]] * 2)


def test_rsi_zone_lookup():
    X = pd.DataFrame({"rsi_14": [20.0, 25.0, 50.0, 80.0, np.nan]})
    y = np.array([2, 2, 1, 0, 1])
    m = RsiZone()
    m.calibrate = False
    m.fit(X, y, X.iloc[:0], np.array([], int))
    p = m.predict_proba(pd.DataFrame({"rsi_14": [10.0, 90.0, np.nan]}))
    # zone < 30: counts (0,0,2) + 1 -> (1,1,3)/5
    assert np.allclose(p[0], [.2, .2, .6])
    # zone > 70: (1,0,0) + 1 -> (2,1,1)/4
    assert np.allclose(p[1], [.5, .25, .25])
    # unknown state: smoothed base rate (1, 2, 2) + 1 -> (2, 3, 3)/8
    assert np.allclose(p[2], [.25, .375, .375])


def _random_xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)),
                     columns=["rsi_14", "trend_duration", "macd_hist_pct", "a", "b"])
    X["rsi_14"] = 50 + 20 * X["rsi_14"]
    X.loc[::7, "a"] = np.nan
    X["empty"] = np.nan
    return X, rng.integers(0, 3, n)


@pytest.mark.parametrize("model", make_models(), ids=lambda m: m.name)
def test_all_models_proba_shape(model):
    X, y = _random_xy()
    model.fit(X[:400], y[:400], X[400:500], y[400:500])
    p = model.predict_proba(X[500:])
    assert p.shape == (100, 3)
    assert not np.isnan(p).any()
    assert ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p.sum(1), 1.0, atol=1e-9)


@pytest.mark.parametrize("model", make_models(), ids=lambda m: m.name)
def test_missing_class_still_three_columns(model):
    X, _ = _random_xy()
    y = np.where(np.arange(len(X)) % 2 == 0, 0, 2)
    model.fit(X[:400], y[:400], X[400:500], y[400:500])
    p = model.predict_proba(X[500:])
    assert p.shape == (100, 3) and np.allclose(p.sum(1), 1.0)


def test_model_order():
    assert [m.name for m in make_models()] == [
        "base_rate_v1", "ema_cross_v1", "rsi_v1", "macd_v1", "logreg_v1", "xgb_v1"]
