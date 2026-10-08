"""Dataset and the as-of multi-timeframe join (spec section 4.1)."""
from datetime import timedelta

import numpy as np
import pandas as pd

from features.pipeline import FEATURE_COLUMNS, TEXT_COLUMNS
from ml.dataset import (
    EXCLUDED, MODEL_FEATURES, TREND_REGIMES, VOL_REGIMES, SymbolData,
    asof_join, build_dataset, encode, symbol_features,
)

UTC0 = pd.Timestamp("2024-01-01", tz="UTC")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")


def _features(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({c: rng.normal(size=len(index)) for c in FEATURE_COLUMNS
                        if c not in TEXT_COLUMNS}, index=index)
    out["atr_pct"] = 0.01
    out["trend_regime"] = "sideways"
    out["volatility_regime"] = "low"
    return out[list(FEATURE_COLUMNS)]


def _idx(start, n, step):
    return pd.DatetimeIndex([start + i * step for i in range(n)], name="open_time")


def test_model_features_excludes_price_levels():
    assert not EXCLUDED & set(MODEL_FEATURES)
    for c in ("rsi_14", "atr_pct", "dist_ema_200"):
        assert c in MODEL_FEATURES
    assert len(MODEL_FEATURES) == len(FEATURE_COLUMNS) - 17


def test_asof_4h_visible_only_after_close():
    four = pd.DataFrame({"x": [1.0, 2.0]}, index=_idx(UTC0, 2, timedelta(hours=4)))
    base = _idx(UTC0, 9, timedelta(hours=1))
    out = asof_join(base, four, timedelta(hours=4), "h4_")
    assert list(out.columns) == ["h4_x"]
    at = lambda h: out.loc[UTC0 + timedelta(hours=h), "h4_x"]
    assert np.isnan(at(2))
    assert at(3) == 1.0 and at(6) == 1.0
    assert at(7) == 2.0


def test_asof_1d_visible_only_after_close():
    day = pd.DataFrame({"x": [5.0]}, index=_idx(UTC0, 1, timedelta(days=1)))
    base = _idx(UTC0, 30, timedelta(hours=1))
    out = asof_join(base, day, timedelta(days=1), "d1_")
    assert np.isnan(out.loc[UTC0 + timedelta(hours=22), "d1_x"])
    assert out.loc[UTC0 + timedelta(hours=23), "d1_x"] == 5.0


def test_encode_one_hot_fixed_columns():
    sd = _symbol_data(48)
    frame = symbol_features(sd)
    X = encode(frame, "ETHUSDT", SYMBOLS)
    assert not any(X.dtypes == object)
    for label in TREND_REGIMES:
        assert f"trend_regime={label}" in X.columns
        assert f"h4_trend_regime={label}" in X.columns
    for label in VOL_REGIMES:
        assert f"d1_volatility_regime={label}" in X.columns
    assert (X["trend_regime=sideways"] == 1.0).all()
    assert (X["trend_regime=strong_bullish"] == 0.0).all()
    assert (X["symbol=ETHUSDT"] == 1.0).all() and (X["symbol=BTCUSDT"] == 0.0).all()


def _symbol_data(hours: int, seed: int = 0) -> SymbolData:
    i1 = _idx(UTC0, hours, timedelta(hours=1))
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, hours))), index=i1)
    return SymbolData(
        close=close,
        f1h=_features(i1, seed),
        f4h=_features(_idx(UTC0, hours // 4, timedelta(hours=4)), seed + 1),
        f1d=_features(_idx(UTC0, max(hours // 24, 1), timedelta(days=1)), seed + 2),
    )


def test_build_dataset_sorted_and_labelled():
    data = {"BTCUSDT": _symbol_data(100, 0), "ETHUSDT": _symbol_data(80, 1)}
    ds = build_dataset(data, 4)
    assert len(ds.X) == len(ds.y) == (100 - 4) + (80 - 4)
    assert (ds.tau == ds.open_time + pd.Timedelta(hours=1)).all()
    assert ds.tau.is_monotonic_increasing
    assert set(np.unique(ds.y)) <= {0, 1, 2}
    assert set(ds.symbol) == {"BTCUSDT", "ETHUSDT"}
    assert "symbol=BTCUSDT" in ds.X.columns
