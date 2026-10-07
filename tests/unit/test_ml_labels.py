"""Labels (spec section 3)."""
import math

import numpy as np
import pandas as pd
import pytest

from ml.labels import DOWN, FLAT, UP, make_labels, threshold


def _hours(*hs):
    base = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DatetimeIndex([base + pd.Timedelta(hours=h) for h in hs],
                            name="open_time")


def test_threshold_scales_with_sqrt_h():
    assert threshold(0.01, 4) == pytest.approx(0.01)
    assert threshold(0.01, 1) == pytest.approx(0.005)


def test_classes():
    idx = _hours(0, 1, 2, 3, 4)
    close = pd.Series([100.0, 101.0, 100.0, 99.0, 100.0], index=idx)
    atr = pd.Series(0.01, index=idx)
    out = make_labels(close, atr, 1)
    assert out.dtype == "Int8"
    assert out.iloc[:4].tolist() == [UP, DOWN, DOWN, UP]
    assert out.isna().iloc[4]


def test_flat_inside_threshold():
    idx = _hours(0, 1)
    close = pd.Series([100.0, 100.2], index=idx)
    out = make_labels(close, pd.Series(0.01, index=idx), 1)
    assert out.iloc[0] == FLAT


def test_gap_gives_na():
    idx = _hours(0, 1, 3, 4, 5, 6, 7)
    close = pd.Series(np.linspace(100, 106, len(idx)), index=idx)
    atr = pd.Series(0.01, index=idx)
    h1 = make_labels(close, atr, 1)
    assert h1.isna().iloc[1]          # hour 2 missing
    assert not h1.isna().iloc[0]
    h4 = make_labels(close, atr, 4)
    # hour 0 -> 4 and hour 1 -> 5 span the missing hour 2
    assert h4.isna().iloc[0] and h4.isna().iloc[1]
    assert not h4.isna().iloc[2]      # 3 -> 7, complete


def test_null_atr_gives_na():
    idx = _hours(0, 1, 2)
    close = pd.Series([100.0, 102.0, 104.0], index=idx)
    atr = pd.Series([math.nan, 0.01, 0.01], index=idx)
    out = make_labels(close, atr, 1)
    assert out.isna().iloc[0] and out.iloc[1] == UP


def test_last_h_rows_na():
    idx = _hours(*range(10))
    close = pd.Series(np.arange(100.0, 110.0), index=idx)
    out = make_labels(close, pd.Series(0.01, index=idx), 4)
    assert out.isna().iloc[-4:].all() and not out.isna().iloc[:-4].any()
