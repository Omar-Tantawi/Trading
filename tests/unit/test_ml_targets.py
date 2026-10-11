"""vol3 and dir2 labels (3b spec section 2)."""
import math

import numpy as np
import pandas as pd
import pytest

from ml.targets import TARGETS, dir2_labels, get_target, vol3_labels

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def _idx(*hours):
    return pd.DatetimeIndex([T0 + pd.Timedelta(hours=h) for h in hours], name="open_time")


def _bars(close, high=None, low=None, hours=None):
    idx = _idx(*(hours if hours is not None else range(len(close))))
    c = pd.Series(close, index=idx, dtype=float)
    h = pd.Series(high if high is not None else close, index=idx, dtype=float)
    l = pd.Series(low if low is not None else close, index=idx, dtype=float)
    return c, h, l


def test_registry():
    assert set(TARGETS) == {"move3", "vol3", "dir2"}
    assert get_target("vol3").classes == ("quiet", "normal", "wild")
    assert get_target("dir2").classes == ("down", "up")
    assert get_target("move3").classes == ("down", "flat", "up")
    with pytest.raises(ValueError, match="move3, vol3, dir2"):
        get_target("nope")


def test_vol3_classes_h1():
    # u = 0.01 * sqrt(1): quiet < 0.005, normal < 0.0125, wild >= 0.0125
    c, h, l = _bars([100, 100, 100, 100], high=[100, 100.3, 101, 102], low=[100, 99.9, 100, 100])
    atr = pd.Series(0.01, index=c.index)
    out = vol3_labels(c, h, l, atr, 1)
    # row 0 -> bar 1: max(ln(100.3/100), ln(100/99.9)) = 0.003 -> quiet
    # row 1 -> bar 2: ln(101/100) = 0.00995 -> normal; row 2 -> bar 3: ln(102/100) -> wild
    assert out.iloc[:3].tolist() == [0, 1, 2] and out.isna().iloc[3]


def test_vol3_scales_with_sqrt_h_and_ignores_bar_t():
    # bar 0 itself spikes (must not count); bars 1..4 stay within 0.9%
    c, h, l = _bars([100] * 6, high=[150, 100.9, 100.5, 100.2, 100.1, 100],
                    low=[50, 100, 100, 100, 100, 100])
    atr = pd.Series(0.01, index=c.index)       # u at H=4 = 0.02: quiet < 0.01
    assert vol3_labels(c, h, l, atr, 4).iloc[0] == 0


def test_vol3_gap_and_null_atr():
    c, h, l = _bars([100, 101, 102, 103], hours=[0, 1, 3, 4])
    atr = pd.Series([0.01, math.nan, 0.01, 0.01], index=c.index)
    out = vol3_labels(c, h, l, atr, 1)
    assert out.isna().iloc[1]      # NULL atr (and gap)
    out4 = vol3_labels(c, h, l, pd.Series(0.01, index=c.index), 1)
    assert out4.isna().iloc[1] and not out4.isna().iloc[0]


def test_dir2():
    c, _, _ = _bars([100, 101, 100.5, 100.5, 101])
    atr = pd.Series([0.01, 0.01, 0.01, math.nan, 0.01], index=c.index)
    out = dir2_labels(c, atr, 1)
    assert out.iloc[0] == 1 and out.iloc[1] == 0
    assert out.isna().iloc[2]          # r = 0
    assert out.isna().iloc[3]          # NULL atr
    assert out.isna().iloc[4]          # no future bar


def test_move3_label_set_follows_labels_module():
    from ml.labels import LABEL_SET
    assert get_target("move3").label_set == LABEL_SET
