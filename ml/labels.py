"""Class labels for the prediction models (spec section 3).

A 1h feature row keyed t is known when its bar closes, at t + 1h. The label
compares that bar's close with the close of the bar keyed t + H hours:

    r   = ln(close_{t+H} / close_t)
    thr = LABEL_K * atr_pct_t * sqrt(H)
    up if r > thr, down if r < -thr, flat otherwise.

The label is missing when atr_pct_t is missing or any 1h bar between t and
t + H is missing.
"""
import math

import numpy as np
import pandas as pd

LABEL_K = 0.5
LABEL_SET = 1
HORIZONS: tuple[int, ...] = (1, 4, 24)

DOWN, FLAT, UP = 0, 1, 2
CLASSES = ("down", "flat", "up")

_HOUR = pd.Timedelta(hours=1)


def threshold(atr_pct, horizon: int):
    """The log-return a move must exceed to count as up or down."""
    return LABEL_K * atr_pct * math.sqrt(horizon)


def make_labels(close: pd.Series, atr_pct: pd.Series, horizon: int) -> pd.Series:
    """Labels on close.index (Int8, missing as <NA>)."""
    index = close.index
    target = index + horizon * _HOUR
    pos_now = np.arange(len(index))
    pos_then = index.get_indexer(target)
    # Present and exactly `horizon` bars later in position: no gap between.
    complete = (pos_then >= 0) & (pos_then - pos_now == horizon)

    future = close.to_numpy(dtype=float)[np.where(complete, pos_then, 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(future / close.to_numpy(dtype=float))
    thr = threshold(atr_pct.reindex(index).to_numpy(dtype=float), horizon)

    labels = np.full(len(index), FLAT, dtype=np.int8)
    labels[r > thr] = UP
    labels[r < -thr] = DOWN
    valid = complete & ~np.isnan(thr) & np.isfinite(r)
    return pd.Series(pd.array(np.where(valid, labels, 0), dtype="Int8"),
                     index=index).where(valid, pd.NA)
