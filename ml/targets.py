"""The questions a model can answer (3b spec section 2).

* move3: down / flat / up at the end of the horizon (sub-project 3).
* vol3:  quiet / normal / wild: the largest move either way during the
         horizon, against the coin's usual move.
* dir2:  down / up at the end of the horizon.

Every target uses the same rows: a label is missing when any 1h bar of the
horizon is missing or atr_pct at the decision bar is NULL.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ml.labels import make_labels

VOL_QUIET = 0.5
VOL_WILD = 1.25

_HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class Target:
    name: str
    classes: tuple[str, ...]
    label_set: int
    labels: Callable          # (SymbolData, horizon) -> Int8 Series

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def _complete(index: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """True where every 1h bar from t+1h to t+H·1h is stored."""
    pos_then = index.get_indexer(index + horizon * _HOUR)
    return (pos_then >= 0) & (pos_then - np.arange(len(index)) == horizon)


def _as_labels(values: np.ndarray, valid: np.ndarray, index) -> pd.Series:
    return pd.Series(pd.array(np.where(valid, values, 0), dtype="Int8"),
                     index=index).where(valid, pd.NA)


def vol3_labels(close: pd.Series, high: pd.Series, low: pd.Series,
                atr_pct: pd.Series, horizon: int) -> pd.Series:
    index = close.index
    ok = _complete(index, horizon)
    c0 = close.to_numpy(dtype=float)
    # max high / min low over the next `horizon` bars (positions i+1 … i+H;
    # positional is exact where `ok`, since those bars are consecutive)
    fut_high = high.shift(-1).rolling(horizon, min_periods=horizon).max() \
        .shift(-(horizon - 1)).to_numpy(dtype=float)
    fut_low = low.shift(-1).rolling(horizon, min_periods=horizon).min() \
        .shift(-(horizon - 1)).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.maximum(np.log(fut_high / c0), np.log(c0 / fut_low))
    unit = atr_pct.reindex(index).to_numpy(dtype=float) * np.sqrt(horizon)
    values = np.where(move < VOL_QUIET * unit, 0,
                      np.where(move < VOL_WILD * unit, 1, 2))
    valid = ok & np.isfinite(move) & np.isfinite(unit)
    return _as_labels(values, valid, index)


def dir2_labels(close: pd.Series, atr_pct: pd.Series, horizon: int) -> pd.Series:
    index = close.index
    ok = _complete(index, horizon)
    c = close.to_numpy(dtype=float)
    future = c[np.where(ok, index.get_indexer(index + horizon * _HOUR), 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(future / c)
    atr = atr_pct.reindex(index).to_numpy(dtype=float)
    valid = ok & np.isfinite(r) & (r != 0) & np.isfinite(atr)
    return _as_labels((r > 0).astype(int), valid, index)


def _sd_index(sd):
    return sd.f1h.index


TARGETS: dict[str, Target] = {
    "move3": Target("move3", ("down", "flat", "up"), 1,
                    lambda sd, h: make_labels(sd.close.reindex(_sd_index(sd)),
                                              sd.f1h["atr_pct"], h)),
    "vol3": Target("vol3", ("quiet", "normal", "wild"), 1,
                   lambda sd, h: vol3_labels(sd.close.reindex(_sd_index(sd)),
                                             sd.high.reindex(_sd_index(sd)),
                                             sd.low.reindex(_sd_index(sd)),
                                             sd.f1h["atr_pct"], h)),
    "dir2": Target("dir2", ("down", "up"), 1,
                   lambda sd, h: dir2_labels(sd.close.reindex(_sd_index(sd)),
                                             sd.f1h["atr_pct"], h)),
}


def get_target(name: str) -> Target:
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(f"unknown target {name!r}; expected one of "
                         + ", ".join(TARGETS)) from None
