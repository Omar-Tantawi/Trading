"""Trend structure features (spec section 4.3).

Pure calculation: no database, no clock. A swing point at bar i needs the 3
bars after it, so it is confirmed at bar i + SWING_K and is invisible to every
feature before that bar. Everything here is vectorised NumPy (no per-bar
Python loop), so it stays cheap on very long histories.

Integer columns use the pandas nullable Int64 dtype; the rest are float64.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from features.indicators import _safe_div

SWING_K = 3

COLUMNS = [
    "swing_high", "swing_low", "dist_swing_high", "dist_swing_low",
    "swing_high_dir", "swing_low_dir", "structure",
    "ema_slope_20", "ema_slope_50", "trend_duration", "trend_accel",
    "bos", "bars_since_bos",
]


def _confirmed_swings(values: np.ndarray, *, high: bool) -> np.ndarray:
    """Per bar t, the last two swing prices confirmed at or before t.

    Returns an (n, 2) array [latest, previous] with NaN where fewer than one
    or two swings are confirmed. A swing at i is strictly beyond the k bars on
    each side and is confirmed at i + SWING_K.
    """
    n = len(values)
    k = SWING_K
    out = np.full((n, 2), np.nan)
    if n < 2 * k + 1:
        return out
    windows = sliding_window_view(values, 2 * k + 1)  # windows[j] is centred on j + k
    centre = windows[:, k]
    before = windows[:, :k]
    after = windows[:, k + 1:]
    if high:
        is_swing = (centre > before.max(axis=1)) & (centre > after.max(axis=1))
    else:
        is_swing = (centre < before.min(axis=1)) & (centre < after.min(axis=1))

    confirmed_at = np.flatnonzero(is_swing) + 2 * k  # i + k, with i = j + k
    prices = centre[is_swing]
    # Number of swings confirmed at or before each bar.
    count = np.searchsorted(confirmed_at, np.arange(n), side="right")
    has_one = count >= 1
    has_two = count >= 2
    out[has_one, 0] = prices[count[has_one] - 1]
    out[has_two, 1] = prices[count[has_two] - 2]
    return out


def _direction(latest: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """+1 / -1 / 0 as float with NaN where either swing is missing."""
    return np.where(np.isnan(latest) | np.isnan(previous), np.nan,
                    np.sign(latest - previous))


def _signed_run_length(diff_sign: np.ndarray) -> np.ndarray:
    """Signed length of the current run of equal values in diff_sign
    (+k, -k, or 0 for equal). NaN where diff_sign is NaN; a run begins at the
    first valid value and breaks on any change or gap."""
    n = len(diff_sign)
    valid = ~np.isnan(diff_sign)
    sign = np.where(valid, diff_sign, 2.0)  # sentinel never equals a real sign
    index = np.arange(n)
    starts = np.ones(n, dtype=bool)
    starts[1:] = sign[1:] != sign[:-1]
    run_start = np.maximum.accumulate(np.where(starts, index, 0))
    out = diff_sign * (index - run_start + 1)
    return np.where(valid, out, np.nan)


def _bars_since_nonzero(bos: np.ndarray) -> np.ndarray:
    """Bars since the latest non-zero bos, signed by its direction (0 on the
    bar itself); NaN until one exists. bos may hold NaN."""
    index = np.arange(len(bos))
    fired = np.nan_to_num(bos) != 0
    last = np.maximum.accumulate(np.where(fired, index, -1))
    seen = last >= 0
    direction = np.where(seen, bos[np.where(seen, last, 0)], np.nan)
    return np.where(seen, direction * (index - last), np.nan)


def _as_int(values: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index).astype("Int64")


def compute(bars: pd.DataFrame, ema_20: pd.Series, ema_50: pd.Series) -> pd.DataFrame:
    idx = bars.index
    close = bars["close"]
    close_v = close.to_numpy(dtype="float64")

    highs = _confirmed_swings(bars["high"].to_numpy(dtype="float64"), high=True)
    lows = _confirmed_swings(bars["low"].to_numpy(dtype="float64"), high=False)
    h2, h1 = highs[:, 0], highs[:, 1]
    l2, l1 = lows[:, 0], lows[:, 1]

    high_dir = _direction(h2, h1)
    low_dir = _direction(l2, l1)
    structure = np.where(
        np.isnan(high_dir) | np.isnan(low_dir), np.nan,
        np.where((high_dir == 1) & (low_dir == 1), 1.0,
                 np.where((high_dir == -1) & (low_dir == -1), -1.0, 0.0)),
    )

    # Break of structure: both closes are compared with the H2 (L2) valid at t.
    prev_close = np.concatenate(([np.nan], close_v[:-1]))
    with np.errstate(invalid="ignore"):
        up = (close_v > h2) & (prev_close <= h2)
        down = (close_v < l2) & (prev_close >= l2)
    bos = np.where(up, 1.0, np.where(down, -1.0, 0.0))
    bos = np.where(np.isnan(h2) | np.isnan(l2), np.nan, bos)

    e20 = ema_20.reindex(idx).astype("float64")
    e50 = ema_50.reindex(idx).astype("float64")
    slope_20 = _safe_div(e20, e20.shift(5)) - 1
    slope_50 = _safe_div(e50, e50.shift(5)) - 1
    relation = np.sign((e20 - e50).to_numpy())  # NaN where either EMA is NaN

    out = pd.DataFrame(index=idx)
    out["swing_high"] = pd.Series(h2, index=idx)
    out["swing_low"] = pd.Series(l2, index=idx)
    out["dist_swing_high"] = _safe_div(close, out["swing_high"]) - 1
    out["dist_swing_low"] = _safe_div(close, out["swing_low"]) - 1
    out["swing_high_dir"] = _as_int(high_dir, idx)
    out["swing_low_dir"] = _as_int(low_dir, idx)
    out["structure"] = _as_int(structure, idx)
    out["ema_slope_20"] = slope_20
    out["ema_slope_50"] = slope_50
    out["trend_duration"] = _as_int(_signed_run_length(relation), idx)
    out["trend_accel"] = slope_20 - slope_20.shift(5)
    out["bos"] = _as_int(bos, idx)
    out["bars_since_bos"] = _as_int(_bars_since_nonzero(bos), idx)
    return out[COLUMNS]
