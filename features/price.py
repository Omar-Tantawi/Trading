"""Price features (spec section 4.1).

Pure calculation: no database, no clock. Results are float64 on the input
index. Window features use min_periods equal to the window, so they are NaN
until the window is full; further warm-up blanking is the pipeline's job.
"""

import numpy as np
import pandas as pd

from features.indicators import _safe_div

RETURN_WINDOWS = (1, 3, 6, 12, 24)
DIST_WINDOWS = (20, 100)


def log_return(close: pd.Series, n: int) -> pd.Series:
    """ln(close_t / close_{t-n})."""
    return np.log(_safe_div(close, close.shift(n))).astype("float64")


def dist_high(close: pd.Series, high: pd.Series, n: int) -> pd.Series:
    """close_t / max(high over the n-bar window incl. t) - 1 (<= 0)."""
    return (_safe_div(close, high.rolling(n, min_periods=n).max()) - 1).astype("float64")


def dist_low(close: pd.Series, low: pd.Series, n: int) -> pd.Series:
    """close_t / min(low over the n-bar window incl. t) - 1 (>= 0)."""
    return (_safe_div(close, low.rolling(n, min_periods=n).min()) - 1).astype("float64")


def ratio_to_previous_mean(s: pd.Series, n: int) -> pd.Series:
    """s_t / mean(s over the previous n bars, excluding t); NaN when that mean
    is 0. Shared by range_ratio_20 and volume_ratio_20."""
    previous_mean = s.shift(1).rolling(n, min_periods=n).mean()
    return _safe_div(s, previous_mean).astype("float64")


def compute(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low = bars["close"], bars["high"], bars["low"]
    out = pd.DataFrame(index=bars.index)
    for n in RETURN_WINDOWS:
        out[f"ret_{n}"] = log_return(close, n)
    for n in DIST_WINDOWS:
        out[f"dist_high_{n}"] = dist_high(close, high, n)
    for n in DIST_WINDOWS:
        out[f"dist_low_{n}"] = dist_low(close, low, n)
    out["accel_6"] = out["ret_6"] - out["ret_6"].shift(6)
    out["range_ratio_20"] = ratio_to_previous_mean(high - low, 20)
    return out.astype("float64")
