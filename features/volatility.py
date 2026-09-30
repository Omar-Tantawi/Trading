"""Volatility features (spec section 4.4).

Pure calculation: no database, no clock. `ret_1` and `atr_pct` arrive from the
pipeline already warm-up blanked, so NaN inputs are normal and every output is
NaN wherever its inputs are. `bb_width` and `atr_pct` are volatility features
too, but they are computed once, in the indicator step, and not stored twice.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from features.indicators import _safe_div

VOL_PCT_WINDOW = timedelta(days=365)
VOL_PCT_MIN_HISTORY = timedelta(days=30)
STD_WINDOW = 20
MINUTES_PER_YEAR = 525_600  # crypto trades 24/7

COLUMNS = ["ret_std_20", "hist_vol_20", "range_pct", "vol_pct_365d"]


def bars_per_year(step: timedelta) -> float:
    return MINUTES_PER_YEAR / (step.total_seconds() / 60.0)


def vol_percentile(atr_pct: pd.Series) -> pd.Series:
    """Percentile (0-100) of atr_pct at t among the non-NaN atr_pct values with
    open_time in (t - 365 days, t]: 100 * count(values <= current) / count(values).

    NaN while the earliest non-NaN atr_pct is less than 30 days older than t,
    and wherever atr_pct itself is NaN. The window is time-based, so a gap in
    the data shrinks it rather than reaching further back.
    """
    window = atr_pct.rolling(VOL_PCT_WINDOW, closed="right", min_periods=1)
    at_or_below = window.rank(method="max")  # NaN where the current value is NaN
    count = window.count()
    pct = 100.0 * _safe_div(at_or_below, count)

    first_valid = atr_pct.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=atr_pct.index, dtype="float64")
    history = atr_pct.index - first_valid
    return pct.where(history >= VOL_PCT_MIN_HISTORY).astype("float64")


def compute(
    bars: pd.DataFrame, ret_1: pd.Series, atr_pct: pd.Series, step: timedelta
) -> pd.DataFrame:
    out = pd.DataFrame(index=bars.index)
    out["ret_std_20"] = ret_1.rolling(STD_WINDOW, min_periods=STD_WINDOW).std()
    out["hist_vol_20"] = out["ret_std_20"] * np.sqrt(bars_per_year(step))
    out["range_pct"] = _safe_div(bars["high"] - bars["low"], bars["close"])
    out["vol_pct_365d"] = vol_percentile(atr_pct)
    return out.astype("float64")
