"""Rules-based market regime (spec section 4.7).

Pure calculation: no database, no clock. Two independent labels, because a
market can be sideways and extremely volatile at once. The thresholds are named
constants; sub-project 3 measures whether they carry signal. Labels are plain
strings in object columns, and None where an input is missing.
"""

import numpy as np
import pandas as pd

ADX_TREND_MIN = 20.0
ADX_STRONG_MIN = 25.0
VOL_LOW_MAX = 20.0
VOL_NORMAL_MAX = 80.0
VOL_HIGH_MAX = 95.0

COLUMNS = ["trend_regime", "volatility_regime"]


def _labels(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Object array of plain Python str, None where an input was missing."""
    out = values.astype(object)
    out[missing] = None
    return out


def compute(
    adx_14: pd.Series,
    plus_di_14: pd.Series,
    minus_di_14: pd.Series,
    vol_pct_365d: pd.Series,
) -> pd.DataFrame:
    adx = adx_14.to_numpy(dtype="float64")
    plus = plus_di_14.to_numpy(dtype="float64")
    minus = minus_di_14.to_numpy(dtype="float64")
    pct = vol_pct_365d.to_numpy(dtype="float64")

    bullish = plus > minus
    trend = np.select(
        [
            (adx < ADX_TREND_MIN) | (plus == minus),
            adx < ADX_STRONG_MIN,
        ],
        [
            "sideways",
            np.where(bullish, "weak_bullish", "weak_bearish"),
        ],
        default=np.where(bullish, "strong_bullish", "strong_bearish"),
    )
    trend_missing = np.isnan(adx) | np.isnan(plus) | np.isnan(minus)

    volatility = np.select(
        [pct < VOL_LOW_MAX, pct < VOL_NORMAL_MAX, pct < VOL_HIGH_MAX],
        ["low", "normal", "high"],
        default="extreme",
    )

    return pd.DataFrame(
        {
            "trend_regime": pd.Series(
                _labels(trend, trend_missing), index=adx_14.index, dtype=object),
            "volatility_regime": pd.Series(
                _labels(volatility, np.isnan(pct)), index=adx_14.index, dtype=object),
        },
        index=adx_14.index,
    )
