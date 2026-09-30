"""Candle-shape features (spec section 4.6).

Pure calculation: no database, no clock. Results are float64 on the input
index. Every fraction is NaN when the bar's range is 0; the ATR-scaled
columns are NaN when atr_14 is 0 or missing. atr_14 is passed in (the
pipeline supplies indicators.atr) so it is computed once.
"""

import pandas as pd

from features.indicators import _safe_div


def compute(bars: pd.DataFrame, atr_14: pd.Series) -> pd.DataFrame:
    open_, high, low, close = bars["open"], bars["high"], bars["low"], bars["close"]
    rng = high - low
    body = close - open_
    out = pd.DataFrame(index=bars.index)
    out["body_frac"] = _safe_div(body, rng)
    out["upper_wick_frac"] = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), rng)
    out["lower_wick_frac"] = _safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, rng)
    out["open_pos"] = _safe_div(open_ - low, rng)
    out["close_pos"] = _safe_div(close - low, rng)
    out["body_atr"] = _safe_div(body, atr_14)
    out["size_atr"] = _safe_div(rng, atr_14)
    return out.astype("float64")
