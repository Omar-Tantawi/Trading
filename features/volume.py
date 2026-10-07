"""Volume features (spec section 4.5).

Pure calculation: no database, no clock. Results are float64 on the input
index; divide-by-zero gives NaN, never inf.
"""

import numpy as np
import pandas as pd

from features.indicators import _safe_div
from features.price import ratio_to_previous_mean


def compute(bars: pd.DataFrame) -> pd.DataFrame:
    vol, taker = bars["volume"], bars["taker_buy_base"]
    out = pd.DataFrame(index=bars.index)
    out["volume_ratio_20"] = ratio_to_previous_mean(vol, 20)

    previous = vol.shift(1)
    out["volume_change"] = np.log(_safe_div(vol.where(vol > 0), previous))

    prev_window = vol.shift(1).rolling(20, min_periods=20)
    out["volume_z_20"] = _safe_div(vol - prev_window.mean(), prev_window.std(ddof=1))

    out["buy_volume"] = taker
    out["sell_volume"] = vol - taker
    out["buy_share"] = _safe_div(taker, vol)
    return out.astype("float64")
