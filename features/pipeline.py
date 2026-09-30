"""The feature pipeline (spec sections 3, 4 and 5.3).

Pure calculation: no database, no clock. Composes the calculation modules and
gives the guarantees on top of them:

* Warm-up blanking. Every column is set to NaN for its first WARMUP[col] bars,
  and each module's outputs are blanked *before* a later module sees them, so
  dependants never read a warm-up value.
* No look-ahead. Every step looks only backwards (swing points count from their
  confirmation bar), so a row is the same whether or not later bars exist.
* Incremental equals full. A window starting at `lookback_start` gives the same
  rows as the full history, past the window's first bars (spec section 5.3).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from features import candles, indicators, price, regime, trend, volatility, volume

FEATURE_SET = 1
FEATURE_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")

FEATURE_COLUMNS: tuple[str, ...] = (
    # price
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "dist_high_20", "dist_high_100", "dist_low_20", "dist_low_100",
    "accel_6", "range_ratio_20",
    # indicators
    "ema_9", "ema_20", "ema_50", "ema_100", "ema_200",
    "dist_ema_9", "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200",
    "sma_20", "sma_50", "sma_200", "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "atr_14", "atr_pct", "plus_di_14", "minus_di_14", "adx_14",
    "bb_upper", "bb_lower", "bb_pct_b", "bb_width",
    "stoch_k_14", "stoch_d_3", "roc_10",
    # trend
    "swing_high", "swing_low", "dist_swing_high", "dist_swing_low",
    "swing_high_dir", "swing_low_dir", "structure",
    "ema_slope_20", "ema_slope_50", "trend_duration", "trend_accel",
    "bos", "bars_since_bos",
    # volatility
    "ret_std_20", "hist_vol_20", "range_pct", "vol_pct_365d",
    # volume
    "volume_ratio_20", "volume_change", "volume_z_20",
    "buy_volume", "sell_volume", "buy_share",
    # candle shape
    "body_frac", "upper_wick_frac", "lower_wick_frac", "open_pos", "close_pos",
    "body_atr", "size_atr",
    # regime
    "trend_regime", "volatility_regime",
)

INT_COLUMNS = frozenset({"swing_high_dir", "swing_low_dir", "structure", "bos"})  # smallint
INT32_COLUMNS = frozenset({"trend_duration", "bars_since_bos"})  # integer
TEXT_COLUMNS = frozenset({"trend_regime", "volatility_regime"})


def _build_warmup() -> dict[str, int]:
    """Warm-up in bars per column, copied from spec section 4."""
    w: dict[str, int] = {}
    # 4.1 price
    for n in price.RETURN_WINDOWS:
        w[f"ret_{n}"] = n
    for n in price.DIST_WINDOWS:
        w[f"dist_high_{n}"] = n - 1
        w[f"dist_low_{n}"] = n - 1
    w["accel_6"] = 12
    w["range_ratio_20"] = 20
    # 4.2 indicators
    for n in (9, 20, 50, 100, 200):
        w[f"ema_{n}"] = 3 * n
        w[f"dist_ema_{n}"] = 3 * n
    for n in (20, 50, 200):
        w[f"sma_{n}"] = n - 1
        w[f"dist_sma_{n}"] = n - 1
    w["rsi_14"] = 42
    w["macd_pct"] = 78
    w["macd_signal_pct"] = 105
    w["macd_hist_pct"] = 105
    for c in ("atr_14", "atr_pct", "plus_di_14", "minus_di_14"):
        w[c] = 42
    w["adx_14"] = 84
    for c in ("bb_upper", "bb_lower", "bb_pct_b", "bb_width"):
        w[c] = 19
    w["stoch_k_14"] = 13
    w["stoch_d_3"] = 15
    w["roc_10"] = 10
    # 4.3 trend: "-" means no fixed warm-up (NULL until the inputs exist)
    for c in ("swing_high", "swing_low", "dist_swing_high", "dist_swing_low",
              "swing_high_dir", "swing_low_dir", "structure", "bos", "bars_since_bos"):
        w[c] = 0
    w["ema_slope_20"] = 3 * 20 + 5
    w["ema_slope_50"] = 3 * 50 + 5
    w["trend_duration"] = 150
    w["trend_accel"] = 70
    # 4.4 volatility
    w["ret_std_20"] = 20
    w["hist_vol_20"] = 20
    w["range_pct"] = 0
    w["vol_pct_365d"] = 0  # time-based
    # 4.5 volume
    w["volume_ratio_20"] = 20
    w["volume_change"] = 1
    w["volume_z_20"] = 20
    for c in ("buy_volume", "sell_volume", "buy_share"):
        w[c] = 0
    # 4.6 candle shape
    for c in ("body_frac", "upper_wick_frac", "lower_wick_frac", "open_pos", "close_pos"):
        w[c] = 0
    w["body_atr"] = 42
    w["size_atr"] = 42
    # 4.7 regime: the spec gives no separate warm-up. trend_regime needs adx_14
    # (the slowest of its inputs); volatility_regime follows the time-based
    # vol_pct_365d.
    w["trend_regime"] = 84
    w["volatility_regime"] = 0
    return w


WARMUP: dict[str, int] = _build_warmup()

# Columns whose NULL span is not a fixed number of bars: "-" in the spec, or
# based on time rather than bars. WARMUP holds 0 for them.
DATA_DEPENDENT_WARMUP: frozenset[str] = frozenset({
    "swing_high", "swing_low", "dist_swing_high", "dist_swing_low",
    "swing_high_dir", "swing_low_dir", "structure", "bos", "bars_since_bos",
    "vol_pct_365d", "volatility_regime",
})

# Recursive features need their seed to decay (EMA 200 needs about 2,070 bars),
# the 365-day percentile window must be complete, and swing and duration
# counters must see their history.
LOOKBACK_BARS = 2100
LOOKBACK_DAYS = 400


def lookback_start(last_built: datetime, step: timedelta) -> datetime:
    """Where an incremental build starts loading bars (spec section 5.3)."""
    return last_built - max(LOOKBACK_BARS * step, timedelta(days=LOOKBACK_DAYS))


def _missing_value(col: str):
    if col in TEXT_COLUMNS:
        return None
    if col in INT_COLUMNS | INT32_COLUMNS:
        return pd.NA
    return np.nan


def _blank(frame: pd.DataFrame) -> pd.DataFrame:
    """Set each column's first WARMUP[col] rows to missing (NaN, NA or None)."""
    for col in frame.columns:
        w = min(WARMUP[col], len(frame))
        if w > 0:
            frame.iloc[:w, frame.columns.get_loc(col)] = _missing_value(col)
    return frame


def _empty() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="open_time", tz="UTC")
    columns: dict[str, pd.Series] = {}
    for col in FEATURE_COLUMNS:
        if col in INT_COLUMNS | INT32_COLUMNS:
            dtype = "Int64"
        elif col in TEXT_COLUMNS:
            dtype = object
        else:
            dtype = "float64"
        columns[col] = pd.Series([], index=index, dtype=dtype)
    return pd.DataFrame(columns, index=index)


def compute_features(bars: pd.DataFrame, step: timedelta) -> pd.DataFrame:
    """All FEATURE_COLUMNS, in order, on bars.index.

    `bars` holds float64 open, high, low, close, volume, taker_buy_base on a
    UTC DatetimeIndex; `step` is the bar length.
    """
    if bars.empty:
        return _empty()

    ind = _blank(indicators.compute(bars))
    pri = _blank(price.compute(bars))
    vol = _blank(volume.compute(bars))
    can = _blank(candles.compute(bars, ind["atr_14"]))
    vlt = _blank(volatility.compute(bars, pri["ret_1"], ind["atr_pct"], step))
    trd = _blank(trend.compute(bars, ind["ema_20"], ind["ema_50"]))
    reg = _blank(regime.compute(
        ind["adx_14"], ind["plus_di_14"], ind["minus_di_14"], vlt["vol_pct_365d"]))

    out = pd.concat([pri, ind, trd, vlt, vol, can, reg], axis=1)
    return out[list(FEATURE_COLUMNS)]
