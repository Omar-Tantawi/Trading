"""The model's input rows (spec section 4.1).

Pure: no database, no clock. One row per (symbol, 1h bar) with a label. The
columns are the scale-free 1h features, the same features of the 4h and 1d
bars that had closed by the decision time, one-hot regimes and the symbol.
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from features.pipeline import FEATURE_COLUMNS, TEXT_COLUMNS

# Price levels and base-unit volumes: drawn on charts, never model inputs
# (2a spec section 3.2).
EXCLUDED = frozenset({
    "ema_9", "ema_20", "ema_50", "ema_100", "ema_200",
    "sma_20", "sma_50", "sma_200", "atr_14", "bb_upper", "bb_lower",
    "swing_high", "swing_low", "buy_volume", "sell_volume",
})
MODEL_FEATURES: tuple[str, ...] = tuple(
    c for c in FEATURE_COLUMNS if c not in EXCLUDED and c not in TEXT_COLUMNS)

TREND_REGIMES = ("sideways", "weak_bullish", "weak_bearish",
                 "strong_bullish", "strong_bearish")
VOL_REGIMES = ("low", "normal", "high", "extreme")
_REGIMES = {"trend_regime": TREND_REGIMES, "volatility_regime": VOL_REGIMES}

_HOUR = pd.Timedelta(hours=1)
_PREFIXES = ("", "h4_", "d1_")


@dataclass
class SymbolData:
    close: pd.Series        # 1h closes on a UTC open_time index
    f1h: pd.DataFrame       # FEATURE_COLUMNS, as features.store.read_features
    f4h: pd.DataFrame
    f1d: pd.DataFrame
    high: pd.Series | None = None   # 1h highs and lows (needed by vol3)
    low: pd.Series | None = None


@dataclass
class Dataset:
    X: pd.DataFrame          # float64, RangeIndex
    y: np.ndarray            # int: 0 down, 1 flat, 2 up
    tau: pd.DatetimeIndex    # decision time = open_time + 1h
    symbol: np.ndarray
    open_time: pd.DatetimeIndex


def asof_join(base_index: pd.DatetimeIndex, other: pd.DataFrame,
              other_step: timedelta, prefix: str) -> pd.DataFrame:
    """For each 1h bar t, the `other` row with the largest T such that
    T + other_step <= t + 1h: the newest coarser bar closed by the decision
    time. Columns get `prefix`; NaN/None where none has closed yet."""
    left = pd.DataFrame({"_tau": base_index + _HOUR})
    right = other.copy()
    right.columns = [prefix + c for c in right.columns]
    right["_avail"] = other.index + pd.Timedelta(other_step)
    right = right.reset_index(drop=True).sort_values("_avail")
    left["_tau"] = left["_tau"].astype(right["_avail"].dtype)
    merged = pd.merge_asof(left, right, left_on="_tau", right_on="_avail",
                           direction="backward", allow_exact_matches=True)
    merged.index = base_index
    return merged.drop(columns=["_tau", "_avail"])


def symbol_features(sd: SymbolData) -> pd.DataFrame:
    """Model features and regime text columns of all three timeframes, on
    the 1h index."""
    keep = list(MODEL_FEATURES) + list(TEXT_COLUMNS)
    index = sd.f1h.index
    parts = [
        sd.f1h[keep],
        asof_join(index, sd.f4h[keep], timedelta(hours=4), "h4_"),
        asof_join(index, sd.f1d[keep], timedelta(days=1), "d1_"),
    ]
    return pd.concat(parts, axis=1)


def encode(frame: pd.DataFrame, symbol: str,
           symbols: tuple[str, ...]) -> pd.DataFrame:
    """All-float design matrix: numbers as float64, regimes one-hot over fixed
    label lists, the symbol one-hot over `symbols`."""
    out = {}
    for col in frame.columns:
        base = col.removeprefix("h4_").removeprefix("d1_")
        if base in _REGIMES:
            values = frame[col].astype(object)
            for label in _REGIMES[base]:
                out[f"{col}={label}"] = (values == label).astype("float64")
        else:
            out[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    for s in symbols:
        out[f"symbol={s}"] = np.full(len(frame), float(s == symbol))
    return pd.DataFrame(out, index=frame.index)


def build_dataset(data: dict[str, SymbolData], horizon: int,
                  symbols: tuple[str, ...] | None = None,
                  target=None) -> Dataset:
    """Labelled rows of every symbol for `target` (default move3), sorted by
    (tau, symbol)."""
    from ml.targets import get_target

    target = target or get_target("move3")
    symbols = tuple(symbols or data)
    frames = []
    for symbol, sd in data.items():
        labels = target.labels(sd, horizon)
        X = encode(symbol_features(sd), symbol, symbols)
        keep = labels.notna().to_numpy()
        X = X[keep]
        X.insert(0, "_y", labels[keep].astype(int).to_numpy())
        X.insert(0, "_symbol", symbol)
        frames.append(X)
    full = pd.concat(frames)
    full["_open_time"] = full.index
    full = full.sort_values(["_open_time", "_symbol"], kind="stable")
    open_time = pd.DatetimeIndex(full.pop("_open_time"), name="open_time")
    symbol_col = full.pop("_symbol").to_numpy()
    y = full.pop("_y").to_numpy(dtype=int)
    full = full.reset_index(drop=True)
    return Dataset(X=full, y=y, tau=open_time + _HOUR, symbol=symbol_col,
                   open_time=open_time)
