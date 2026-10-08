"""Synthetic market data for the ML tests: hourly bars run through the real
2a feature pipeline, with 4h and 1d bars aggregated from the same hours."""
from datetime import timedelta

import numpy as np
import pandas as pd

from features.pipeline import compute_features
from ml.dataset import SymbolData

SIGMA = 0.006


def _bars(log_ret: np.ndarray, share: np.ndarray, start: pd.Timestamp,
          rng: np.random.Generator) -> pd.DataFrame:
    n = len(log_ret)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    top = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, SIGMA / 2, n)))
    bottom = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, SIGMA / 2, n)))
    volume = rng.lognormal(3.0, 0.5, n)
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="open_time")
    return pd.DataFrame({"open": open_, "high": top, "low": bottom, "close": close,
                         "volume": volume, "taker_buy_base": volume * share},
                        index=index)


def _resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = bars.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "taker_buy_base": "sum"})
    return out.dropna()


def symbol_from_bars(bars: pd.DataFrame) -> SymbolData:
    b4, b1d = _resample(bars, "4h"), _resample(bars, "1D")
    return SymbolData(
        close=bars["close"], high=bars["high"], low=bars["low"],
        f1h=compute_features(bars, timedelta(hours=1)),
        f4h=compute_features(b4, timedelta(hours=4)),
        f1d=compute_features(b1d, timedelta(days=1)),
    )


def random_walk_bars(seed: int, start: str, hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    share = rng.uniform(0.3, 0.7, hours)
    return _bars(rng.normal(0, SIGMA, hours), share, pd.Timestamp(start, tz="UTC"), rng)


def random_walk_symbol(seed: int, start: str, hours: int) -> SymbolData:
    return symbol_from_bars(random_walk_bars(seed, start, hours))


def planted_signal_symbol(seed: int, start: str, hours: int,
                          horizon: int) -> SymbolData:
    """The buyer share of bar t pushes the drift of the next `horizon` hours:
    information a model can only find through the buy_share feature."""
    rng = np.random.default_rng(seed)
    share = rng.uniform(0.3, 0.7, hours)
    drift = np.zeros(hours)
    for k in range(1, horizon + 1):
        drift[k:] += 0.015 * (share[:-k] - 0.5)
    log_ret = rng.normal(0, SIGMA, hours) + drift
    return symbol_from_bars(_bars(log_ret, share, pd.Timestamp(start, tz="UTC"), rng))
