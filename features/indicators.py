"""Technical indicators (spec section 4.2).

Pure calculation: no database, no clock. Every function takes float Series on a
shared index and returns float64 Series on that same index. Divide-by-zero
gives NaN, never inf. Warm-up blanking is not done here (the pipeline does it).
"""

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b, with NaN wherever b is 0 (or either side is NaN)."""
    out = a / b.where(b != 0)
    return out.replace([np.inf, -np.inf], np.nan)


def ema(s: pd.Series, n: int) -> pd.Series:
    """Exponential moving average, alpha = 2 / (n + 1), seeded with the first
    valid value (leading NaNs are skipped)."""
    return s.ewm(span=n, adjust=False).mean().astype("float64")


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing: the EMA recursion with alpha = 1 / n."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean().astype("float64")


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilder(delta.clip(lower=0), n)
    avg_loss = wilder((-delta).clip(lower=0), n)
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    out = 100 - 100 / (1 + rs)
    no_loss = (avg_loss == 0) & avg_gain.notna()
    out = out.mask(no_loss & (avg_gain > 0), 100.0)
    out = out.mask(no_loss & (avg_gain == 0), 50.0)
    return out.astype("float64")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """max(high - low, |high - prev close|, |low - prev close|); at bar 0 the
    missing previous close drops out, leaving high - low."""
    prev_close = close.shift(1)
    parts = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return parts.max(axis=1).astype("float64")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def dmi(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (plus_di, minus_di, adx)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm = plus_dm.where(up.notna() & down.notna())
    minus_dm = minus_dm.where(up.notna() & down.notna())

    atr_n = atr(high, low, close, n)
    plus_di = 100 * _safe_div(wilder(plus_dm, n), atr_n)
    minus_di = 100 * _safe_div(wilder(minus_dm, n), atr_n)

    di_sum = plus_di + minus_di
    dx = 100 * _safe_div((plus_di - minus_di).abs(), di_sum)
    dx = dx.mask((di_sum == 0), 0.0)
    adx = wilder(dx, n)
    return plus_di.astype("float64"), minus_di.astype("float64"), adx


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Raw (price-unit) MACD line, signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Returns (upper, lower): sma +/- k population standard deviations."""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    return (mid + k * std).astype("float64"), (mid - k * std).astype("float64")


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Returns (%K, %D). %K is NaN when the n-bar range is 0."""
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    k = 100 * _safe_div(close - lowest, highest - lowest)
    return k.astype("float64"), k.rolling(d).mean().astype("float64")


def compute(bars: pd.DataFrame) -> pd.DataFrame:
    """The 32 spec section 4.2 columns, in spec order, without warm-up blanking."""
    close, high, low = bars["close"], bars["high"], bars["low"]
    out: dict[str, pd.Series] = {}

    emas = {n: ema(close, n) for n in (9, 20, 50, 100, 200)}
    for n, s in emas.items():
        out[f"ema_{n}"] = s
    for n, s in emas.items():
        out[f"dist_ema_{n}"] = _safe_div(close, s) - 1

    smas = {n: close.rolling(n).mean() for n in (20, 50, 200)}
    for n, s in smas.items():
        out[f"sma_{n}"] = s
    for n, s in smas.items():
        out[f"dist_sma_{n}"] = _safe_div(close, s) - 1

    out["rsi_14"] = rsi(close, 14)

    line, sig, hist = macd(close)
    out["macd_pct"] = _safe_div(line, close)
    out["macd_signal_pct"] = _safe_div(sig, close)
    out["macd_hist_pct"] = _safe_div(hist, close)

    atr_14 = atr(high, low, close, 14)
    out["atr_14"] = atr_14
    out["atr_pct"] = _safe_div(atr_14, close)

    plus_di, minus_di, adx = dmi(high, low, close, 14)
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    out["adx_14"] = adx

    upper, lower = bollinger(close, 20, 2.0)
    width = upper - lower
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_pct_b"] = _safe_div(close - lower, width)
    out["bb_width"] = _safe_div(width, smas[20])

    k, d = stochastic(high, low, close, 14, 3)
    out["stoch_k_14"] = k
    out["stoch_d_3"] = d

    out["roc_10"] = _safe_div(close, close.shift(10)) - 1

    return pd.DataFrame(out, index=bars.index).astype("float64")
