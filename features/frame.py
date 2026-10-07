"""Loading closed bars for the feature build (spec sections 2, 3.5 and 5.2).

The only place a stored Decimal price becomes a float: every column is cast
to float8 in SQL, so nothing else in the feature engine ever sees a Decimal.
"""
from datetime import datetime, timedelta

import pandas as pd

from data.quality.checks import STEP
from data.storage.repository import _floor
from features.pipeline import FEATURE_TIMEFRAMES

BAR_COLUMNS = ("open", "high", "low", "close", "volume", "taker_buy_base")

_ONE_MINUTE = timedelta(minutes=1)


def _check_aware(dt: datetime, label: str) -> None:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{label} must be timezone-aware UTC, got {dt!r}")


def _step(timeframe: str) -> timedelta:
    if timeframe not in FEATURE_TIMEFRAMES:
        raise ValueError(f"unknown feature timeframe {timeframe!r}")
    return STEP[timeframe]


def build_cutoff(conn, symbol: str, timeframe: str) -> datetime | None:
    """Exclusive upper bound on the open_time of a buildable bar: the last
    stored 1m candle's end, floored to the timeframe. A bar is eligible only
    when its whole span is covered by stored 1m data (spec section 5.2), so
    the bucket still forming is never built. None if the symbol has no 1m
    candles.
    """
    step = _step(timeframe)
    with conn.cursor() as cur:
        cur.execute("SELECT max(open_time) FROM candles_1m WHERE symbol = %s",
                    (symbol,))
        last_1m = cur.fetchone()[0]
    if last_1m is None:
        return None
    return _floor(last_1m + _ONE_MINUTE, step)


def load_bars(conn, symbol: str, timeframe: str, start: datetime | None,
              end: datetime) -> pd.DataFrame:
    """Bars from candles_<timeframe> with start <= open_time < end (no lower
    bound when `start` is None), as float64 OHLCV plus taker_buy_base on a
    UTC DatetimeIndex named open_time.
    """
    _step(timeframe)
    _check_aware(end, "end")
    where, params = ["symbol = %s", "open_time < %s"], [symbol, end]
    if start is not None:
        _check_aware(start, "start")
        where.append("open_time >= %s")
        params.append(start)
    casts = ", ".join(f"{c}::float8" for c in BAR_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT open_time, {casts} FROM candles_{timeframe} "
            f"WHERE {' AND '.join(where)} ORDER BY open_time",
            params,
        )
        rows = cur.fetchall()

    columns = list(zip(*rows)) if rows else [()] * (len(BAR_COLUMNS) + 1)
    index = pd.DatetimeIndex(pd.to_datetime(list(columns[0]), utc=True),
                             name="open_time")
    out = pd.DataFrame(
        {c: pd.Series(list(v), dtype="float64")
         for c, v in zip(BAR_COLUMNS, columns[1:])})
    out.index = index
    return out
