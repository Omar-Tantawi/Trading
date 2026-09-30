"""Reading and writing the feature tables (spec 5.1, 5.3).

The only module of the feature engine that talks to the database for
features. Like data/storage/repository.py, its write function does not commit:
the caller owns the transaction.

A missing feature is SQL NULL in all three of its pandas forms: NaN in
float64, pd.NA in Int64 and None in text. NaN is never stored as 'NaN'.
"""
from datetime import datetime

import pandas as pd

from features.pipeline import (
    FEATURE_COLUMNS, FEATURE_SET, FEATURE_TIMEFRAMES, INT32_COLUMNS,
    INT_COLUMNS, TEXT_COLUMNS,
)

_COLUMN_LIST = ", ".join(FEATURE_COLUMNS)
_INT_LIKE = INT_COLUMNS | INT32_COLUMNS


def feature_table(timeframe: str) -> str:
    """The table name for a timeframe. The only place a name reaches SQL."""
    if timeframe not in FEATURE_TIMEFRAMES:
        raise ValueError(f"unknown feature timeframe {timeframe!r}")
    return f"features_{timeframe}"


def _check_aware(dt: datetime, label: str) -> None:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{label} must be timezone-aware UTC, got {dt!r}")


def _sql_values(col: str, series: pd.Series) -> list:
    """The column as Python values, with every missing value as None."""
    present = series.notna()
    if col in _INT_LIKE:
        return [int(v) if ok else None
                for v, ok in zip(series.astype(object), present)]
    values = series.astype(object).where(present, None).tolist()
    if col not in TEXT_COLUMNS and any(v in (float("inf"), float("-inf"))
                                       for v in values if v is not None):
        raise ValueError(f"{col} holds an infinite value; a feature is "
                         "NaN, never infinite")
    return values


def upsert_features(conn, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
    """Write `frame` (FEATURE_COLUMNS on a UTC open_time index) for one
    symbol and timeframe: COPY into a temporary staging table, then
    INSERT ... ON CONFLICT (symbol, open_time) DO UPDATE, so re-running is
    always safe. Every row is stamped with FEATURE_SET. Does not commit.
    Returns the rows written.
    """
    table = feature_table(timeframe)
    if frame.empty:
        return 0
    if frame.index.tz is None:
        raise ValueError("frame index must be timezone-aware UTC")
    data = frame[list(FEATURE_COLUMNS)]  # KeyError if a column is missing

    n = len(data)
    columns = {col: _sql_values(col, data[col]) for col in FEATURE_COLUMNS}
    times = data.index.to_pydatetime()
    rows = zip([symbol] * n, times, [FEATURE_SET] * n,
               *(columns[c] for c in FEATURE_COLUMNS))

    cols = f"symbol, open_time, feature_set, {_COLUMN_LIST}"
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in ("feature_set", *FEATURE_COLUMNS))
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS staging_features "
            f"(LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        cur.execute("TRUNCATE staging_features")
        with cur.copy(f"COPY staging_features ({cols}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute(
            f"""
            INSERT INTO {table} ({cols})
            SELECT {cols} FROM staging_features
            ON CONFLICT (symbol, open_time) DO UPDATE SET
                {updates}, computed_at = now()
            """
        )
        return cur.rowcount


def last_built(conn, symbol: str, timeframe: str) -> datetime | None:
    """The newest stored open_time for (symbol, timeframe), or None."""
    table = feature_table(timeframe)
    with conn.cursor() as cur:
        cur.execute(f"SELECT max(open_time) FROM {table} WHERE symbol = %s",
                    (symbol,))
        return cur.fetchone()[0]


def has_stale_feature_set(conn, symbol: str, timeframe: str) -> bool:
    """True if any stored row was produced by a different FEATURE_SET."""
    table = feature_table(timeframe)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT EXISTS (SELECT 1 FROM {table} "
            f"WHERE symbol = %s AND feature_set <> %s)",
            (symbol, FEATURE_SET),
        )
        return cur.fetchone()[0]


def _typed_column(col: str, values) -> pd.Series:
    """One column, in the dtype compute_features produces for it."""
    if col in _INT_LIKE:
        return pd.Series(pd.array(list(values), dtype="Int64"))
    if col in TEXT_COLUMNS:
        return pd.Series(list(values), dtype=object)
    return pd.Series(list(values), dtype="float64")


def read_features(conn, symbol: str, timeframe: str,
                  start: datetime | None = None,
                  end: datetime | None = None) -> pd.DataFrame:
    """FEATURE_COLUMNS for [start, end) on a UTC index named open_time, in the
    dtypes compute_features produces: float64, Int64 and object (str/None).
    NULL comes back as NaN, pd.NA or None."""
    table = feature_table(timeframe)
    where, params = ["symbol = %s"], [symbol]
    if start is not None:
        _check_aware(start, "start")
        where.append("open_time >= %s")
        params.append(start)
    if end is not None:
        _check_aware(end, "end")
        where.append("open_time < %s")
        params.append(end)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT open_time, {_COLUMN_LIST} FROM {table} "
            f"WHERE {' AND '.join(where)} ORDER BY open_time",
            params,
        )
        rows = cur.fetchall()

    columns = list(zip(*rows)) if rows else [()] * (len(FEATURE_COLUMNS) + 1)
    index = pd.DatetimeIndex(pd.to_datetime(list(columns[0]), utc=True),
                             name="open_time")
    out = pd.DataFrame(
        {col: _typed_column(col, values)
         for col, values in zip(FEATURE_COLUMNS, columns[1:])})
    out.index = index
    return out


def latest_features(conn, symbol: str, timeframe: str) -> dict | None:
    """The newest row as a dict: open_time plus every feature column, with
    NULL as None. None if nothing is stored."""
    table = feature_table(timeframe)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT open_time, {_COLUMN_LIST} FROM {table} "
            f"WHERE symbol = %s ORDER BY open_time DESC LIMIT 1",
            (symbol,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return dict(zip(("open_time", *FEATURE_COLUMNS), row))
