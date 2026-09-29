import logging
import time
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

import psycopg

from data.storage.db import CONNECT_TIMEOUT_SECONDS

log = logging.getLogger(__name__)

CANDLE_COLUMNS = (
    "symbol", "open_time", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "trade_count", "taker_buy_base",
    "taker_buy_quote", "source",
)


@dataclass
class Candle:
    symbol: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trade_count: int
    taker_buy_base: Decimal
    taker_buy_quote: Decimal
    source: str

    def as_row(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))


def _check_aware(dt: datetime, label: str) -> None:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{label} must be timezone-aware UTC, got {dt!r}")


def upsert_symbol(conn: psycopg.Connection, **fields_) -> None:
    cols = list(fields_)
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "symbol")
    sql = f"INSERT INTO symbols ({', '.join(cols)}) VALUES ({placeholders})"
    sql += f" ON CONFLICT (symbol) DO UPDATE SET {updates}, updated_at = now()" if updates \
        else " ON CONFLICT (symbol) DO NOTHING"
    with conn.cursor() as cur:
        cur.execute(sql, tuple(fields_.values()))
    conn.commit()


def upsert_candles(conn: psycopg.Connection, candles: Iterable[Candle]) -> int:
    rows = list(candles)
    if not rows:
        return 0
    for c in rows:
        _check_aware(c.open_time, "open_time")
        _check_aware(c.close_time, "close_time")

    cols = ", ".join(CANDLE_COLUMNS)
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in CANDLE_COLUMNS
        if c not in ("symbol", "open_time")
    )
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS staging_candles "
            "(LIKE candles_1m INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        cur.execute("TRUNCATE staging_candles")
        with cur.copy(f"COPY staging_candles ({cols}) FROM STDIN") as copy:
            for c in rows:
                copy.write_row(c.as_row())
        cur.execute(
            f"""
            INSERT INTO candles_1m ({cols})
            SELECT {cols} FROM staging_candles
            ON CONFLICT (symbol, open_time) DO UPDATE SET {updates}
            WHERE source_priority(EXCLUDED.source)
                  >= source_priority(candles_1m.source)
            """
        )
    conn.commit()
    return len(rows)


BOOK_TICKER_COLUMNS = ("symbol", "ts", "bid_price", "bid_qty", "ask_price", "ask_qty")


def upsert_book_ticker(conn: psycopg.Connection, rows: list[tuple]) -> int:
    """Store at most one row per symbol per second: the last quote received
    in that second, stamped with the second itself.

    Enforced in SQL: within a batch DISTINCT ON keeps the latest ts of each
    (symbol, second); across batches ON CONFLICT DO UPDATE lets a later
    flush replace the row for a second an earlier flush already wrote.
    Replaying a batch is idempotent. Returns the rows inserted or updated.
    """
    if not rows:
        return 0
    cols = ", ".join(BOOK_TICKER_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS staging_book_ticker "
            "(LIKE book_ticker INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        cur.execute("TRUNCATE staging_book_ticker")
        with cur.copy(f"COPY staging_book_ticker ({cols}) FROM STDIN") as copy:
            for r in rows:
                copy.write_row(r)
        cur.execute(
            f"""
            INSERT INTO book_ticker ({cols})
            SELECT DISTINCT ON (symbol, date_trunc('second', ts, 'UTC'))
                   symbol, date_trunc('second', ts, 'UTC'),
                   bid_price, bid_qty, ask_price, ask_qty
              FROM staging_book_ticker
             ORDER BY symbol, date_trunc('second', ts, 'UTC'), ts DESC
            ON CONFLICT (symbol, ts) DO UPDATE SET
                bid_price = EXCLUDED.bid_price, bid_qty = EXCLUDED.bid_qty,
                ask_price = EXCLUDED.ask_price, ask_qty = EXCLUDED.ask_qty
            """
        )
        written = cur.rowcount
    conn.commit()
    return written


def last_candle_time(conn: psycopg.Connection, symbol: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT max(open_time) FROM candles_1m WHERE symbol = %s", (symbol,)
        )
        return cur.fetchone()[0]


CANDLE_TABLES = {"candles_1m", "candles_5m", "candles_15m", "candles_1h",
                 "candles_4h", "candles_1d"}


def find_gaps(conn, symbol: str, start: datetime, end: datetime,
              step: timedelta = timedelta(minutes=1),
              table: str = "candles_1m") -> list[tuple[datetime, datetime]]:
    """Missing ranges [gap_start, gap_end) among the open_times expected at
    every `step` in [start, end). Computed in SQL with lag(), so only the
    gaps -- never the candles -- come back to Python.

    Two sentinels (start - step and end) make a missing head or tail of the
    range show up as a gap too.
    """
    if table not in CANDLE_TABLES:
        raise ValueError(f"unknown candle table {table!r}")
    _check_aware(start, "start")
    _check_aware(end, "end")
    if end <= start:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH t AS (
                SELECT open_time FROM {table}
                 WHERE symbol = %(symbol)s
                   AND open_time >= %(start)s AND open_time < %(end)s
                UNION ALL SELECT %(before)s::timestamptz
                UNION ALL SELECT %(end)s::timestamptz
            )
            SELECT prev + %(step)s, open_time
              FROM (SELECT open_time,
                           lag(open_time) OVER (ORDER BY open_time) AS prev
                      FROM t) l
             WHERE open_time - prev > %(step)s
             ORDER BY 1
            """,
            {"symbol": symbol, "start": start, "end": end,
             "before": start - step, "step": step},
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def get_candles(conn, symbol: str, timeframe: str, start: datetime,
                end: datetime) -> list[dict]:
    table = "candles_1m" if timeframe == "1m" else f"candles_{timeframe}"
    if table not in {"candles_1m", "candles_5m", "candles_15m", "candles_1h",
                     "candles_4h", "candles_1d"}:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM {table} WHERE symbol = %s AND open_time >= %s "
            f"AND open_time < %s ORDER BY open_time",
            (symbol, start, end),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def start_run(conn, component: str, symbol: str | None,
              period_start: datetime | None, period_end: datetime | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingestion_runs (component, symbol, period_start, "
            "period_end, status) VALUES (%s, %s, %s, %s, 'running') RETURNING id",
            (component, symbol, period_start, period_end),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_run(conn, run_id: int, status: str, rows_written: int = 0,
               error: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingestion_runs SET status = %s, rows_written = %s, "
            "error = %s, finished_at = now() WHERE id = %s",
            (status, rows_written, error, run_id),
        )
    conn.commit()


def completed_periods(conn, component: str, symbol: str) -> set[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT period_start, period_end FROM ingestion_runs WHERE "
            "component = %s AND symbol = %s AND status = 'success'",
            (component, symbol),
        )
        return {(r[0], r[1]) for r in cur.fetchall()}


# Continuous aggregates over candles_1m, with their bucket widths.
AGGREGATE_STEPS = {
    "candles_5m": timedelta(minutes=5),
    "candles_15m": timedelta(minutes=15),
    "candles_1h": timedelta(hours=1),
    "candles_4h": timedelta(hours=4),
    "candles_1d": timedelta(days=1),
}
# Every bucket width divides a day, and TimescaleDB's default bucket origin
# is a UTC midnight, so buckets align to multiples of the width from here.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _floor(t: datetime, step: timedelta) -> datetime:
    return _EPOCH + ((t - _EPOCH) // step) * step


def _ceil(t: datetime, step: timedelta) -> datetime:
    floor = _floor(t, step)
    return floor if floor == t else floor + step


def refresh_aggregates(conn, start: datetime, end: datetime,
                       now: datetime | None = None) -> list[str]:
    """Refresh all five continuous aggregates over the 1m range [start, end).

    The refresh policies only reach back their start_offset (3 days for 5m,
    up to 365 days for 1d), so history loaded or overwritten further back
    never reaches the higher timeframes on its own. This is the explicit
    refresh that does.

    Runs on a separate autocommit connection to the same database as
    `conn` (TimescaleDB refuses to refresh inside a transaction block).
    Each view's window is widened to whole buckets, but never past the
    bucket still forming at `now`: that one is left to the policies.
    Returns the views refreshed.
    """
    _check_aware(start, "start")
    _check_aware(end, "end")
    now = now or datetime.now(timezone.utc)
    refreshed = []
    # get_parameters() carries connect_timeout only if `conn` was opened with
    # one; make sure this connection always has it.
    params = {"connect_timeout": CONNECT_TIMEOUT_SECONDS,
              **conn.info.get_parameters()}
    with psycopg.connect(**params, password=conn.info.password,
                         autocommit=True) as ac:
        for view, step in AGGREGATE_STEPS.items():
            w_start = _floor(start, step)
            w_end = min(_ceil(end, step), _floor(now, step))
            if w_end - w_start < step:
                continue  # not one complete bucket to refresh yet
            began = time.monotonic()
            ac.execute("CALL refresh_continuous_aggregate(%s, %s, %s)",
                       (view, w_start, w_end))
            log.info("refreshed %s over [%s, %s) in %.1fs",
                     view, w_start, w_end, time.monotonic() - began)
            refreshed.append(view)
    return refreshed
