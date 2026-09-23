from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from typing import Iterable

import psycopg

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


def upsert_book_ticker(conn: psycopg.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        with cur.copy(
            "COPY book_ticker (symbol, ts, bid_price, bid_qty, ask_price, ask_qty) "
            "FROM STDIN"
        ) as copy:
            for r in rows:
                copy.write_row(r)
    conn.commit()
    return len(rows)


def last_candle_time(conn: psycopg.Connection, symbol: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT max(open_time) FROM candles_1m WHERE symbol = %s", (symbol,)
        )
        return cur.fetchone()[0]


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
