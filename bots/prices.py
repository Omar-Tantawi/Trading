"""The price of a coin at a time, from the stored 1m candles (spec §3)."""
from datetime import datetime, timedelta

MAX_AGE = timedelta(minutes=10)


def price_at(conn, symbol: str, t: datetime) -> float | None:
    """The open of the latest 1m candle starting at or before t, if it
    started at most 10 minutes before t; otherwise None."""
    with conn.cursor() as cur:
        cur.execute("SELECT open FROM candles_1m WHERE symbol = %s AND open_time <= %s "
                    "AND open_time >= %s ORDER BY open_time DESC LIMIT 1",
                    (symbol, t, t - MAX_AGE))
        row = cur.fetchone()
    return float(row[0]) if row else None
