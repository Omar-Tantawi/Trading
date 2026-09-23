from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.storage.repository import (
    Candle,
    last_candle_time,
    upsert_book_ticker,
    upsert_candles,
    upsert_symbol,
)

pytestmark = pytest.mark.db

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_candle(minute: int, close: str = "42000.12345678", source: str = "archive"):
    open_time = T0 + timedelta(minutes=minute)
    return Candle(
        symbol="BTCUSDT",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=Decimal("42000.00000001"),
        high=Decimal("42100.5"),
        low=Decimal("41900.25"),
        close=Decimal(close),
        volume=Decimal("1.23456789"),
        quote_volume=Decimal("51840.5"),
        trade_count=42,
        taker_buy_base=Decimal("0.6"),
        taker_buy_quote=Decimal("25000.1"),
        source=source,
    )


def test_upsert_is_idempotent_and_preserves_precision(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [make_candle(i) for i in range(10)]

    assert upsert_candles(db_conn, rows) == 10
    upsert_candles(db_conn, rows)

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*), min(open) FROM candles_1m")
        count, min_open = cur.fetchone()
    assert count == 10
    assert min_open == Decimal("42000.00000001")


def test_lower_precedence_source_does_not_overwrite(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [make_candle(0, close="1.0", source="archive")])
    upsert_candles(db_conn, [make_candle(0, close="999.0", source="ws")])

    with db_conn.cursor() as cur:
        cur.execute("SELECT close, source FROM candles_1m")
        close, source = cur.fetchone()
    assert close == Decimal("1.00000000")
    assert source == "archive"


def test_higher_precedence_source_does_overwrite(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [make_candle(0, close="1.0", source="ws")])
    upsert_candles(db_conn, [make_candle(0, close="999.0", source="archive")])

    with db_conn.cursor() as cur:
        cur.execute("SELECT close, source FROM candles_1m")
        close, source = cur.fetchone()
    assert close == Decimal("999.00000000")
    assert source == "archive"


def test_last_candle_time(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [make_candle(i) for i in range(5)])
    assert last_candle_time(db_conn, "BTCUSDT") == T0 + timedelta(minutes=4)
    assert last_candle_time(db_conn, "ETHUSDT") is None


def test_rejects_naive_datetime(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    bad = make_candle(0)
    bad.open_time = bad.open_time.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        upsert_candles(db_conn, [bad])


def test_upsert_book_ticker_is_idempotent(db_conn):
    rows = [
        ("BTCUSDT", T0, Decimal("42000.10000000"), Decimal("1.5"),
         Decimal("42000.20000000"), Decimal("2.0")),
        ("BTCUSDT", T0 + timedelta(seconds=1), Decimal("42001.10000000"),
         Decimal("1.1"), Decimal("42001.30000000"), Decimal("0.9")),
    ]

    assert upsert_book_ticker(db_conn, rows) == 2
    upsert_book_ticker(db_conn, rows)  # replay must not raise or duplicate

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM book_ticker")
        (count,) = cur.fetchone()
    assert count == 2
