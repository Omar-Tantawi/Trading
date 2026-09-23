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


def test_find_gaps_reports_head_middle_and_tail_gaps(db_conn):
    from data.storage.repository import find_gaps

    upsert_symbol(db_conn, symbol="BTCUSDT")
    present = [2, 3, 4, 7, 8]  # 0-1 missing, 5-6 missing, 9 missing
    upsert_candles(db_conn, [make_candle(i) for i in present])

    gaps = find_gaps(db_conn, "BTCUSDT", T0, T0 + timedelta(minutes=10))
    m = lambda i: T0 + timedelta(minutes=i)  # noqa: E731
    assert gaps == [(m(0), m(2)), (m(5), m(7)), (m(9), m(10))]


def test_find_gaps_of_an_empty_range_is_the_whole_range(db_conn):
    from data.storage.repository import find_gaps

    gaps = find_gaps(db_conn, "BTCUSDT", T0, T0 + timedelta(minutes=10))
    assert gaps == [(T0, T0 + timedelta(minutes=10))]


def _quote(symbol, ts, bid):
    return (symbol, ts, Decimal(bid), Decimal("1"), Decimal(bid) + Decimal("0.1"),
            Decimal("2"))


def test_book_ticker_keeps_one_row_per_symbol_per_second_the_last_quote(db_conn):
    """~393 quotes/s across four symbols projected to ~5 GB/day. Keep at
    most one row per symbol per second: the last quote in that second."""
    ms = lambda n: T0 + timedelta(milliseconds=n)  # noqa: E731
    first_flush = [
        _quote("BTCUSDT", ms(100), "100"),
        _quote("BTCUSDT", ms(700), "101"),
        _quote("ETHUSDT", ms(800), "50"),
        _quote("BTCUSDT", ms(1200), "102"),
    ]
    # The next flush can still carry a later quote for the same second.
    second_flush = [_quote("BTCUSDT", ms(950), "103")]

    upsert_book_ticker(db_conn, first_flush)
    upsert_book_ticker(db_conn, second_flush)
    upsert_book_ticker(db_conn, second_flush)  # replay: no change

    with db_conn.cursor() as cur:
        cur.execute("SELECT symbol, ts, bid_price FROM book_ticker ORDER BY 1, 2")
        rows = cur.fetchall()
    assert rows == [
        ("BTCUSDT", T0, Decimal("103.00000000")),
        ("BTCUSDT", T0 + timedelta(seconds=1), Decimal("102.00000000")),
        ("ETHUSDT", T0, Decimal("50.00000000")),
    ]


def test_book_ticker_is_compressed_after_a_day_and_never_deleted(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT proc_name, config FROM timescaledb_information.jobs "
            "WHERE hypertable_name = 'book_ticker'"
        )
        jobs = {name: config for name, config in cur.fetchall()}
        cur.execute(
            "SELECT attname, segmentby_column_index, orderby_column_index, "
            "orderby_asc FROM timescaledb_information.compression_settings "
            "WHERE hypertable_name = 'book_ticker'"
        )
        settings = {r[0]: r[1:] for r in cur.fetchall()}

    assert jobs.get("policy_compression", {}).get("compress_after") == "1 day"
    assert "policy_retention" not in jobs, "deleting data is the user's call"
    assert settings["symbol"][0] == 1, "segmentby symbol"
    assert settings["ts"][1:] == (1, False), "orderby ts DESC"
