from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.collectors.live import LiveCollector
from data.storage.repository import Candle, last_candle_time, upsert_candles, upsert_symbol

pytestmark = pytest.mark.db


def _kline_row(open_dt: datetime) -> list:
    """A raw Binance REST kline row, as BinanceRest.klines would return it."""
    open_ms = int(open_dt.timestamp() * 1000)
    close_ms = open_ms + 59_999
    return [open_ms, "100.00000000", "101.00000000", "99.00000000",
            "100.50000000", "10.00000000", close_ms, "1005.00000000",
            5, "5.00000000", "502.50000000", "0"]


def _archive_candle(open_time: datetime) -> Candle:
    return Candle(
        symbol="BTCUSDT", open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=Decimal("100"), high=Decimal("100"), low=Decimal("100"),
        close=Decimal("100"), volume=Decimal("1"), quote_volume=Decimal("100"),
        trade_count=1, taker_buy_base=Decimal("1"), taker_buy_quote=Decimal("100"),
        source="archive",
    )


class PagingFakeApi:
    """Mimics a paged REST endpoint: a full page, then a shorter page whose
    last row has not closed yet, then an empty page that ends the loop."""

    def __init__(self):
        self.calls: list[tuple] = []

    def klines(self, symbol, start_ms, limit=1000):
        self.calls.append((symbol, start_ms, limit))
        start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        call_index = len(self.calls)
        if call_index == 1:
            return [_kline_row(start + timedelta(minutes=i)) for i in range(3)]
        if call_index == 2:
            # One closed candle, then one that is nowhere near closed (a day
            # in the future): gap_fill must drop it regardless of exactly
            # when "now" is sampled inside the function.
            far_future = datetime.now(timezone.utc) + timedelta(days=1)
            return [_kline_row(start), _kline_row(far_future)]
        return []


class ExplodingApi:
    """Fails the test if gap_fill calls REST when it shouldn't."""

    def klines(self, *args, **kwargs):
        raise AssertionError("REST should not be called when there is no gap")


def test_gap_fill_pages_through_rest_and_excludes_forming_candle(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    last_open = (datetime.now(timezone.utc) - timedelta(minutes=10)) \
        .replace(second=0, microsecond=0)
    upsert_candles(db_conn, [_archive_candle(last_open)])
    assert last_candle_time(db_conn, "BTCUSDT") == last_open

    api = PagingFakeApi()
    collector = LiveCollector(api=api, connect_fn=lambda: db_conn)

    written = collector.gap_fill(db_conn, "BTCUSDT")

    # Page 1: 3 closed candles. Page 2: 1 closed + 1 dropped because it has
    # not closed yet. Page 3: empty, which ends the loop.
    assert len(api.calls) == 3
    assert written == 4

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM candles_1m WHERE symbol = %s AND source = 'rest'",
            ("BTCUSDT",),
        )
        (rest_count,) = cur.fetchone()
    assert rest_count == 4

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM candles_1m WHERE symbol = %s "
            "AND open_time > now()",
            ("BTCUSDT",),
        )
        (future_count,) = cur.fetchone()
    assert future_count == 0


def test_gap_fill_is_a_noop_when_nothing_is_missing(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    last_open = datetime.now(timezone.utc).replace(second=0, microsecond=0) \
        - timedelta(minutes=1)
    upsert_candles(db_conn, [_archive_candle(last_open)])

    collector = LiveCollector(api=ExplodingApi(), connect_fn=lambda: db_conn)
    assert collector.gap_fill(db_conn, "BTCUSDT") == 0


def test_gap_fill_returns_zero_when_symbol_has_no_history(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")

    collector = LiveCollector(api=ExplodingApi(), connect_fn=lambda: db_conn)
    assert collector.gap_fill(db_conn, "BTCUSDT") == 0
