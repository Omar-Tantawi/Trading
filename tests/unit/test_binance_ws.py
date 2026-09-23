from datetime import datetime, timezone
from decimal import Decimal

from data.collectors.binance_ws import (
    parse_book_ticker_message,
    parse_kline_message,
    stream_url,
)

CLOSED = {
    "stream": "btcusdt@kline_1m",
    "data": {"e": "kline", "E": 1704067259999, "s": "BTCUSDT", "k": {
        "t": 1704067200000, "T": 1704067259999, "s": "BTCUSDT", "i": "1m",
        "o": "42283.58", "c": "42475.23", "h": "42554.57", "l": "42261.02",
        "v": "1271.51075", "n": 52891, "x": True, "q": "53948745.14",
        "V": "657.8598", "Q": "27909858.09"}},
}


def test_stream_url_lowercases_and_joins():
    url = stream_url("wss://stream.binance.com:9443", ["BTCUSDT", "ETHUSDT"])
    assert url == (
        "wss://stream.binance.com:9443/stream?streams="
        "btcusdt@kline_1m/btcusdt@bookTicker/"
        "ethusdt@kline_1m/ethusdt@bookTicker"
    )


def test_parses_closed_candle():
    candle = parse_kline_message(CLOSED["data"])
    assert candle is not None
    assert candle.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert candle.close == Decimal("42475.23")
    assert candle.trade_count == 52891
    assert candle.source == "ws"


def test_ignores_unclosed_candle():
    msg = {"e": "kline", "s": "BTCUSDT", "k": dict(CLOSED["data"]["k"], x=False)}
    assert parse_kline_message(msg) is None


def test_parses_book_ticker():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    row = parse_book_ticker_message(
        {"u": 1, "s": "BTCUSDT", "b": "42000.01", "B": "1.5",
         "a": "42000.99", "A": "2.5"}, ts
    )
    assert row == ("BTCUSDT", ts, Decimal("42000.01"), Decimal("1.5"),
                   Decimal("42000.99"), Decimal("2.5"))
