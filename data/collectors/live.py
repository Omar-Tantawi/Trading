import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import websockets

from data.collectors.binance_rest import REST_SETTLE, BinanceRest
from data.collectors.binance_ws import (
    parse_book_ticker_message,
    parse_kline_message,
    stream_url,
)
from data.config import get_settings
from data.storage.db import connect
from data.storage.repository import (
    Candle,
    last_candle_time,
    upsert_book_ticker,
    upsert_candles,
)
from data.collectors.archive import parse_timestamp
from decimal import Decimal

log = logging.getLogger(__name__)

BOOK_FLUSH_SECONDS = 5
MAX_BUFFERED_CANDLES = 10_000


def minutes_missing(last_open: datetime, now: datetime) -> int:
    """How many closed 1m candles are missing between last_open and now."""
    for label, dt in (("last_open", last_open), ("now", now)):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError(f"{label} must be timezone-aware UTC")
    last_closed = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
    return max(0, int((last_closed - last_open).total_seconds() // 60))


class LiveCollector:
    def __init__(self, settings=None, api: BinanceRest | None = None,
                 connect_fn=connect):
        self.settings = settings or get_settings()
        self.api = api or BinanceRest()
        self.connect_fn = connect_fn
        self.book_buffer: list[tuple] = []
        self.candle_buffer: list[Candle] = []
        self.last_message_at: datetime | None = None

    def gap_fill(self, conn, symbol: str) -> int:
        """Pull missed minutes from REST after a disconnect."""
        last = last_candle_time(conn, symbol)
        if last is None:
            return 0
        now = datetime.now(timezone.utc)
        if minutes_missing(last, now) == 0:
            return 0
        log.warning("gap detected for %s since %s; backfilling via REST",
                    symbol, last)
        written = 0
        cursor = last + timedelta(minutes=1)
        while cursor < now:
            rows = self.api.klines(symbol, int(cursor.timestamp() * 1000))
            if not rows:
                break
            candles = [
                Candle(
                    symbol=symbol,
                    open_time=parse_timestamp(r[0]),
                    close_time=parse_timestamp(r[6]),
                    open=Decimal(r[1]), high=Decimal(r[2]), low=Decimal(r[3]),
                    close=Decimal(r[4]), volume=Decimal(r[5]),
                    quote_volume=Decimal(r[7]), trade_count=int(r[8]),
                    taker_buy_base=Decimal(r[9]), taker_buy_quote=Decimal(r[10]),
                    source="rest",
                )
                for r in rows
            ]
            # Closed, and settled for REST_SETTLE: a REST row stored while
            # Binance can still revise it would outrank the final ws row.
            candles = [c for c in candles if c.close_time < now - REST_SETTLE]
            if not candles:
                break
            written += upsert_candles(conn, candles)
            cursor = candles[-1].open_time + timedelta(minutes=1)
        return written

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        url = stream_url(self.settings.binance_ws_url, self.settings.symbols)
        while not stop_event.is_set():
            conn = None
            try:
                conn = self.connect_fn()
                for symbol in self.settings.symbols:
                    self.gap_fill(conn, symbol)
                async with websockets.connect(url, ping_interval=20) as ws:
                    log.info("live collector connected: %s", self.settings.symbols)
                    flusher = asyncio.create_task(self._flush_loop(conn, stop_event))
                    try:
                        async for raw in ws:
                            if stop_event.is_set():
                                break
                            self._handle(json.loads(raw), conn)
                    finally:
                        flusher.cancel()
                        if self.book_buffer:
                            rows, self.book_buffer = self.book_buffer, []
                            try:
                                upsert_book_ticker(conn, rows)
                            except Exception as exc:
                                log.error("final book ticker flush failed: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("live collector error: %s; reconnecting in 5s", exc)
                await asyncio.sleep(5)
            finally:
                if conn is not None:
                    conn.close()

    def _handle(self, message: dict, conn) -> None:
        data = message.get("data", message)
        self.last_message_at = datetime.now(timezone.utc)
        if data.get("e") == "kline":
            candle = parse_kline_message(data)
            if candle is not None:
                upsert_candles(conn, [candle])
        elif "b" in data and "a" in data:
            self.book_buffer.append(
                parse_book_ticker_message(data, self.last_message_at)
            )
            if len(self.book_buffer) > MAX_BUFFERED_CANDLES:
                log.error("book ticker buffer overflow; dropping %d rows",
                          len(self.book_buffer))
                self.book_buffer.clear()

    async def _flush_loop(self, conn, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(BOOK_FLUSH_SECONDS)
            if self.book_buffer:
                rows, self.book_buffer = self.book_buffer, []
                try:
                    upsert_book_ticker(conn, rows)
                except Exception as exc:
                    log.error("book ticker flush failed: %s", exc)
                    conn.rollback()
