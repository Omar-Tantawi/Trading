import asyncio
import json
import logging
import random
from datetime import datetime, timedelta, timezone

import websockets

from data.collectors.binance_rest import REST_SETTLE, BinanceRest, RateLimitedError
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

# Reconnect backoff after an error: 1 s, ~2 s, ~4 s ... capped at 60 s, and
# reset once a connection delivers messages again.
RECONNECT_BASE_SECONDS = 1.0
RECONNECT_CAP_SECONDS = 60.0


def reconnect_delay(failures: int, jitter: bool = True) -> float:
    """Delay before reconnect attempt number `failures` (1-based): exponential
    with jitter applied inside the cap. The exponent is bounded so a very
    long outage can never overflow the float."""
    raw = RECONNECT_BASE_SECONDS * 2 ** min(max(failures - 1, 0), 16)
    if jitter and failures > 1:
        raw *= 0.8 + 0.4 * random.random()
    return min(RECONNECT_CAP_SECONDS, raw)


async def _pause(stop_event: asyncio.Event, seconds: float) -> None:
    """Sleep up to `seconds`, returning early if the collector is stopped."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


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
        failures = 0
        while not stop_event.is_set():
            conn = None
            pause = 0.0
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
                            failures = 0  # messages flowing: connection healthy
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
            except RateLimitedError as exc:
                # Binance told us to stop. Retrying early is how a 429
                # becomes a 418 IP ban, so wait exactly what it asked for.
                pause = exc.retry_after
                log.critical(
                    "live collector RATE LIMITED by Binance (HTTP %s): pausing "
                    "%.0fs as instructed (Retry-After) before reconnecting; "
                    "no requests until then", exc.status, exc.retry_after,
                )
            except Exception as exc:
                failures += 1
                pause = reconnect_delay(failures)
                log.error("live collector error: %s; reconnecting in %.1fs "
                          "(attempt %d)", exc, pause, failures)
            finally:
                if conn is not None:
                    conn.close()
            if pause and not stop_event.is_set():
                # The DB connection is already closed: never hold it idle
                # through a pause that can last as long as a rate-limit ban.
                await _pause(stop_event, pause)

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
