import logging
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

import httpx

from data.collectors import archive
from data.collectors.binance_rest import BinanceRest, backoff_delays
from data.storage.repository import (
    Candle,
    completed_periods,
    finish_run,
    last_candle_time,
    start_run,
    upsert_candles,
)

log = logging.getLogger(__name__)


def months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


class ArchiveDownloader:
    def __init__(self, base_url: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep, attempts: int = 4):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=120.0, follow_redirects=True)
        self.sleep = sleep
        self.attempts = attempts

    def _get(self, url: str) -> bytes | None:
        """Returns the body, or None for 404 (which means 'does not exist').

        5xx responses and transport errors (timeouts, refused or dropped
        connections) are retried with backoff. There is no sleep after the
        final attempt: nothing follows it.
        """
        delays = backoff_delays(self.attempts)
        last_problem: object = None
        for attempt, delay in enumerate(delays):
            more_attempts = attempt < len(delays) - 1
            try:
                resp = self.client.get(url)
            except httpx.TransportError as exc:
                last_problem = exc
                if more_attempts:
                    log.warning("%s -> %s, retrying in %.1fs",
                                url, type(exc).__name__, delay)
                    self.sleep(delay)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                last_problem = f"HTTP {resp.status_code}"
                if more_attempts:
                    log.warning("%s -> %s, retrying in %.1fs",
                                url, resp.status_code, delay)
                    self.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.content
        raise RuntimeError(f"gave up downloading {url}: {last_problem}")

    def _fetch_verified(self, url: str) -> bytes | None:
        blob = self._get(url)
        if blob is None:
            return None
        checksum = self._get(url + ".CHECKSUM")
        if checksum is None:
            raise ValueError(f"no checksum published for {url}")
        if not archive.verify_sha256(blob, checksum.decode("utf-8")):
            raise ValueError(f"checksum mismatch for {url}; refusing to load")
        return blob

    def fetch_month(self, symbol: str, year: int, month: int) -> bytes | None:
        return self._fetch_verified(
            archive.monthly_url(self.base_url, symbol, year, month)
        )

    def fetch_day(self, symbol: str, day: date) -> bytes | None:
        return self._fetch_verified(archive.daily_url(self.base_url, symbol, day))


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12
           else datetime(year, month + 1, 1, tzinfo=timezone.utc))
    return start, end


def backfill_symbol(conn, symbol: str, downloader, api: BinanceRest,
                    until: datetime | None = None,
                    start: datetime | None = None) -> int:
    """Load monthly archives, then daily archives, then the REST tail.

    Resumable: months already recorded as successful runs are skipped.
    Months that predate the symbol's listing 404 and are skipped quietly.

    `start`, if given, moves the beginning of the monthly-archive loop
    forward to max(start, the symbol's listing date) instead of always
    starting from the listing date. Omitted, behavior is unchanged.
    """
    until = until or datetime.now(timezone.utc)
    listed_at = api.first_candle_time(symbol)
    loop_start = max(start, listed_at) if start else listed_at
    done = completed_periods(conn, "backfill", symbol)
    total = 0

    for year, month in months_between(loop_start, until):
        m_start, m_end = _month_bounds(year, month)
        if (m_start, m_end) in done:
            continue
        if m_end > until:
            break  # incomplete month: handled by daily archives / REST below
        run_id = start_run(conn, "backfill", symbol, m_start, m_end)
        try:
            blob = downloader.fetch_month(symbol, year, month)
            if blob is None:
                # Before listing, or not published yet. Not an error.
                finish_run(conn, run_id, "success", 0)
                continue
            candles = archive.rows_from_zip(blob, symbol)
            written = upsert_candles(conn, candles)
            finish_run(conn, run_id, "success", written)
            total += written
            log.info("%s %04d-%02d: %d candles", symbol, year, month, written)
        except Exception as exc:
            # A DB-level failure inside upsert_candles leaves the connection's
            # transaction aborted; without a rollback, finish_run's own UPDATE
            # would raise InFailedSqlTransaction and abort the whole backfill.
            conn.rollback()
            finish_run(conn, run_id, "failed", 0, str(exc))
            log.error("%s %04d-%02d failed: %s", symbol, year, month, exc)

    total += _backfill_tail(conn, symbol, downloader, api, until, loop_start)
    return total


def _backfill_tail(conn, symbol: str, downloader, api, until: datetime,
                   fallback_start: datetime | None = None) -> int:
    """Fill from the last stored candle to `until` using daily archives, then
    the REST API for whatever is too recent to be archived."""
    written = 0
    last = last_candle_time(conn, symbol)
    cursor = (last + timedelta(minutes=1)) if last else (
        fallback_start or api.first_candle_time(symbol)
    )

    day = cursor.date()
    while day < until.date():
        blob = downloader.fetch_day(symbol, day)
        if blob is not None:
            candles = [c for c in archive.rows_from_zip(blob, symbol)
                       if c.open_time >= cursor]
            written += upsert_candles(conn, candles)
        day += timedelta(days=1)

    last = last_candle_time(conn, symbol)
    cursor = (last + timedelta(minutes=1)) if last else cursor
    while cursor < until:
        rows = api.klines(symbol, int(cursor.timestamp() * 1000), limit=1000)
        if not rows:
            break
        candles = [
            Candle(
                symbol=symbol,
                open_time=archive.parse_timestamp(r[0]),
                close_time=archive.parse_timestamp(r[6]),
                open=Decimal(r[1]), high=Decimal(r[2]), low=Decimal(r[3]),
                close=Decimal(r[4]), volume=Decimal(r[5]),
                quote_volume=Decimal(r[7]), trade_count=int(r[8]),
                taker_buy_base=Decimal(r[9]), taker_buy_quote=Decimal(r[10]),
                source="rest",
            )
            for r in rows
        ]
        # Drop the still-open final candle: only closed candles are stored.
        candles = [c for c in candles if c.close_time < until]
        if not candles:
            break
        written += upsert_candles(conn, candles)
        cursor = candles[-1].open_time + timedelta(minutes=1)
    return written
