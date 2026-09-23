import logging
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

import httpx

from data.collectors import archive
from data.collectors.binance_rest import REST_SETTLE, BinanceRest, backoff_delays
from data.storage.repository import (
    Candle,
    completed_periods,
    find_gaps,
    finish_run,
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

    def _fetch_verified(self, url: str,
                        checksum_may_lag: bool = False) -> bytes | None:
        blob = self._get(url)
        if blob is None:
            return None
        checksum = self._get(url + ".CHECKSUM")
        if checksum is None:
            if checksum_may_lag:
                # Binance publishes a day's .CHECKSUM some time after its
                # zip. Until then the day is "not yet available": skip it
                # (the REST gap fill covers it) rather than fail the symbol.
                log.info("%s: checksum not published yet; skipping for now", url)
                return None
            raise ValueError(f"no checksum published for {url}")
        if not archive.verify_sha256(blob, checksum.decode("utf-8")):
            raise ValueError(f"checksum mismatch for {url}; refusing to load")
        return blob

    def fetch_month(self, symbol: str, year: int, month: int) -> bytes | None:
        return self._fetch_verified(
            archive.monthly_url(self.base_url, symbol, year, month)
        )

    def fetch_day(self, symbol: str, day: date) -> bytes | None:
        return self._fetch_verified(archive.daily_url(self.base_url, symbol, day),
                                    checksum_may_lag=True)


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
    A month whose archive 404s is recorded 'missing' and retried next run.

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
                # The loop starts at the listing date, so every month here
                # should exist. A 404 means "not published yet" (e.g. a run
                # on the 1st-3rd, before Binance publishes last month).
                # 'missing' is not in `done`, so the next run asks again;
                # recording 'success' would skip this month forever.
                finish_run(conn, run_id, "missing", 0)
                log.warning("%s %04d-%02d: monthly archive not published yet; "
                            "will retry on the next run", symbol, year, month)
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


def _closed_open_bound(cutoff: datetime) -> datetime:
    """Exclusive upper bound on the open_time of candles that have closed
    before `cutoff` (a candle opening at T closes at T + 59.999 s)."""
    t = cutoff - timedelta(milliseconds=59_999)
    floor = t.replace(second=0, microsecond=0)
    return floor if floor == t else floor + timedelta(minutes=1)


def _rest_candle(symbol: str, r: list) -> Candle:
    return Candle(
        symbol=symbol,
        open_time=archive.parse_timestamp(r[0]),
        close_time=archive.parse_timestamp(r[6]),
        open=Decimal(r[1]), high=Decimal(r[2]), low=Decimal(r[3]),
        close=Decimal(r[4]), volume=Decimal(r[5]),
        quote_volume=Decimal(r[7]), trade_count=int(r[8]),
        taker_buy_base=Decimal(r[9]), taker_buy_quote=Decimal(r[10]),
        source="rest",
    )


def _rest_fill(conn, symbol: str, api, gap_start: datetime, gap_end: datetime,
               cutoff: datetime) -> int:
    """Fill [gap_start, gap_end) from REST, 1000 candles per call."""
    written = 0
    cursor = gap_start
    while cursor < gap_end:
        rows = api.klines(symbol, int(cursor.timestamp() * 1000), limit=1000)
        if not rows:
            break
        # Keep only this gap, and only closed candles: never the forming one.
        candles = [c for c in (_rest_candle(symbol, r) for r in rows)
                   if c.open_time < gap_end and c.close_time < cutoff]
        if not candles:
            break  # nothing exists in the rest of this gap on the exchange
        written += upsert_candles(conn, candles)
        cursor = candles[-1].open_time + timedelta(minutes=1)
    return written


def _backfill_tail(conn, symbol: str, downloader, api, until: datetime,
                   loop_start: datetime) -> int:
    """Complete the current (incomplete) month: re-walk its daily archives,
    then REST-fill every gap that remains in [tail_start, until).

    The tail is anchored at max(loop_start, first day of until's month) --
    never at max(open_time), because any newer row (a ws candle from the
    live collector, a day after a 404) would hide every hole before it.
    Re-walking the month's daily archives on every run is idempotent and
    upgrades rest/ws rows to archive via the SQL source precedence.
    """
    # Only closed candles, and (for REST) only ones that closed at least
    # REST_SETTLE ago in wall-clock time.
    cutoff = min(until, datetime.now(timezone.utc) - REST_SETTLE)
    month_start = until.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    tail_start = max(loop_start, month_start).replace(second=0, microsecond=0)
    tail_end = _closed_open_bound(cutoff)
    if tail_start >= tail_end:
        return 0
    written = 0

    day = tail_start.date()
    while day < until.date():
        blob = downloader.fetch_day(symbol, day)
        if blob is not None:
            candles = [c for c in archive.rows_from_zip(blob, symbol)
                       if tail_start <= c.open_time < tail_end]
            written += upsert_candles(conn, candles)
        day += timedelta(days=1)

    gaps = find_gaps(conn, symbol, tail_start, tail_end)
    conn.commit()  # end the read transaction; nothing may hold locks idle
    for gap_start, gap_end in gaps:
        filled = _rest_fill(conn, symbol, api, gap_start, gap_end, cutoff)
        log.info("%s: REST gap %s -> %s: %d candles",
                 symbol, gap_start, gap_end, filled)
        written += filled
    return written
