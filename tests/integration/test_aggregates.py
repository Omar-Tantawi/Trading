"""Higher timeframes must agree with the 1m truth through the refresh path
production actually uses: backfill_symbol (which refreshes what it wrote)
followed by the scheduled refresh policies, run here with CALL run_job.

Never a manual refresh_continuous_aggregate(..., NULL, NULL): production
never does that, and it is exactly what hid the aggregates holding only
their policies' recent windows.
"""
import io
import random
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.collectors.backfill import backfill_symbol
from data.storage.repository import Candle, get_candles, upsert_candles, upsert_symbol

pytestmark = pytest.mark.db

SYMBOL = "ETHUSDT"
# Older than every policy's start_offset (the widest is 365 days), so no
# scheduled policy will ever materialize or re-materialize it on its own.
T0 = datetime(2023, 5, 1, tzinfo=timezone.utc)
DAYS = 2
VIEWS = {
    "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1),
}


def _random_candles(n: int, source: str, seed: int) -> list[Candle]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        base = Decimal(rng.randint(1800_00, 2000_00)) / 100
        open_time = T0 + timedelta(minutes=i)
        out.append(Candle(
            symbol=SYMBOL, open_time=open_time,
            close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
            open=base, high=base + Decimal(rng.randint(1, 900)) / 100,
            low=base - Decimal(rng.randint(1, 900)) / 100,
            close=base + Decimal(rng.randint(-500, 500)) / 100,
            volume=Decimal(rng.randint(1, 10_000)) / 1000,
            quote_volume=Decimal(rng.randint(1, 10_000_000)) / 1000,
            trade_count=rng.randint(1, 500),
            taker_buy_base=Decimal(rng.randint(0, 5_000)) / 1000,
            taker_buy_quote=Decimal(rng.randint(0, 5_000_000)) / 1000,
            source=source,
        ))
    return out


def _as_zip(candles: list[Candle]) -> bytes:
    lines = []
    for c in candles:
        lines.append(",".join(str(v) for v in (
            int(c.open_time.timestamp() * 1000), c.open, c.high, c.low, c.close,
            c.volume, int(c.close_time.timestamp() * 1000), c.quote_volume,
            c.trade_count, c.taker_buy_base, c.taker_buy_quote, 0,
        )))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{SYMBOL}-1m-2023-05.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def _expected(candles: list[Candle], step: timedelta) -> dict:
    """Independent aggregation in plain Python, exact on Decimals."""
    buckets: dict[datetime, list[Candle]] = defaultdict(list)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for c in candles:
        buckets[epoch + ((c.open_time - epoch) // step) * step].append(c)
    out = {}
    for start, group in buckets.items():
        group.sort(key=lambda c: c.open_time)
        out[start] = {
            "open": group[0].open,
            "high": max(c.high for c in group),
            "low": min(c.low for c in group),
            "close": group[-1].close,
            "volume": sum((c.volume for c in group), Decimal(0)),
            "quote_volume": sum((c.quote_volume for c in group), Decimal(0)),
            "trade_count": sum(c.trade_count for c in group),
            "taker_buy_base": sum((c.taker_buy_base for c in group), Decimal(0)),
            "taker_buy_quote": sum((c.taker_buy_quote for c in group), Decimal(0)),
        }
    return out


class MonthServer:
    def __init__(self, blob):
        self.blob = blob

    def fetch_month(self, symbol, year, month):
        return self.blob if (year, month) == (2023, 5) else None

    def fetch_day(self, symbol, day):
        return None


class QuietApi:
    def first_candle_time(self, symbol):
        return T0

    def klines(self, symbol, start_ms, limit=1000):
        return []


@pytest.fixture(autouse=True)
def views_empty_over_the_test_range(db_conn, autocommit_conn):
    """db_conn DELETEs the 1m rows, but deleting only logs an invalidation:
    rows materialized by an earlier test (or an earlier run) would stay in
    the views and could make a test pass on stale data. Refreshing the
    now-empty range clears them first."""
    with autocommit_conn.cursor() as cur:
        for tf in VIEWS:
            cur.execute("CALL refresh_continuous_aggregate(%s, %s, %s)",
                        (f"candles_{tf}", T0, T0 + timedelta(days=DAYS)))
            cur.execute(f"SELECT count(*) FROM candles_{tf} WHERE symbol = %s "
                        "AND open_time >= %s AND open_time < %s",
                        (SYMBOL, T0, T0 + timedelta(days=DAYS)))
            assert cur.fetchone()[0] == 0


def _run_refresh_policies(autocommit_conn) -> None:
    with autocommit_conn.cursor() as cur:
        cur.execute(
            "SELECT job_id FROM timescaledb_information.jobs "
            "WHERE proc_name = 'policy_refresh_continuous_aggregate' "
            "AND hypertable_name = ANY(%s)",
            ([f"candles_{tf}" for tf in VIEWS],),
        )
        job_ids = [r[0] for r in cur.fetchall()]
        assert len(job_ids) == len(VIEWS)
        for job_id in job_ids:
            cur.execute("CALL run_job(%s)", (job_id,))


def _assert_every_view_matches(conn, truth: list[Candle]) -> None:
    end = T0 + timedelta(days=DAYS)
    for tf, step in VIEWS.items():
        rows = get_candles(conn, SYMBOL, tf, T0, end)
        want = _expected(truth, step)
        assert [r["open_time"] for r in rows] == sorted(want), f"{tf} buckets"
        for row in rows:
            w = want[row["open_time"]]
            for col, value in w.items():
                assert row[col] == value, f"{tf} {row['open_time']} {col}"


def test_backfilled_history_reaches_every_timeframe(db_conn, autocommit_conn):
    upsert_symbol(db_conn, symbol=SYMBOL)
    archive = _random_candles(DAYS * 1440, "archive", seed=7)

    backfill_symbol(db_conn, SYMBOL, MonthServer(_as_zip(archive)), QuietApi(),
                    until=datetime(2023, 6, 1, tzinfo=timezone.utc))
    _run_refresh_policies(autocommit_conn)

    _assert_every_view_matches(db_conn, archive)


def test_an_archive_overwrite_older_than_every_policy_window_reaches_the_views(
        db_conn, autocommit_conn):
    """The aggregates once materialized ws values for these minutes (as the
    policies did when the live collector wrote them). The archive later
    overwrites the 1m rows, far outside every policy window."""
    upsert_symbol(db_conn, symbol=SYMBOL)
    upsert_candles(db_conn, _random_candles(DAYS * 1440, "ws", seed=99))
    with autocommit_conn.cursor() as cur:
        for tf in VIEWS:  # the past state: ws values materialized
            cur.execute("CALL refresh_continuous_aggregate(%s, %s, %s)",
                        (f"candles_{tf}", T0, T0 + timedelta(days=DAYS)))

    archive = _random_candles(DAYS * 1440, "archive", seed=7)
    backfill_symbol(db_conn, SYMBOL, MonthServer(_as_zip(archive)), QuietApi(),
                    until=datetime(2023, 6, 1, tzinfo=timezone.utc))
    _run_refresh_policies(autocommit_conn)

    _assert_every_view_matches(db_conn, archive)


def test_what_landed_before_a_failure_still_reaches_the_views(db_conn):
    """A backfill that loads a month and then aborts in the tail (here on a
    rate limit) must still refresh the aggregates over what it wrote: a
    re-run skips the loaded month, so nothing else would ever refresh it."""
    from data.collectors.binance_rest import RateLimitedError

    class RateLimitedApi(QuietApi):
        def klines(self, symbol, start_ms, limit=1000):
            raise RateLimitedError(429, 60)

    upsert_symbol(db_conn, symbol=SYMBOL)
    archive = _random_candles(DAYS * 1440, "archive", seed=11)

    with pytest.raises(RateLimitedError):
        backfill_symbol(db_conn, SYMBOL, MonthServer(_as_zip(archive)),
                        RateLimitedApi(),
                        until=datetime(2023, 6, 2, tzinfo=timezone.utc))

    _assert_every_view_matches(db_conn, archive)
