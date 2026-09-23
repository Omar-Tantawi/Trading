from datetime import datetime, timedelta, timezone

import pytest

from data.collectors.backfill import backfill_symbol
from data.storage.repository import last_candle_time, upsert_symbol

pytestmark = pytest.mark.db


class FakeDownloader:
    """Serves one month of data, 404s for everything else."""

    def __init__(self, blob_by_month):
        self.blob_by_month = blob_by_month
        self.calls = []

    def fetch_month(self, symbol, year, month):
        self.calls.append((symbol, year, month))
        return self.blob_by_month.get((year, month))

    def fetch_day(self, symbol, day):
        return None


class FakeApi:
    def first_candle_time(self, symbol):
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

    def klines(self, symbol, start_ms, limit=1000):
        return []


def make_zip(text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", text)
    return buf.getvalue()


CSV = (
    "1704067200000,42283.58,42554.57,42261.02,42475.23,1271.51075,"
    "1704067259999,53948745.14,52891,657.8598,27909858.09,0\n"
)


def test_backfill_loads_and_is_resumable(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = FakeDownloader({(2024, 1): make_zip(CSV)})

    # `until` must be past the end of January, or the month is treated as
    # incomplete and left to the daily/REST tail instead of the archive.
    written = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 2, 5, tzinfo=timezone.utc),
    )
    assert written == 1

    first_call_count = len(dl.calls)
    written_again = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 2, 5, tzinfo=timezone.utc),
    )
    assert written_again == 0, "already-loaded months must be skipped"
    assert len(dl.calls) == first_call_count, "no re-download on second run"


def test_backfill_records_runs(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = FakeDownloader({(2024, 1): make_zip(CSV)})
    backfill_symbol(db_conn, "BTCUSDT", dl, FakeApi(),
                    until=datetime(2024, 2, 5, tzinfo=timezone.utc))

    with db_conn.cursor() as cur:
        cur.execute("SELECT component, status FROM ingestion_runs")
        rows = cur.fetchall()
    assert ("backfill", "success") in rows


def test_failed_month_rolls_back_and_does_not_abort_remaining_months(db_conn):
    """Regression test for a real DB-level failure mid-load.

    January's row has an out-of-range `open` price: numeric(20,8) allows at
    most 12 digits before the decimal point, and this value has 13. The
    COPY into staging_candles genuinely raises a Postgres numeric overflow
    error inside upsert_candles, leaving the connection's transaction
    aborted. Without a rollback before finish_run, finish_run's own UPDATE
    would raise InFailedSqlTransaction, escape backfill_symbol, strand
    January's ingestion_runs row as 'running' forever, and prevent February
    (and the REST tail) from ever running.
    """
    upsert_symbol(db_conn, symbol="BTCUSDT")
    bad_csv = (
        "1704067200000,1000000000000.12345678,42554.57,42261.02,42475.23,"
        "1271.51075,1704067259999,53948745.14,52891,657.8598,27909858.09,0\n"
    )
    good_csv = (
        "1706745600000,42283.58,42554.57,42261.02,42475.23,1271.51075,"
        "1706745659999,53948745.14,52891,657.8598,27909858.09,0\n"
    )
    dl = FakeDownloader({
        (2024, 1): make_zip(bad_csv),
        (2024, 2): make_zip(good_csv),
    })

    written = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 3, 5, tzinfo=timezone.utc),
    )

    # January's overflow must not abort February: only February's row lands.
    assert written == 1

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT period_start, status, error FROM ingestion_runs "
            "WHERE component = 'backfill' AND symbol = 'BTCUSDT' "
            "ORDER BY period_start"
        )
        by_month = {r[0].month: (r[1], r[2]) for r in cur.fetchall()}

    jan_status, jan_error = by_month[1]
    feb_status, feb_error = by_month[2]
    assert jan_status == "failed"
    assert jan_error, "the error must be recorded, not left blank"
    assert feb_status == "success"


class NoPriorCandlesApi:
    """A symbol with nothing loaded yet: the only elapsed month is still
    in progress, so the monthly loop defers everything to the tail."""

    def __init__(self, first_time):
        self.first_time = first_time
        self.kline_calls = []

    def first_candle_time(self, symbol):
        return self.first_time

    def klines(self, symbol, start_ms, limit=1000):
        self.kline_calls.append(start_ms)
        start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        rows = []
        for i in range(5):
            open_time = start + timedelta(minutes=i)
            close_time = open_time + timedelta(minutes=1) - timedelta(milliseconds=1)
            rows.append([
                int(open_time.timestamp() * 1000),
                "100.0", "101.0", "99.0", "100.5", "10.0",
                int(close_time.timestamp() * 1000),
                "1000.0", 5, "5.0", "500.0", "0",
            ])
        return rows


def test_start_skips_months_before_it(db_conn):
    """--from should move the monthly-archive loop's beginning forward,
    so months before `start` are never even requested."""
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = FakeDownloader({
        (2024, 1): make_zip(CSV),
        (2024, 2): make_zip(CSV),
    })

    written = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 3, 5, tzinfo=timezone.utc),
        start=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    assert written == 1
    assert dl.calls == [("BTCUSDT", 2024, 2)], \
        "January predates `start` and must never be requested"


def test_start_is_ignored_when_it_predates_listing(db_conn):
    """A `start` earlier than the symbol's own listing date changes nothing:
    the loop still begins at the listing date (FakeApi.first_candle_time
    returns 2024-01-01)."""
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = FakeDownloader({(2024, 1): make_zip(CSV)})

    written = backfill_symbol(
        db_conn, "BTCUSDT", dl, FakeApi(),
        until=datetime(2024, 2, 5, tzinfo=timezone.utc),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    assert written == 1
    assert dl.calls == [("BTCUSDT", 2024, 1)]


def test_tail_seeds_from_first_candle_time_when_nothing_is_loaded_yet(db_conn):
    """Regression test: a symbol with no candles at all, and whose only
    elapsed month is incomplete, must still backfill via the REST tail
    instead of _backfill_tail silently returning 0."""
    upsert_symbol(db_conn, symbol="ETHUSDT")
    first_time = datetime(2024, 3, 1, tzinfo=timezone.utc)
    until = datetime(2024, 3, 1, 0, 5, tzinfo=timezone.utc)
    dl = FakeDownloader({})  # no monthly archives; fetch_day always None
    api = NoPriorCandlesApi(first_time)

    written = backfill_symbol(db_conn, "ETHUSDT", dl, api, until=until)

    assert written == 5
    assert api.kline_calls, "the REST tail must have been used at all"
    last = last_candle_time(db_conn, "ETHUSDT")
    assert last == datetime(2024, 3, 1, 0, 4, tzinfo=timezone.utc)


def test_a_404_month_is_recorded_missing_and_re_requested_next_run(db_conn):
    """A post-listing month that 404s (e.g. a re-run on the 1st-3rd, before
    Binance publishes the previous month) must not be recorded 'success':
    that would skip it forever. It is recorded 'missing' and retried."""
    upsert_symbol(db_conn, symbol="BTCUSDT")
    until = datetime(2024, 2, 5, tzinfo=timezone.utc)
    dl = FakeDownloader({})  # January not published yet

    assert backfill_symbol(db_conn, "BTCUSDT", dl, FakeApi(), until=until) == 0
    with db_conn.cursor() as cur:
        cur.execute("SELECT status, rows_written FROM ingestion_runs "
                    "WHERE component = 'backfill' AND symbol = 'BTCUSDT'")
        assert cur.fetchall() == [("missing", 0)]

    # Binance publishes January; the next run must fetch and load it.
    dl.blob_by_month[(2024, 1)] = make_zip(CSV)
    written = backfill_symbol(db_conn, "BTCUSDT", dl, FakeApi(), until=until)
    assert written == 1
    assert dl.calls.count(("BTCUSDT", 2024, 1)) == 2, "January must be re-requested"
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM ingestion_runs WHERE component = "
                    "'backfill' AND symbol = 'BTCUSDT' ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["missing", "success"]


def test_migration_002_reopens_months_the_old_code_marked_success(db_conn):
    """Databases written by the old code hold 0-row 'success' rows for
    404'd months. Migration 002 turns them into 'missing' so they are
    re-requested; real loads (rows > 0) are untouched."""
    from pathlib import Path

    from data.storage.db import MIGRATIONS_DIR
    from data.storage.repository import finish_run, start_run

    jan = (datetime(2024, 1, 1, tzinfo=timezone.utc),
           datetime(2024, 2, 1, tzinfo=timezone.utc))
    feb = (datetime(2024, 2, 1, tzinfo=timezone.utc),
           datetime(2024, 3, 1, tzinfo=timezone.utc))
    finish_run(db_conn, start_run(db_conn, "backfill", "BTCUSDT", *jan), "success", 0)
    finish_run(db_conn, start_run(db_conn, "backfill", "BTCUSDT", *feb), "success", 44640)

    sql = Path(MIGRATIONS_DIR, "002_backfill_missing_status.sql").read_text()
    with db_conn.cursor() as cur:
        cur.execute(sql)
    db_conn.commit()

    from data.storage.repository import completed_periods
    assert completed_periods(db_conn, "backfill", "BTCUSDT") == {feb}


# --- C1: the tail must heal every hole in the current month -----------------

def _kline(open_time: datetime, price: str = "100.0") -> list:
    close_time = open_time + timedelta(minutes=1) - timedelta(milliseconds=1)
    return [int(open_time.timestamp() * 1000), price, price, price, price,
            "10.0", int(close_time.timestamp() * 1000), "1000.0", 5, "5.0",
            "500.0", "0"]


def _day_zip(day: datetime) -> bytes:
    lines = [",".join(str(v) for v in _kline(day + timedelta(minutes=i), "200.0"))
             for i in range(1440)]
    return make_zip("\n".join(lines) + "\n")


class DailyDownloader:
    """No monthly archives; serves the given daily archives, 404s the rest."""

    def __init__(self, days: dict):
        self.days = days
        self.day_calls = []

    def fetch_month(self, symbol, year, month):
        return None

    def fetch_day(self, symbol, day):
        self.day_calls.append(day)
        return self.days.get(day)


class ExchangeApi:
    """A REST endpoint that has every minute from `listed` up to and
    including the still-forming candle at `now`."""

    def __init__(self, listed: datetime, now: datetime):
        self.listed = listed
        self.now = now
        self.starts = []

    def first_candle_time(self, symbol):
        return self.listed

    def klines(self, symbol, start_ms, limit=1000):
        self.starts.append(start_ms)
        t = max(datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc), self.listed)
        rows = []
        while t <= self.now and len(rows) < limit:
            rows.append(_kline(t))
            t += timedelta(minutes=1)
        return rows


def _minutes(conn, symbol, start, end) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM candles_1m WHERE symbol = %s "
                    "AND open_time >= %s AND open_time < %s", (symbol, start, end))
        return cur.fetchone()[0]


def _ws_candle(open_time: datetime):
    from data.storage.repository import Candle
    from decimal import Decimal
    return Candle(symbol="BTCUSDT", open_time=open_time,
                  close_time=open_time + timedelta(seconds=59, milliseconds=999),
                  open=Decimal("1"), high=Decimal("1"), low=Decimal("1"),
                  close=Decimal("1"), volume=Decimal("1"),
                  quote_volume=Decimal("1"), trade_count=1,
                  taker_buy_base=Decimal("1"), taker_buy_quote=Decimal("1"),
                  source="ws")


MAR1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
UNTIL = datetime(2024, 3, 4, 12, 0, tzinfo=timezone.utc)
EXPECTED_MINUTES = 3 * 1440 + 720  # 03-01 00:00 .. 03-04 11:59


def test_ws_rows_today_with_no_history_leave_no_hole_in_the_current_month(db_conn):
    """The smoke-run state: the live collector stored two ws candles "now"
    and nothing else exists. The tail used to start at max(open_time) and
    silently skip everything before it."""
    from data.storage.repository import upsert_candles

    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [_ws_candle(datetime(2024, 3, 4, 11, 48, tzinfo=timezone.utc)),
                             _ws_candle(datetime(2024, 3, 4, 11, 49, tzinfo=timezone.utc))])
    dl = DailyDownloader({
        MAR1.date(): _day_zip(MAR1),
        (MAR1 + timedelta(days=1)).date(): _day_zip(MAR1 + timedelta(days=1)),
        # 03-03 not published yet; 03-04 is today.
    })
    api = ExchangeApi(listed=MAR1, now=UNTIL)

    backfill_symbol(db_conn, "BTCUSDT", dl, api, until=UNTIL)

    assert _minutes(db_conn, "BTCUSDT", MAR1, UNTIL) == EXPECTED_MINUTES
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM candles_1m WHERE symbol = 'BTCUSDT' "
                    "AND open_time >= %s", (UNTIL,))
        assert cur.fetchone()[0] == 0, "the forming candle must not be stored"


def test_a_daily_archive_404_mid_tail_does_not_become_a_permanent_hole(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    dl = DailyDownloader({
        MAR1.date(): _day_zip(MAR1),
        # 03-02 404s
        (MAR1 + timedelta(days=2)).date(): _day_zip(MAR1 + timedelta(days=2)),
    })
    api = ExchangeApi(listed=MAR1, now=UNTIL)

    backfill_symbol(db_conn, "BTCUSDT", dl, api, until=UNTIL)

    assert _minutes(db_conn, "BTCUSDT", MAR1, UNTIL) == EXPECTED_MINUTES


def test_rerun_re_walks_the_months_daily_archives_and_upgrades_rest_rows(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    day2 = MAR1 + timedelta(days=1)
    dl = DailyDownloader({MAR1.date(): _day_zip(MAR1)})  # 03-02 not published
    api = ExchangeApi(listed=MAR1, now=UNTIL)
    backfill_symbol(db_conn, "BTCUSDT", dl, api, until=UNTIL)

    dl.days[day2.date()] = _day_zip(day2)  # Binance publishes 03-02
    backfill_symbol(db_conn, "BTCUSDT", dl, api, until=UNTIL)

    with db_conn.cursor() as cur:
        cur.execute("SELECT source, count(*) FROM candles_1m WHERE symbol = "
                    "'BTCUSDT' AND open_time >= %s AND open_time < %s GROUP BY 1",
                    (day2, day2 + timedelta(days=1)))
        assert dict(cur.fetchall()) == {"archive": 1440}
    assert _minutes(db_conn, "BTCUSDT", MAR1, UNTIL) == EXPECTED_MINUTES


def test_backfill_tail_does_not_store_a_rest_candle_that_closed_under_two_seconds_ago(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    t_before = datetime.now(timezone.utc)
    start = t_before.replace(second=0, microsecond=0) - timedelta(minutes=5)
    rows = [_kline(start), _kline(start + timedelta(minutes=1)),
            _kline(t_before - timedelta(seconds=60, milliseconds=500))]

    class Api:
        def first_candle_time(self, symbol):
            return datetime(2017, 8, 17, tzinfo=timezone.utc)

        def klines(self, symbol, start_ms, limit=1000):
            return [r for r in rows if r[0] >= start_ms]

    backfill_symbol(db_conn, "BTCUSDT", DailyDownloader({}), Api(), start=start)

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM candles_1m WHERE symbol = 'BTCUSDT'")
        assert cur.fetchone()[0] == 2
