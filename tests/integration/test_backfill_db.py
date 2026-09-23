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
