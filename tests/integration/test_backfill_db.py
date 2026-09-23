from datetime import datetime, timezone

import pytest

from data.collectors.backfill import backfill_symbol
from data.storage.repository import upsert_symbol

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
