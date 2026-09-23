from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.quality.checks import DataQualityError, assert_trainable, run_quality_checks
from data.storage.repository import Candle, upsert_candles, upsert_symbol

pytestmark = pytest.mark.db

T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)


def candle(minute: int, **over):
    open_time = T0 + timedelta(minutes=minute)
    base = dict(
        symbol="BTCUSDT", open_time=open_time,
        close_time=open_time + timedelta(seconds=59),
        open=Decimal("100"), high=Decimal("110"), low=Decimal("90"),
        close=Decimal("105"), volume=Decimal("1"), quote_volume=Decimal("105"),
        trade_count=3, taker_buy_base=Decimal("0.5"),
        taker_buy_quote=Decimal("52.5"), source="archive",
    )
    base.update(over)
    return Candle(**base)


def test_clean_data_passes(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [candle(i) for i in range(120)])

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "PASS"
    assert report.missing == 0
    assert report.total_candles == 120
    assert "STATUS" in report.render()


def test_gap_reduces_completeness(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(200) if not (50 <= i < 100)]
    upsert_candles(db_conn, rows)

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.missing == 50
    assert report.completeness_pct < 100
    assert report.verdict in {"WARN", "FAIL"}


def test_invalid_candle_fails_and_blocks_training(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(10)]
    rows.append(candle(10, high=Decimal("50"), low=Decimal("90")))
    upsert_candles(db_conn, rows)

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "FAIL"
    assert report.invalid == 1

    with pytest.raises(DataQualityError, match="refusing to train"):
        assert_trainable(db_conn, ["BTCUSDT"])


def test_known_outage_is_not_counted_against_the_data(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(200) if not (50 <= i < 100)]
    upsert_candles(db_conn, rows)

    outage = [(T0 + timedelta(minutes=50), T0 + timedelta(minutes=100))]
    report = run_quality_checks(db_conn, "BTCUSDT", known_outages=outage)
    assert report.missing == 0
    assert report.verdict == "PASS"


def test_report_is_stored(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [candle(i) for i in range(10)])
    run_quality_checks(db_conn, "BTCUSDT")

    with db_conn.cursor() as cur:
        cur.execute("SELECT symbol, verdict FROM data_quality_reports")
        assert cur.fetchone()[0] == "BTCUSDT"


def test_no_data_fails_loudly_and_is_stored(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "FAIL"
    assert report.total_candles == 0
    assert report.details["reason"] == "no data"

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT verdict FROM data_quality_reports WHERE symbol = %s",
            ("BTCUSDT",),
        )
        assert cur.fetchone()[0] == "FAIL"

    with pytest.raises(DataQualityError, match="refusing to train"):
        assert_trainable(db_conn, ["BTCUSDT"])


def _counting_cursor_class():
    """A cursor_factory that counts every row reaching Python, however it
    is fetched."""
    import psycopg

    class CountingCursor(psycopg.Cursor):
        fetched = 0

        def fetchone(self):
            row = super().fetchone()
            if row is not None:
                CountingCursor.fetched += 1
            return row

        def fetchmany(self, size=0):
            rows = super().fetchmany(size)
            CountingCursor.fetched += len(rows)
            return rows

        def fetchall(self):
            rows = super().fetchall()
            CountingCursor.fetched += len(rows)
            return rows

        def __iter__(self):
            for row in super().__iter__():
                CountingCursor.fetched += 1
                yield row

    return CountingCursor


def test_quality_checks_never_pull_the_history_into_python(db_conn):
    """On BTCUSDT's ~4.8M rows the old engine materialized ~5 GB of Python
    dicts. Gaps and invalid counts must be computed in SQL."""
    upsert_symbol(db_conn, symbol="BTCUSDT")
    rows = [candle(i) for i in range(3000) if not (100 <= i < 110)]
    rows.append(candle(3000, high=Decimal("50"), low=Decimal("90")))   # invalid
    rows.append(candle(3001, volume=Decimal("-1")))                    # invalid
    upsert_candles(db_conn, rows)

    counting = _counting_cursor_class()
    db_conn.cursor_factory = counting
    report = run_quality_checks(db_conn, "BTCUSDT")

    assert report.total_candles == 2992
    assert report.missing == 10
    assert report.invalid == 2
    assert counting.fetched < 50, f"{counting.fetched} rows pulled into Python"


@pytest.mark.parametrize("over", [
    dict(high=Decimal("50"), low=Decimal("90")),       # high < low
    dict(high=Decimal("100"), close=Decimal("105")),   # high < close
    dict(high=Decimal("99")),                          # high < open
    dict(low=Decimal("101")),                          # low > open
    dict(volume=Decimal("-1")),                        # negative volume
    dict(trade_count=-1),                              # negative trade count
])
def test_each_impossible_candle_is_counted_invalid(db_conn, over):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    upsert_candles(db_conn, [candle(0), candle(1, **over), candle(2)])
    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.invalid == 1
    assert report.verdict == "FAIL"


def test_a_flat_zero_volume_candle_is_valid(db_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    flat = dict(open=Decimal("10"), high=Decimal("10"), low=Decimal("10"),
                close=Decimal("10"), volume=Decimal("0"))
    upsert_candles(db_conn, [candle(0), candle(1, **flat), candle(2)])
    assert run_quality_checks(db_conn, "BTCUSDT").invalid == 0


# --- C2: classify gaps; an unexplained hole over an hour blocks training -----

def _record_month(conn, status: str, rows: int, month_start=T0):
    from data.storage.repository import finish_run, start_run

    month_end = month_start.replace(month=month_start.month + 1)
    run_id = start_run(conn, "backfill", "BTCUSDT", month_start, month_end)
    finish_run(conn, run_id, status, rows)


def _history_with_gap(conn, gap_minutes: int, total: int = 3000, at: int = 1000):
    upsert_symbol(conn, symbol="BTCUSDT")
    upsert_candles(conn, [candle(i) for i in range(total)
                          if not (at <= i < at + gap_minutes)])


def test_an_unexplained_gap_over_an_hour_fails_and_blocks_training(db_conn):
    # 61 of 3000 minutes missing is ~98% complete: the old percentage rule
    # called this WARN and assert_trainable let it through without a word.
    _history_with_gap(db_conn, 61)

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "FAIL"
    assert report.missing == 61
    with pytest.raises(DataQualityError, match="refusing to train"):
        assert_trainable(db_conn, ["BTCUSDT"])


def test_a_gap_in_a_month_whose_archive_loaded_is_an_explained_outage(db_conn):
    # The monthly archive is Binance's own record: a gap inside it is the
    # exchange's outage, not ours.
    _history_with_gap(db_conn, 61)
    _record_month(db_conn, "success", 44579)

    report = run_quality_checks(db_conn, "BTCUSDT")
    assert report.verdict == "PASS"
    assert report.missing == 0
    assert report.details["explained_missing"] == 61
    assert assert_trainable(db_conn, ["BTCUSDT"]) == []


@pytest.mark.parametrize("status,rows", [("missing", 0), ("success", 0), ("failed", 0)])
def test_a_month_without_a_loaded_archive_explains_nothing(db_conn, status, rows):
    _history_with_gap(db_conn, 61)
    _record_month(db_conn, status, rows)

    assert run_quality_checks(db_conn, "BTCUSDT").verdict == "FAIL"


def test_a_gap_spilling_past_the_loaded_month_is_unexplained(db_conn):
    # Loaded month is April; the data and gap are in May.
    _history_with_gap(db_conn, 61)
    _record_month(db_conn, "success", 43200, month_start=T0.replace(month=4))

    assert run_quality_checks(db_conn, "BTCUSDT").verdict == "FAIL"


def test_exactly_sixty_unexplained_minutes_is_warn_not_fail(db_conn):
    _history_with_gap(db_conn, 60)
    assert run_quality_checks(db_conn, "BTCUSDT").verdict == "WARN"


def test_a_short_unexplained_gap_warns_and_assert_trainable_says_so(db_conn, caplog):
    import logging

    _history_with_gap(db_conn, 10)

    with caplog.at_level(logging.WARNING, logger="data.quality.checks"):
        warnings = assert_trainable(db_conn, ["BTCUSDT"])

    assert [r.verdict for r in warnings] == ["WARN"]
    assert warnings[0].symbol == "BTCUSDT"
    assert any("BTCUSDT" in rec.getMessage() and "WARN" in rec.getMessage()
               for rec in caplog.records), "a WARN must never pass silently"


def test_render_shows_where_the_largest_gaps_are(db_conn):
    _history_with_gap(db_conn, 61)
    text = run_quality_checks(db_conn, "BTCUSDT").render()

    gap_start = T0 + timedelta(minutes=1000)          # 2024-05-01 16:40
    gap_end = T0 + timedelta(minutes=1061)            # 2024-05-01 17:41
    assert f"{gap_start:%Y-%m-%d %H:%M}" in text
    assert f"{gap_end:%Y-%m-%d %H:%M}" in text
    assert "UNEXPLAINED" in text
