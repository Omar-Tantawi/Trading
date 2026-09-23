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
