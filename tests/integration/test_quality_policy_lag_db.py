"""N1: a bucket that just closed is not a hole until the refresh policy has
had time to materialize it (spec section 8)."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.quality.checks import POLICY_LAG, run_quality_checks
from data.storage.repository import (Candle, refresh_aggregates, upsert_candles,
                                     upsert_symbol)

pytestmark = pytest.mark.db

SYMBOL = "BTCUSDT"
# Far older than every policy's start_offset (the widest is 365 days), so no
# scheduled policy job ever materializes this range behind the test's back.
T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)
TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")


def _candle(minute: int) -> Candle:
    open_time = T0 + timedelta(minutes=minute)
    return Candle(
        symbol=SYMBOL, open_time=open_time,
        close_time=open_time + timedelta(seconds=59),
        open=Decimal("100"), high=Decimal("110"), low=Decimal("90"),
        close=Decimal("105"), volume=Decimal("1"), quote_volume=Decimal("105"),
        trade_count=3, taker_buy_base=Decimal("0.5"),
        taker_buy_quote=Decimal("52.5"), source="archive",
    )


def _clear_views(autocommit_conn) -> None:
    """db_conn deletes the 1m rows but that only logs an invalidation: rows an
    earlier test materialized would linger. Refreshing the now-empty range
    clears them."""
    with autocommit_conn.cursor() as cur:
        for tf in TIMEFRAMES:
            cur.execute("CALL refresh_continuous_aggregate(%s, %s, %s)",
                        (f"candles_{tf}", T0, T0 + timedelta(days=2)))


def _load_1m_but_materialize_only_the_first_4h(db_conn, autocommit_conn) -> None:
    """1m candles 00:00 through 08:04; the 4h view refreshed over [00:00, 04:00)
    only, so the 00:00 bucket is materialized and the 04:00 bucket (complete in
    the 1m data) is not."""
    _clear_views(autocommit_conn)
    upsert_symbol(db_conn, symbol=SYMBOL)
    upsert_candles(db_conn, [_candle(i) for i in range(8 * 60 + 5)])
    refresh_aggregates(db_conn, T0, T0 + timedelta(hours=4),
                       now=T0 + timedelta(hours=8, minutes=5))
    with autocommit_conn.cursor() as cur:
        cur.execute("SELECT open_time FROM candles_4h WHERE symbol = %s "
                    "AND open_time >= %s AND open_time < %s ORDER BY 1",
                    (SYMBOL, T0, T0 + timedelta(days=1)))
        assert [r[0] for r in cur.fetchall()] == [T0]  # 04:00 stays invisible


def test_policy_lag_matches_registered_policies(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT hypertable_name, schedule_interval, "
            "(config->>'end_offset')::interval "
            "FROM timescaledb_information.jobs "
            "WHERE proc_name = 'policy_refresh_continuous_aggregate'"
        )
        policies = {name: (schedule, end_offset)
                    for name, schedule, end_offset in cur.fetchall()}

    assert set(POLICY_LAG) == set(TIMEFRAMES)
    for tf in TIMEFRAMES:
        schedule, end_offset = policies[f"candles_{tf}"]
        assert POLICY_LAG[tf] == schedule + end_offset + timedelta(minutes=5), tf


def test_freshly_closed_4h_bucket_is_not_a_hole(db_conn, autocommit_conn):
    _load_1m_but_materialize_only_the_first_4h(db_conn, autocommit_conn)

    report = run_quality_checks(db_conn, SYMBOL, "4h",
                                now=T0 + timedelta(hours=8, minutes=5))

    assert report.verdict == "PASS"
    assert report.missing == 0
    assert report.checked_to == T0


def test_bucket_missing_beyond_the_lag_still_counts(db_conn, autocommit_conn):
    _load_1m_but_materialize_only_the_first_4h(db_conn, autocommit_conn)

    report = run_quality_checks(db_conn, SYMBOL, "4h",
                                now=T0 + timedelta(hours=10))

    assert report.missing == 1
    assert report.verdict == "FAIL"
    assert report.checked_to == T0 + timedelta(hours=4)


def test_the_lag_never_pushes_the_end_past_the_1m_data(db_conn, autocommit_conn):
    """A late clock must not extend the range beyond the last complete bucket
    the 1m data supports."""
    _load_1m_but_materialize_only_the_first_4h(db_conn, autocommit_conn)

    report = run_quality_checks(db_conn, SYMBOL, "4h", now=T0 + timedelta(days=30))

    assert report.checked_to == T0 + timedelta(hours=4)
