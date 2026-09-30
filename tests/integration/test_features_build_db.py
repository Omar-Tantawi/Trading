import logging
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from data.storage.repository import (
    Candle, refresh_aggregates, upsert_candles, upsert_symbol,
)
from features import build as build_module
from features.build import build_features, build_symbol
from features.frame import build_cutoff, load_bars
from features.pipeline import FEATURE_TIMEFRAMES, compute_features
from features.store import has_stale_feature_set, read_features

pytestmark = pytest.mark.db

SYMBOL = "BTCUSDT"
# Well in the past, so every bucket is closed.
T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
FIVE_MIN = timedelta(minutes=5)
DAY_MINUTES = 24 * 60


def _walk(minutes):
    """Minute candles from a smooth deterministic walk (valid OHLC)."""
    out = []
    for m in minutes:
        open_time = T0 + timedelta(minutes=m)
        base = 100 + 8 * math.sin(m / 700) + 3 * math.sin(m / 53) + m / 4000
        o = Decimal(f"{base:.2f}")
        c = Decimal(f"{base + 0.4 * math.sin(m / 3):.2f}")
        out.append(Candle(
            symbol=SYMBOL, open_time=open_time,
            close_time=open_time + timedelta(seconds=59),
            open=o, high=max(o, c) + Decimal("0.25"),
            low=min(o, c) - Decimal("0.25"), close=c,
            volume=Decimal(f"{2 + math.sin(m / 11):.3f}"),
            quote_volume=Decimal("200"), trade_count=5,
            taker_buy_base=Decimal(f"{1 + 0.5 * math.sin(m / 7):.3f}"),
            taker_buy_quote=Decimal("100"), source="archive"))
    return out


def _insert(conn, minutes):
    upsert_symbol(conn, symbol=SYMBOL)
    upsert_candles(conn, _walk(minutes))
    conn.commit()


def _count(conn, tf):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM features_{tf} WHERE symbol = %s",
                    (SYMBOL,))
        return cur.fetchone()[0]


# --- cutoff and loading ------------------------------------------------------

def test_build_cutoff_excludes_partial_bucket(db_conn):
    assert build_cutoff(db_conn, SYMBOL, "5m") is None
    _insert(db_conn, range(0, 59))  # 00:00 ... 00:58
    assert build_cutoff(db_conn, SYMBOL, "5m") == T0 + timedelta(minutes=55)
    assert build_cutoff(db_conn, SYMBOL, "15m") == T0 + timedelta(minutes=45)
    assert build_cutoff(db_conn, SYMBOL, "1h") == T0
    assert build_cutoff(db_conn, SYMBOL, "1d") == T0


def test_build_cutoff_includes_a_bucket_ending_at_the_last_candle(db_conn):
    _insert(db_conn, range(0, 60))  # 00:00 ... 00:59, the hour is complete
    assert build_cutoff(db_conn, SYMBOL, "1h") == T0 + HOUR
    assert build_cutoff(db_conn, SYMBOL, "5m") == T0 + HOUR


def test_load_bars_returns_floats_on_utc_index(db_conn):
    _insert(db_conn, range(0, 15))
    refresh_aggregates(db_conn, T0, T0 + timedelta(minutes=15))
    candles = _walk(range(0, 5))

    bars = load_bars(db_conn, SYMBOL, "5m", None, T0 + timedelta(minutes=15))

    assert list(bars.columns) == [
        "open", "high", "low", "close", "volume", "taker_buy_base"]
    assert all(dt == "float64" for dt in bars.dtypes)
    assert bars.index.name == "open_time"
    assert str(bars.index.tz) == "UTC"
    assert list(bars.index) == [T0 + i * FIVE_MIN for i in range(3)]
    first = bars.iloc[0]
    assert first["open"] == float(candles[0].open)
    assert first["close"] == float(candles[4].close)
    assert first["high"] == float(max(c.high for c in candles))
    assert first["low"] == float(min(c.low for c in candles))
    assert first["volume"] == pytest.approx(float(sum(c.volume for c in candles)))
    assert first["taker_buy_base"] == pytest.approx(
        float(sum(c.taker_buy_base for c in candles)))


def test_load_bars_bounds_are_start_inclusive_end_exclusive(db_conn):
    _insert(db_conn, range(0, 30))
    refresh_aggregates(db_conn, T0, T0 + timedelta(minutes=30))

    bars = load_bars(db_conn, SYMBOL, "5m", T0 + FIVE_MIN, T0 + 4 * FIVE_MIN)
    assert list(bars.index) == [T0 + FIVE_MIN, T0 + 2 * FIVE_MIN,
                                T0 + 3 * FIVE_MIN]

    later = T0 + timedelta(days=400)
    empty = load_bars(db_conn, SYMBOL, "5m", later, later + HOUR)
    assert empty.empty and str(empty.index.tz) == "UTC"
    assert list(empty.columns) == list(bars.columns)


# --- building ----------------------------------------------------------------

def test_first_build_is_full_then_incremental_only_adds(db_conn, autocommit_conn):
    _insert(db_conn, range(0, 3 * DAY_MINUTES))

    first = build_features(db_conn, SYMBOL, "1h")
    assert first.full is True
    assert first.rows_written == 72
    assert first.start == T0 and first.end == T0 + 71 * HOUR
    assert first.skipped_reason is None
    # committed: visible from a different connection
    assert _count(autocommit_conn, "1h") == 72

    _insert(db_conn, range(3 * DAY_MINUTES, 3 * DAY_MINUTES + 6 * 60))
    second = build_features(db_conn, SYMBOL, "1h")
    assert second.full is False
    assert second.rows_written == 6
    assert second.start == T0 + 72 * HOUR
    assert second.end == T0 + 77 * HOUR
    assert _count(db_conn, "1h") == 78


def test_second_build_without_new_data_writes_nothing(db_conn):
    _insert(db_conn, range(0, 2 * DAY_MINUTES))
    build_features(db_conn, SYMBOL, "1h")

    again = build_features(db_conn, SYMBOL, "1h")

    assert again.rows_written == 0
    assert again.start is None and again.end is None
    assert again.full is False
    assert _count(db_conn, "1h") == 48


def test_a_partial_bucket_is_not_built(db_conn):
    _insert(db_conn, range(0, 150))  # 00:00 ... 02:29

    result = build_features(db_conn, SYMBOL, "1h")

    assert result.rows_written == 2
    assert result.end == T0 + HOUR


def test_build_without_candles_is_skipped(db_conn):
    upsert_symbol(db_conn, symbol=SYMBOL)

    result = build_features(db_conn, SYMBOL, "1h")

    assert result.rows_written == 0
    assert result.skipped_reason == "no 1m candles"


def test_incremental_equals_rebuild(db_conn, assert_features_match):
    # 5m: 9 days is 2592 bars, more than the 2100-bar lookback, so the
    # incremental build really does load a truncated window.
    _insert(db_conn, range(0, 9 * DAY_MINUTES))
    assert build_features(db_conn, SYMBOL, "5m").full is True
    _insert(db_conn, range(9 * DAY_MINUTES, 10 * DAY_MINUTES))

    incremental = build_features(db_conn, SYMBOL, "5m")

    assert incremental.full is False and incremental.rows_written == 288
    cutoff = build_cutoff(db_conn, SYMBOL, "5m")
    expected = compute_features(
        load_bars(db_conn, SYMBOL, "5m", None, cutoff), FIVE_MIN)
    stored = read_features(db_conn, SYMBOL, "5m")
    assert len(stored) == 10 * 288
    assert_features_match(stored, expected)


def test_stale_feature_set_forces_full_rebuild(db_conn):
    _insert(db_conn, range(0, 2 * DAY_MINUTES))
    build_features(db_conn, SYMBOL, "1h")
    with db_conn.cursor() as cur:
        cur.execute("UPDATE features_1h SET feature_set = 0 "
                    "WHERE open_time = %s", (T0 + 5 * HOUR,))
    db_conn.commit()
    assert has_stale_feature_set(db_conn, SYMBOL, "1h")

    result = build_features(db_conn, SYMBOL, "1h")

    assert result.full is True
    assert result.rows_written == 48
    assert not has_stale_feature_set(db_conn, SYMBOL, "1h")


def test_rebuild_flag_rewrites_everything(db_conn):
    _insert(db_conn, range(0, 2 * DAY_MINUTES))
    build_features(db_conn, SYMBOL, "1h")

    result = build_features(db_conn, SYMBOL, "1h", rebuild=True)

    assert result.full is True and result.rows_written == 48
    assert _count(db_conn, "1h") == 48


def test_build_works_without_prior_aggregate_refresh(db_conn):
    end = T0 + timedelta(days=2)
    # db_conn empties candles_1m but not the aggregates: clear whatever earlier
    # tests left materialized, while candles_1m is empty.
    refresh_aggregates(db_conn, T0, end)
    assert load_bars(db_conn, SYMBOL, "1h", None, end).empty

    _insert(db_conn, range(0, 2 * DAY_MINUTES))
    # the aggregate does not compute unmaterialized buckets on the fly
    assert load_bars(db_conn, SYMBOL, "1h", None, end).empty

    result = build_features(db_conn, SYMBOL, "1h")

    assert result.rows_written == 48
    assert _count(db_conn, "1h") == 48


def test_a_write_larger_than_one_batch_lands_every_row(db_conn, monkeypatch,
                                                       assert_features_match):
    _insert(db_conn, range(0, 3 * DAY_MINUTES))
    monkeypatch.setattr(build_module, "WRITE_BATCH_ROWS", 10)
    calls = []
    real = build_module.upsert_features

    def spy(conn, symbol, timeframe, frame):
        calls.append(len(frame))
        return real(conn, symbol, timeframe, frame)

    monkeypatch.setattr(build_module, "upsert_features", spy)

    result = build_features(db_conn, SYMBOL, "1h")

    assert calls == [10] * 7 + [2]
    assert result.rows_written == 72
    assert _count(db_conn, "1h") == 72
    expected = compute_features(
        load_bars(db_conn, SYMBOL, "1h", None, T0 + 72 * HOUR), HOUR)
    assert_features_match(read_features(db_conn, SYMBOL, "1h"), expected)


# --- build_symbol and the quality gate ----------------------------------------

def test_quality_fail_skips_symbol(db_conn):
    # a 3-hour hole in one day of 1m data, in a month with no archive record
    _insert(db_conn, [m for m in range(0, DAY_MINUTES) if not 600 <= m < 780])

    results = build_symbol(db_conn, SYMBOL)

    assert [r.timeframe for r in results] == list(FEATURE_TIMEFRAMES)
    assert all(r.rows_written == 0 for r in results)
    assert all(r.skipped_reason == "data quality FAIL" for r in results)
    for tf in FEATURE_TIMEFRAMES:
        assert _count(db_conn, tf) == 0


def test_quality_pass_builds_every_timeframe(db_conn):
    _insert(db_conn, range(0, 3 * DAY_MINUTES))

    results = build_symbol(db_conn, SYMBOL)

    rows = {r.timeframe: r.rows_written for r in results}
    assert rows == {"5m": 864, "15m": 288, "1h": 72, "4h": 18, "1d": 3}
    assert all(r.skipped_reason is None and r.full for r in results)


def test_one_failing_timeframe_does_not_stop_the_others(db_conn, monkeypatch,
                                                        caplog):
    _insert(db_conn, range(0, 3 * DAY_MINUTES))
    real = build_module.build_features

    def flaky(conn, symbol, timeframe, **kwargs):
        if timeframe == "1h":
            raise RuntimeError("boom")
        return real(conn, symbol, timeframe, **kwargs)

    monkeypatch.setattr(build_module, "build_features", flaky)

    with caplog.at_level(logging.ERROR, logger="features.build"):
        results = build_symbol(db_conn, SYMBOL)

    by_tf = {r.timeframe: r for r in results}
    assert list(by_tf) == list(FEATURE_TIMEFRAMES)
    assert by_tf["1h"].rows_written == 0
    assert by_tf["1h"].skipped_reason == "error: boom"
    assert _count(db_conn, "1h") == 0
    assert {tf: by_tf[tf].rows_written for tf in ("5m", "15m", "4h", "1d")} == {
        "5m": 864, "15m": 288, "4h": 18, "1d": 3}
    assert any("1h" in r.getMessage() for r in caplog.records)


def test_quality_warn_logs_and_builds(db_conn, caplog):
    # a 10-minute unexplained hole: WARN, not FAIL
    _insert(db_conn, [m for m in range(0, 2 * DAY_MINUTES) if not 600 <= m < 610])

    with caplog.at_level(logging.WARNING, logger="features.build"):
        results = build_symbol(db_conn, SYMBOL, ("1h",))

    assert results[0].rows_written == 48
    assert any("WARN" in r.getMessage() for r in caplog.records)
