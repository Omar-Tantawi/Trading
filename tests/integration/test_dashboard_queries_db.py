"""Dashboard reads (spec section 3) against the test database."""
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.queries import candles, health
from features.build import build_features
from tests.integration.test_features_cli_db import DAY_MINUTES, T0, _insert

pytestmark = pytest.mark.db

NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)


@pytest.fixture
def seeded(db_conn):
    # 12 days of 1m candles, then the last bucket only partly covered
    _insert(db_conn, "BTCUSDT", range(0, 12 * DAY_MINUTES + 30))
    build_features(db_conn, "BTCUSDT", "1h")
    return db_conn


def test_candles_order_limit_and_features(seeded):
    out = candles(seeded, "BTCUSDT", "1h", 100)
    bars = out["bars"]
    assert len(bars) == 100
    times = [b["time"] for b in bars]
    assert times == sorted(times)
    last = datetime.fromtimestamp(times[-1], timezone.utc)
    # the 12th day's 00:00 bar is only 30 minutes covered: never shown
    assert last == T0 + timedelta(days=11, hours=23)
    assert set(bars[0]) == {"time", "open", "high", "low", "close", "volume",
                            "ema_200", "rsi_14"}
    assert bars[-1]["rsi_14"] is not None
    latest = out["latest"]
    assert latest["bar_close"] == (last + timedelta(hours=1)).isoformat()
    assert latest["trend_regime"] in {"sideways", "weak_bullish", "weak_bearish",
                                      "strong_bullish", "strong_bearish"}


def test_ema_200_null_in_warm_up(seeded):
    bars = candles(seeded, "BTCUSDT", "1h", 5000)["bars"]
    assert len(bars) == 12 * 24
    assert bars[0]["ema_200"] is None       # warm-up is 600 bars
    assert bars[-1]["ema_200"] is None


def test_symbol_without_data(db_conn):
    assert candles(db_conn, "ETHUSDT", "1h", 100) == {"bars": [], "latest": None}


def test_health(seeded):
    with seeded.cursor() as cur:
        cur.execute("INSERT INTO data_quality_reports (symbol, timeframe, verdict) "
                    "VALUES ('BTCUSDT', '1m', 'PASS')")
    seeded.commit()
    rows = health(seeded, ["BTCUSDT", "ETHUSDT"], NOW)
    btc, eth = rows
    assert btc["symbol"] == "BTCUSDT" and btc["stale"] is True
    assert btc["age_minutes"] > 60 * 24 * 365
    assert btc["features"]["1h"] is not None and btc["features"]["5m"] is None
    assert btc["verdict"] == "PASS" and btc["verdict_at"] is not None
    assert eth["last_1m"] is None and eth["stale"] is None and eth["verdict"] is None


def test_times_are_utc_whatever_the_session_zone(seeded):
    with seeded.cursor() as cur:
        cur.execute("SET TIME ZONE 'Asia/Tokyo'")
    out = candles(seeded, "BTCUSDT", "1h", 50)
    assert out["latest"]["bar_close"].endswith("+00:00")
    h = health(seeded, ["BTCUSDT"], NOW)[0]
    for value in (h["last_1m"], h["features"]["1h"]):
        assert value.endswith("+00:00")


def test_stale_age_matches_tb_status(seeded):
    last = T0 + timedelta(days=12, minutes=29)
    assert health(seeded, ["BTCUSDT"], last + timedelta(minutes=5))[0]["stale"] is False
    six = health(seeded, ["BTCUSDT"], last + timedelta(minutes=6))[0]
    assert six["stale"] is True and six["age_minutes"] == 6


def test_candles_same_rows_with_partial_features(seeded):
    # features exist only for part of the window: candles still all returned
    out = candles(seeded, "BTCUSDT", "1h", 300)
    assert len(out["bars"]) == 12 * 24
    assert out["bars"][0]["rsi_14"] is None and out["bars"][-1]["rsi_14"] is not None
