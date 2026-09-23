import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.storage.repository import Candle, get_candles, upsert_candles, upsert_symbol

pytestmark = pytest.mark.db

T0 = datetime(2024, 3, 1, tzinfo=timezone.utc)


def _random_candles(n: int) -> list[Candle]:
    random.seed(7)
    out = []
    for i in range(n):
        base = Decimal(random.randint(40000_00, 45000_00)) / 100
        high = base + Decimal("15.5")
        low = base - Decimal("12.25")
        open_time = T0 + timedelta(minutes=i)
        out.append(Candle(
            symbol="BTCUSDT", open_time=open_time,
            close_time=open_time + timedelta(seconds=59),
            open=base, high=high, low=low, close=base + Decimal("1.5"),
            volume=Decimal(i + 1), quote_volume=Decimal((i + 1) * 100),
            trade_count=i + 1, taker_buy_base=Decimal(i),
            taker_buy_quote=Decimal(i * 10), source="archive",
        ))
    return out


def _expected_hourly(candles: list[Candle]) -> dict:
    """Independent hourly aggregation in plain Python, exact on Decimals."""
    buckets: dict[datetime, list[Candle]] = defaultdict(list)
    for c in candles:
        buckets[c.open_time.replace(minute=0, second=0, microsecond=0)].append(c)
    out = {}
    for hour, group in buckets.items():
        group.sort(key=lambda c: c.open_time)
        out[hour] = {
            "open": group[0].open,
            "high": max(c.high for c in group),
            "low": min(c.low for c in group),
            "close": group[-1].close,
            "volume": sum((c.volume for c in group), Decimal(0)),
            "trade_count": sum(c.trade_count for c in group),
        }
    return out


def test_database_1h_candles_match_independent_computation(db_conn, autocommit_conn):
    upsert_symbol(db_conn, symbol="BTCUSDT")
    candles = _random_candles(180)  # exactly 3 hours
    upsert_candles(db_conn, candles)

    # Must run outside a transaction block, hence the autocommit connection.
    with autocommit_conn.cursor() as cur:
        cur.execute("CALL refresh_continuous_aggregate('candles_1h', NULL, NULL)")

    rows = get_candles(db_conn, "BTCUSDT", "1h", T0, T0 + timedelta(hours=3))
    assert len(rows) == 3

    expected = _expected_hourly(candles)
    for row in rows:
        want = expected[row["open_time"]]
        assert row["open"] == want["open"]
        assert row["high"] == want["high"]
        assert row["low"] == want["low"]
        assert row["close"] == want["close"]
        assert row["volume"] == want["volume"]
        assert row["trade_count"] == want["trade_count"]
