from datetime import datetime, timedelta, timezone
from decimal import Decimal

from data.quality.checks import find_gaps, find_invalid, verdict_for

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
MIN = timedelta(minutes=1)


def row(**over):
    base = dict(open_time=T0, open=Decimal("10"), high=Decimal("12"),
                low=Decimal("9"), close=Decimal("11"), volume=Decimal("1"),
                trade_count=5)
    base.update(over)
    return base


def test_find_invalid_catches_high_below_low():
    bad = row(high=Decimal("8"), low=Decimal("9"))
    assert find_invalid([row(), bad]) == [bad]


def test_find_invalid_catches_high_below_close():
    bad = row(high=Decimal("10"), close=Decimal("11"))
    assert find_invalid([bad]) == [bad]


def test_find_invalid_catches_negative_volume():
    bad = row(volume=Decimal("-1"))
    assert find_invalid([bad]) == [bad]


def test_find_invalid_accepts_a_flat_candle():
    flat = row(open=Decimal("10"), high=Decimal("10"),
               low=Decimal("10"), close=Decimal("10"), volume=Decimal("0"))
    assert find_invalid([flat]) == []


def test_find_gaps_returns_missing_ranges():
    times = [T0, T0 + MIN, T0 + 5 * MIN]
    assert find_gaps(times, MIN) == [(T0 + 2 * MIN, T0 + 5 * MIN)]


def test_find_gaps_empty_when_contiguous():
    times = [T0 + i * MIN for i in range(10)]
    assert find_gaps(times, MIN) == []


def test_verdict_thresholds():
    assert verdict_for(completeness=100.0, invalid=0, duplicates=0) == "PASS"
    assert verdict_for(completeness=99.99, invalid=0, duplicates=0) == "PASS"
    assert verdict_for(completeness=99.0, invalid=0, duplicates=0) == "WARN"
    assert verdict_for(completeness=100.0, invalid=1, duplicates=0) == "FAIL"
    assert verdict_for(completeness=80.0, invalid=0, duplicates=0) == "FAIL"


def test_known_outage_minutes_do_not_count_as_missing():
    """Binance has had real multi-hour outages. Those minutes exist nowhere,
    so they must not keep assert_trainable red forever."""
    from data.quality.checks import subtract_known_outages

    gaps = [(T0 + 2 * MIN, T0 + 5 * MIN)]
    outages = [(T0 + 2 * MIN, T0 + 5 * MIN)]
    assert subtract_known_outages(gaps, outages) == []
