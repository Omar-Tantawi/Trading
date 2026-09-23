from datetime import datetime, timedelta, timezone

from data.quality.checks import verdict_for

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
MIN = timedelta(minutes=1)


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
