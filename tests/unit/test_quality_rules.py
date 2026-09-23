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
    from data.quality.checks import classify_gaps

    gaps = [(T0 + 2 * MIN, T0 + 5 * MIN)]
    outages = [(T0 + 2 * MIN, T0 + 5 * MIN)]
    assert classify_gaps(gaps, loaded_months=set(), known_outages=outages) \
        == (gaps, [])


def test_an_unexplained_gap_over_the_limit_fails():
    from data.quality.checks import MAX_UNEXPLAINED_GAP

    assert MAX_UNEXPLAINED_GAP == timedelta(minutes=60)
    assert verdict_for(completeness=99.99, invalid=0, duplicates=0,
                       unexplained_gaps=1,
                       longest_unexplained=timedelta(minutes=61)) == "FAIL"
    assert verdict_for(completeness=99.99, invalid=0, duplicates=0,
                       unexplained_gaps=1,
                       longest_unexplained=timedelta(minutes=60)) == "WARN"
    assert verdict_for(completeness=100.0, invalid=0, duplicates=0,
                       unexplained_gaps=1,
                       longest_unexplained=timedelta(minutes=1)) == "WARN"


def test_classify_gaps_by_loaded_month_and_known_outage():
    from data.quality.checks import classify_gaps

    may = datetime(2024, 5, 1, tzinfo=timezone.utc)
    jun = datetime(2024, 6, 1, tzinfo=timezone.utc)
    in_may = (may + MIN, may + 90 * MIN)
    across = (jun - MIN, jun + MIN)                 # May into June
    in_june = (jun + 5 * MIN, jun + 9 * MIN)
    outage = (jun + 4 * MIN, jun + 10 * MIN)

    explained, unexplained = classify_gaps(
        [in_may, across, in_june], loaded_months={may}, known_outages=[outage],
    )
    assert explained == [in_may, in_june]
    assert unexplained == [across]
