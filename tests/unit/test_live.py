from datetime import datetime, timedelta, timezone

import pytest

from data.collectors.live import minutes_missing


def test_minutes_missing_computes_the_gap():
    last = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    now = datetime(2024, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
    # candles at 12:01..12:04 are missing; 12:05 has not closed yet
    assert minutes_missing(last, now) == 4


def test_no_gap_when_current():
    last = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    now = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    assert minutes_missing(last, now) == 0


def test_requires_aware_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        minutes_missing(datetime(2024, 1, 1, 12, 0),
                        datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc))
