from datetime import datetime, timedelta, timezone

import pytest

from bots.compare import compare, totals
from bots.times import parse_time

PLUS3 = timezone(timedelta(hours=3))


def test_ahead_of_holding():
    # coin flat: holding loses only its buy fee; the bot made 10
    c = compare(invest=500, fee=0.001, profit=10, p_start=100, p_end=100)
    assert c.hold_profit == pytest.approx(-0.5)
    assert c.diff == pytest.approx(10.5)
    assert c.diff_pts == pytest.approx(0.021)
    assert c.bot_pct == pytest.approx(0.02) and c.verdict == "ahead"


def test_behind_holding_when_the_coin_rises():
    c = compare(invest=1000, fee=0.00075, profit=30, p_start=100, p_end=120)
    hold = 1000 * (1 - 0.00075) * 1.2 - 1000
    assert c.hold_profit == pytest.approx(hold)
    assert c.hold_pct == pytest.approx(hold / 1000)
    assert c.diff == pytest.approx(30 - hold) and c.verdict == "behind"


def test_loss_smaller_than_holding_is_ahead():
    c = compare(invest=1000, fee=0.0, profit=-50, p_start=100, p_end=80)
    assert c.hold_profit == pytest.approx(-200) and c.verdict == "ahead"


def test_level():
    c = compare(invest=1000, fee=0.0, profit=100.004, p_start=100, p_end=110)
    assert c.verdict == "level"


def test_totals_skip_bots_without_a_comparison():
    a = compare(invest=500, fee=0.0, profit=10, p_start=100, p_end=100)
    b = compare(invest=1000, fee=0.0, profit=-20, p_start=100, p_end=110)
    t = totals([(500, a), (1000, b), (300, None), (200, None)])
    assert (t.n_bots, t.n_compared) == (4, 2)
    assert t.invested == 1500
    assert t.bot_profit == pytest.approx(-10)
    assert t.hold_profit == pytest.approx(100)
    assert t.diff == pytest.approx(-110)
    assert t.bot_pct == pytest.approx(-10 / 1500)
    assert t.diff_pts == pytest.approx(-110 / 1500) and t.verdict == "behind"


def test_totals_of_nothing():
    t = totals([(300, None)])
    assert t.n_compared == 0 and t.invested == 0 and t.verdict is None


def test_parse_local_time():
    assert parse_time("2026-10-14 09:30", utc=False, local_tz=PLUS3) == \
        datetime(2026, 10, 14, 6, 30, tzinfo=timezone.utc)
    assert parse_time(" 2026-10-14T09:30 ", utc=False, local_tz=PLUS3) == \
        datetime(2026, 10, 14, 6, 30, tzinfo=timezone.utc)


def test_parse_utc_time():
    assert parse_time("2026-10-14 09:30", utc=True) == \
        datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)


def test_parse_uses_the_pc_zone_by_default():
    t = parse_time("2026-10-14 09:30", utc=False)
    assert t.tzinfo == timezone.utc
    assert t == datetime(2026, 10, 14, 9, 30).astimezone().astimezone(timezone.utc)


@pytest.mark.parametrize("text", ["14/10/2026 09:30", "2026-10-14", "yesterday", ""])
def test_parse_rejects_other_formats(text):
    with pytest.raises(ValueError, match="YYYY-MM-DD HH:MM"):
        parse_time(text, utc=False, local_tz=PLUS3)


def test_parse_out_of_range_year_is_a_clean_error():
    with pytest.raises(ValueError):
        parse_time("1926-10-14 09:30", utc=False)
    with pytest.raises(ValueError):
        parse_time("1926-10-14 09:30", utc=True)


def test_level_is_relative_to_the_investment():
    # 0.05 USDT on 1,000,000 USDT is 0.000005 pts: level
    c = compare(invest=1_000_000, fee=0.0, profit=0.05, p_start=100, p_end=100)
    assert c.verdict == "level"
