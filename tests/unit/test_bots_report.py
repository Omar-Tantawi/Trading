import re
from datetime import datetime, timedelta, timezone

from bots.compare import compare, totals
from bots.report import render_list, render_show

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)
T0 = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)


def row(**kw):
    base = dict(id=1, name="BTCUSDT grid", symbol="BTCUSDT", status="running",
                start_time=T0, invest_usdt=500.0, range_low=None, range_high=None,
                grids=None, at=T0 + timedelta(days=10), total_profit=12.3,
                days=10.0, comparison=compare(500, 0.0, 12.3, 100, 101), note=None)
    return {**base, **kw}


def test_list_lines_and_totals():
    rows = [row(),
            row(id=2, name="eth one", symbol="ETHUSDT", status="stopped",
                total_profit=-4.0, comparison=compare(1000, 0.0, -4.0, 100, 90)),
            row(id=3, at=None, total_profit=None, days=None, comparison=None,
                note="no update yet (tb bots update 3 --profit X)"),
            row(id=4, comparison=None,
                note="no stored price for SOLUSDT near 2026-10-11 09:00 UTC")]
    out = render_list(rows, totals([(r["invest_usdt"], r["comparison"]) for r in rows]))
    assert "#1 BTCUSDT grid (BTCUSDT), running since 2026-10-01 09:00 UTC, 500.00 USDT" in out
    assert "bot +12.30 USDT (+2.46%), holding +5.00 USDT (+1.00%)" in out
    assert "bot ahead of holding by 7.30 USDT (+1.46 pts)" in out
    assert "#2 eth one (ETHUSDT), stopped" in out
    assert "bot ahead of holding by 96.00 USDT" in out
    assert "no update yet (tb bots update 3 --profit X)" in out
    assert "no stored price for SOLUSDT" in out
    assert "All bots (2 of 4 compared, 1,000.00 USDT invested)" in out
    assert "Total profit" in out and "Not advice." in out
    assert not ADVICE.search(out)


def test_list_empty():
    out = render_list([], totals([]))
    assert "no bots yet" in out and "tb bots add" in out


def test_behind_and_level_words():
    out = render_list([row(comparison=compare(500, 0.0, 1.0, 100, 110))],
                      totals([(500, compare(500, 0.0, 1.0, 100, 110))]))
    assert "bot behind holding by 49.00 USDT (-9.80 pts)" in out
    lvl = compare(500, 0.0, 10.0, 100, 102)
    assert "bot level with holding" in render_list([row(comparison=lvl)], totals([(500, lvl)]))


def test_show_history():
    bot = row(range_low=90.0, range_high=110.0, grids=10)
    hist = [dict(at=T0 + timedelta(days=1), total_profit=2.0, final=False, days=1.0,
                 comparison=compare(500, 0.0, 2.0, 100, 99), note=None),
            dict(at=T0 + timedelta(days=2), total_profit=3.0, final=True, days=2.0,
                 comparison=None, note="no stored price for BTCUSDT near ...")]
    out = render_show(bot, hist)
    assert "range 90.00 to 110.00, 10 grids" in out
    assert "2026-10-02 09:00 UTC" in out and "(stop)" in out
    assert "no stored price" in out
    assert not ADVICE.search(out)
