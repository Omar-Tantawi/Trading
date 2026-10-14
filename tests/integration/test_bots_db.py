"""Bots, updates and prices against the test database."""
from datetime import datetime, timedelta, timezone

import pytest

from bots.store import (
    add_bot, add_update, delete_bot, get_bot, list_bots, undo, updates,
)

pytestmark = pytest.mark.db

T0 = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
NOW = T0 + timedelta(days=10)


def _bot(conn, **kw):
    args = dict(symbol="BTCUSDT", invest=500.0, start=T0, fee=0.00075, now=NOW)
    return add_bot(conn, **{**args, **kw})


def test_add_and_read(db_conn):
    bid = _bot(db_conn, low=90.0, high=110.0, grids=10)
    b = get_bot(db_conn, bid)
    assert b["name"] == "BTCUSDT grid" and b["invest_usdt"] == 500.0
    assert b["start_time"] == T0 and b["stopped_at"] is None and b["fee"] == 0.00075
    assert (b["range_low"], b["range_high"], b["grids"]) == (90.0, 110.0, 10)
    assert [x["id"] for x in list_bots(db_conn)] == [bid]


@pytest.mark.parametrize("kw, msg", [
    ({"symbol": "BTCEUR"}, "USDT"), ({"symbol": "btc"}, "USDT"),
    ({"invest": 0.0}, "invest"), ({"start": NOW + timedelta(minutes=1)}, "future"),
    ({"low": 90.0}, "both"), ({"low": 110.0, "high": 90.0}, "low"),
    ({"grids": 1}, "grids"), ({"grids": 501}, "grids"),
])
def test_add_rejects(db_conn, kw, msg):
    with pytest.raises(ValueError, match=msg):
        _bot(db_conn, **kw)
    assert list_bots(db_conn) == []


def test_updates_stop_and_rules(db_conn):
    bid = _bot(db_conn)
    add_update(db_conn, bid, T0 + timedelta(days=1), 5.0, final=False, now=NOW)
    add_update(db_conn, bid, T0 + timedelta(days=1), 6.0, final=False, now=NOW)   # same time ok
    for at, profit, msg in [
        (T0 - timedelta(minutes=1), 1.0, "before the bot's start"),
        (T0 + timedelta(hours=1), 1.0, "before its previous update"),
        (NOW + timedelta(minutes=1), 1.0, "future"),
        (T0 + timedelta(days=2), -500.0, "more than"),
    ]:
        with pytest.raises(ValueError, match=msg):
            add_update(db_conn, bid, at, profit, final=False, now=NOW)
    add_update(db_conn, bid, T0 + timedelta(days=3), 9.0, final=True, now=NOW)
    assert get_bot(db_conn, bid)["stopped_at"] == T0 + timedelta(days=3)
    with pytest.raises(ValueError, match="stopped"):
        add_update(db_conn, bid, T0 + timedelta(days=4), 9.0, final=False, now=NOW)
    assert [u["total_profit"] for u in updates(db_conn, bid)] == [5.0, 6.0, 9.0]


def test_unknown_bot(db_conn):
    with pytest.raises(ValueError, match="no bot 99"):
        add_update(db_conn, 99, T0, 1.0, final=False, now=NOW)
    assert undo(db_conn, 99) is None and delete_bot(db_conn, 99) is False


def test_undo_reopens_a_stopped_bot(db_conn):
    bid = _bot(db_conn)
    add_update(db_conn, bid, T0 + timedelta(days=1), 5.0, final=False, now=NOW)
    add_update(db_conn, bid, T0 + timedelta(days=2), 7.0, final=True, now=NOW)
    removed = undo(db_conn, bid)
    assert removed["total_profit"] == 7.0 and removed["final"]
    assert get_bot(db_conn, bid)["stopped_at"] is None
    add_update(db_conn, bid, T0 + timedelta(days=2), 8.0, final=False, now=NOW)
    assert undo(db_conn, bid)["total_profit"] == 8.0
    assert undo(db_conn, bid)["total_profit"] == 5.0
    assert undo(db_conn, bid) is None


def test_delete_cascades(db_conn):
    bid = _bot(db_conn)
    other = _bot(db_conn, symbol="ETHUSDT", name="eth one")
    add_update(db_conn, bid, T0 + timedelta(days=1), 5.0, final=False, now=NOW)
    assert delete_bot(db_conn, bid) is True
    assert get_bot(db_conn, bid) is None and updates(db_conn, bid) == []
    assert [b["name"] for b in list_bots(db_conn)] == ["eth one"]
