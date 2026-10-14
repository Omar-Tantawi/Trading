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
    ({"invest": 0.0}, "invest"), ({"invest": 0.5}, "at least 1"),
    ({"invest": float("nan")}, "number"), ({"invest": float("inf")}, "number"),
    ({"low": float("nan"), "high": 2.0}, "number"),
    ({"start": NOW + timedelta(minutes=1)}, "future"),
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
        (T0 + timedelta(days=2), -500.0, "would leave nothing"),
        (T0 + timedelta(days=2), -600.0, "would leave nothing"),
        (T0 + timedelta(days=2), float("nan"), "number"),
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


# ---- prices and rows

from bots.compare import compare  # noqa: E402
from bots.prices import price_at  # noqa: E402
from bots.summary import bot_rows, history  # noqa: E402
from tests.integration.test_features_cli_db import T0 as C0, _insert  # noqa: E402


def test_price_at(db_conn):
    _insert(db_conn, "BTCUSDT", range(0, 60))
    with db_conn.cursor() as cur:
        cur.execute("SELECT open FROM candles_1m WHERE open_time = %s",
                    (C0 + timedelta(minutes=30),))
        open30 = float(cur.fetchone()[0])
    assert price_at(db_conn, "BTCUSDT", C0 + timedelta(minutes=30)) == open30
    assert price_at(db_conn, "BTCUSDT", C0 + timedelta(minutes=30, seconds=40)) == open30
    assert price_at(db_conn, "BTCUSDT", C0 + timedelta(minutes=59 + 9)) is not None
    assert price_at(db_conn, "BTCUSDT", C0 + timedelta(minutes=59 + 10)) is not None
    assert price_at(db_conn, "BTCUSDT", C0 + timedelta(minutes=59 + 11)) is None
    assert price_at(db_conn, "BTCUSDT", C0 - timedelta(minutes=1)) is None
    assert price_at(db_conn, "ETHUSDT", C0) is None


def test_rows_and_history(db_conn):
    _insert(db_conn, "BTCUSDT", range(0, 120))
    now = C0 + timedelta(days=1)
    a = add_bot(db_conn, symbol="BTCUSDT", invest=500.0, start=C0, fee=0.001, now=now)
    add_update(db_conn, a, C0 + timedelta(minutes=60), 3.0, final=False, now=now)
    add_update(db_conn, a, C0 + timedelta(minutes=100), 4.0, final=False, now=now)
    b = add_bot(db_conn, symbol="SOLUSDT", invest=200.0, start=C0, fee=0.001, now=now)
    add_update(db_conn, b, C0 + timedelta(minutes=5), 1.0, final=False, now=now)
    c = add_bot(db_conn, symbol="BTCUSDT", invest=100.0, start=C0, fee=0.001, now=now)
    rows = {r["id"]: r for r in bot_rows(db_conn)}
    ra = rows[a]
    p0 = price_at(db_conn, "BTCUSDT", C0)
    p1 = price_at(db_conn, "BTCUSDT", C0 + timedelta(minutes=100))
    assert ra["total_profit"] == 4.0 and ra["status"] == "running"
    assert ra["comparison"].hold_profit == compare(500, 0.001, 4.0, p0, p1).hold_profit
    assert rows[b]["comparison"] is None and "no stored price for SOLUSDT" in rows[b]["note"]
    assert rows[c]["note"] == f"no update yet (tb bots update {c} --profit X)"
    h = history(db_conn, a)
    assert [x["total_profit"] for x in h] == [3.0, 4.0]
    assert all(x["comparison"] is not None for x in h)


# ---- commands

import re  # noqa: E402

from typer.testing import CliRunner  # noqa: E402

from data.cli import app  # noqa: E402
from data.storage.db import connect as real_connect  # noqa: E402

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)
runner = CliRunner()


@pytest.fixture
def cli(monkeypatch, migrated_db):
    monkeypatch.setattr("data.cli.connect", lambda: real_connect(migrated_db))
    return runner


def _run(cli, *args):
    out = cli.invoke(app, ["bots", *args])
    assert not ADVICE.search(out.output), out.output
    return out


def test_full_flow(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, 180))
    out = _run(cli, "add", "--symbol", "btcusdt", "--invest", "500", "--start",
               "2024-05-01 00:10", "--utc", "--low", "90", "--high", "110", "--grids", "10")
    assert out.exit_code == 0, out.output
    bid = list_bots(db_conn)[0]["id"]
    assert (f"bot {bid} recorded: BTCUSDT grid, 500.00 USDT, started 2024-05-01 "
            "00:10 UTC") in out.output
    out = _run(cli, "update", str(bid), "--profit", "3.5", "--at", "2024-05-01 01:00", "--utc")
    assert out.exit_code == 0, out.output
    assert "update stored for bot" in out.output and "2024-05-01 01:00 UTC" in out.output
    assert re.search(r"bot \+3\.50 USDT \(\+0\.70%\), holding [+-][\d.]+ USDT.*; bot "
                     r"(ahead of|behind) holding by", out.output), out.output
    out = _run(cli, "list")
    assert out.exit_code == 0 and "running since 2024-05-01 00:10 UTC" in out.output
    assert "All bots (1 of 1 compared" in out.output
    out = _run(cli, "stop", str(bid), "--profit", "4", "--at", "2024-05-01 02:00", "--utc")
    assert out.exit_code == 0 and "stopped" in out.output
    out = _run(cli, "show", str(bid))
    assert out.exit_code == 0 and "(stop)" in out.output and "range 90.00 to 110.00" in out.output
    out = _run(cli, "update", str(bid), "--profit", "5", "--at", "2024-05-01 02:30", "--utc")
    assert out.exit_code == 1 and "is stopped" in out.output
    out = _run(cli, "undo", str(bid))
    assert out.exit_code == 0 and "removed the update of 2024-05-01 02:00 UTC" in out.output
    assert "running again" in out.output
    out = _run(cli, "delete", str(bid))
    assert out.exit_code == 1 and "--yes" in out.output and get_bot(db_conn, bid)
    out = _run(cli, "delete", str(bid), "--yes")
    assert out.exit_code == 0 and get_bot(db_conn, bid) is None


def test_local_time_is_echoed_in_utc(db_conn, cli):
    out = _run(cli, "add", "--symbol", "ETHUSDT", "--invest", "100", "--start",
               "2024-05-01 09:30")
    assert out.exit_code == 0, out.output
    from bots.times import parse_time
    utc = parse_time("2024-05-01 09:30", utc=False)
    assert (f"started {utc:%Y-%m-%d %H:%M} UTC (you typed 2024-05-01 09:30 "
            "local time)") in out.output


@pytest.mark.parametrize("args, msg", [
    (["add", "--symbol", "BTCEUR", "--invest", "5", "--start", "2024-05-01 00:00"], "USDT"),
    (["add", "--symbol", "BTCUSDT", "--invest", "5", "--start", "1 May"], "YYYY-MM-DD"),
    (["add", "--symbol", "BTCUSDT", "--invest", "5", "--start", "1926-05-01 00:00"],
     "out of range"),
    (["update", "7", "--profit", "1"], "no bot 7"),
    (["show", "7"], "no bot 7"),
    (["undo", "7"], "no bot 7"),
    (["delete", "7", "--yes"], "no bot 7"),
])
def test_errors_exit_1_and_store_nothing(db_conn, cli, args, msg):
    out = _run(cli, *args)
    assert out.exit_code == 1 and msg in out.output, out.output
    assert list_bots(db_conn) == []


def test_list_empty(db_conn, cli):
    out = _run(cli, "list")
    assert out.exit_code == 0 and "no bots yet" in out.output


def test_negative_profit_through_the_cli(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, 120))
    assert _run(cli, "add", "--symbol", "BTCUSDT", "--invest", "500", "--start",
                "2024-05-01 00:10", "--utc").exit_code == 0
    bid = list_bots(db_conn)[0]["id"]
    out = _run(cli, "update", str(bid), "--profit", "-5", "--at", "2024-05-01 01:00", "--utc")
    assert out.exit_code == 0, out.output
    assert updates(db_conn, bid)[0]["total_profit"] == -5.0


def test_times_read_back_in_utc_whatever_the_session_zone(db_conn):
    bid = _bot(db_conn)
    with db_conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'Europe/Istanbul'")
    b = get_bot(db_conn, bid)
    assert b["start_time"] == T0 and b["start_time"].utcoffset() == timedelta(0)


def test_database_refuses_bad_rows_too(db_conn):
    import psycopg
    for sql in ("INSERT INTO bots (name, symbol, invest_usdt, start_time, fee) "
                "VALUES ('x', 'BTCUSDT', 0.5, now(), 0)",
                "INSERT INTO bots (name, symbol, invest_usdt, start_time, fee) "
                "VALUES ('x', 'BTCUSDT', 'Infinity', now(), 0)",
                "INSERT INTO bots (name, symbol, invest_usdt, start_time, fee) "
                "VALUES ('x', 'btcusdt', 10, now(), 0)"):
        with pytest.raises(psycopg.errors.CheckViolation):
            db_conn.execute(sql)
        db_conn.rollback()
