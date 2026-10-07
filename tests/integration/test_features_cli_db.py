import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from data.cli import app
from data.config import get_settings
from data.storage.db import connect as real_connect
from data.storage.repository import Candle, upsert_candles, upsert_symbol
from features import build as build_module
from features.pipeline import FEATURE_TIMEFRAMES

pytestmark = pytest.mark.db

runner = CliRunner()

# Well in the past, so every bucket is closed (and every bar is STALE).
T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)
DAY_MINUTES = 24 * 60


def _walk(symbol, minutes):
    out = []
    for m in minutes:
        open_time = T0 + timedelta(minutes=m)
        base = 100 + 8 * math.sin(m / 700) + 3 * math.sin(m / 53) + m / 4000
        o = Decimal(f"{base:.2f}")
        c = Decimal(f"{base + 0.4 * math.sin(m / 3):.2f}")
        out.append(Candle(
            symbol=symbol, open_time=open_time,
            close_time=open_time + timedelta(seconds=59),
            open=o, high=max(o, c) + Decimal("0.25"),
            low=min(o, c) - Decimal("0.25"), close=c,
            volume=Decimal(f"{2 + math.sin(m / 11):.3f}"),
            quote_volume=Decimal("200"), trade_count=5,
            taker_buy_base=Decimal(f"{1 + 0.5 * math.sin(m / 7):.3f}"),
            taker_buy_quote=Decimal("100"), source="archive"))
    return out


def _insert(conn, symbol, minutes):
    upsert_symbol(conn, symbol=symbol)
    upsert_candles(conn, _walk(symbol, minutes))
    conn.commit()


def _count(conn, symbol, tf):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM features_{tf} WHERE symbol = %s",
                    (symbol,))
        return cur.fetchone()[0]


@pytest.fixture
def cli(monkeypatch, migrated_db):
    """Point the CLI at the test database and at two symbols."""
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    get_settings.cache_clear()
    monkeypatch.setattr("data.cli.connect", lambda: real_connect(migrated_db))
    yield runner
    get_settings.cache_clear()


def test_features_build_prints_results_and_exit_codes(db_conn, cli, monkeypatch):
    _insert(db_conn, "BTCUSDT", range(0, 3 * DAY_MINUTES))
    # ETHUSDT: a 3-hour hole in a month with no archive record -> quality FAIL
    _insert(db_conn, "ETHUSDT",
            [m for m in range(0, DAY_MINUTES) if not 600 <= m < 780])

    # one symbol at a time
    result = cli.invoke(app, ["features", "build", "--symbol", "btcusdt"])
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.splitlines() if l.startswith("BTCUSDT")]
    assert [l.split(":")[0] for l in lines] == [
        f"BTCUSDT {tf}" for tf in FEATURE_TIMEFRAMES]
    assert "864 rows" in lines[0] and "full" in lines[0]
    assert "2024-05-01 00:00" in lines[0]
    assert _count(db_conn, "BTCUSDT", "5m") == 864

    # again: nothing new, still success
    result = cli.invoke(app, ["features", "build", "--symbol", "BTCUSDT",
                              "--timeframe", "1h"])
    assert result.exit_code == 0, result.output
    assert "BTCUSDT 1h: up to date" in result.output
    assert "5m" not in result.output

    # everything: BTCUSDT is fine, ETHUSDT is skipped by the gate -> exit 1
    result = cli.invoke(app, ["features", "build"])
    assert result.exit_code == 1
    assert "BTCUSDT 5m: up to date" in result.output
    assert "ETHUSDT 5m: skipped (data quality FAIL)" in result.output
    assert _count(db_conn, "ETHUSDT", "5m") == 0

    # --rebuild recomputes the full history
    result = cli.invoke(app, ["features", "build", "--symbol", "BTCUSDT",
                              "--timeframe", "1d", "--rebuild"])
    assert result.exit_code == 0, result.output
    assert "BTCUSDT 1d: 3 rows" in result.output and "full" in result.output


def test_features_build_rejects_unknown_timeframe(cli):
    result = cli.invoke(app, ["features", "build", "--timeframe", "7m"])
    assert result.exit_code != 0
    assert "7m" in result.output


def test_features_build_isolates_a_symbol_that_raises(db_conn, cli, monkeypatch):
    _insert(db_conn, "BTCUSDT", range(0, DAY_MINUTES))
    _insert(db_conn, "ETHUSDT", range(0, DAY_MINUTES))
    real = build_module.run_quality_checks

    def gate(conn, symbol, *args, **kwargs):
        if symbol == "BTCUSDT":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM this_table_does_not_exist")
        return real(conn, symbol, *args, **kwargs)

    monkeypatch.setattr(build_module, "run_quality_checks", gate)
    result = cli.invoke(app, ["features", "build"])

    assert result.exit_code == 1
    assert "BTCUSDT: feature build failed" in result.output
    assert "Traceback" not in result.output
    # the aborted transaction did not poison the next symbol
    assert "ETHUSDT 1h: 24 rows" in result.output
    assert _count(db_conn, "ETHUSDT", "1h") == 24


def test_analyze_builds_then_prints_state(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, 3 * DAY_MINUTES))

    result = cli.invoke(app, ["analyze", "BTCUSDT"])

    assert result.exit_code == 0, result.output
    assert _count(db_conn, "BTCUSDT", "1h") == 72  # it built first
    assert "BTCUSDT market state" in result.output
    for tf in FEATURE_TIMEFRAMES:
        assert re.search(rf"^{tf}\s+bar closed", result.output, re.M), tf
    # the data is from 2024, so every frame is stale
    assert result.output.count("STALE") == 5
    assert re.search(r"Timeframes: \d+ bullish, \d+ bearish, \d+ sideways",
                     result.output)
    assert re.search(r"\b(buy|sell|long|short|enter|exit)\b",
                     result.output, re.I) is None


def test_analyze_accepts_lowercase(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, DAY_MINUTES))

    result = cli.invoke(app, ["analyze", "btcusdt"])

    assert result.exit_code == 0, result.output
    assert "BTCUSDT market state" in result.output
    assert _count(db_conn, "BTCUSDT", "1h") == 24


def test_analyze_unknown_symbol_exits_1(db_conn, cli):
    result = cli.invoke(app, ["analyze", "NOPEUSDT"])

    assert result.exit_code == 1
    lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 1
    assert "NOPEUSDT" in lines[0] and "no 1m candles" in lines[0]
    assert "Traceback" not in result.output


def test_analyze_no_build_reads_existing_rows(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, DAY_MINUTES))

    # nothing built yet: --no-build must not build
    result = cli.invoke(app, ["analyze", "BTCUSDT", "--no-build"])
    assert result.exit_code == 0, result.output
    assert _count(db_conn, "BTCUSDT", "1h") == 0
    assert result.output.count("not built yet") == 5

    assert cli.invoke(app, ["features", "build", "--symbol", "BTCUSDT"]).exit_code == 0
    result = cli.invoke(app, ["analyze", "BTCUSDT", "--no-build"])
    assert result.exit_code == 0, result.output
    assert "not built yet" not in result.output
    assert re.search(r"^1h\s+bar closed 2024-05-02 00:00 UTC", result.output, re.M)


def test_analyze_reports_a_skipped_build_and_exits_1(db_conn, cli):
    _insert(db_conn, "BTCUSDT",
            [m for m in range(0, DAY_MINUTES) if not 600 <= m < 780])

    result = cli.invoke(app, ["analyze", "BTCUSDT"])

    assert result.exit_code == 1
    assert "BTCUSDT 1h: skipped (data quality FAIL)" in result.output
    assert result.output.count("not built yet") == 5
