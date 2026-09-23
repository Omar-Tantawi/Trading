from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from data.cli import app
from data.storage.db import connect as real_connect

pytestmark = pytest.mark.db

runner = CliRunner()


def test_db_error_in_one_symbol_does_not_poison_the_next(monkeypatch, migrated_db):
    """Regression test for a real DB-level failure mid-backfill.

    `connect()` returns a non-autocommit connection that `tb backfill`
    reuses across all symbols. If BTCUSDT's backfill runs invalid SQL, its
    transaction is left aborted; without a rollback in the CLI's except
    block, ETHUSDT's very next query on that same connection would raise
    InFailedSqlTransaction and be falsely reported as failed too, even
    though nothing is wrong with it.
    """
    conn = real_connect(migrated_db)
    calls = []

    def fake_backfill_symbol(c, symbol, downloader, api, start=None):
        calls.append(symbol)
        if symbol == "BTCUSDT":
            with c.cursor() as cur:
                cur.execute("SELECT * FROM this_table_does_not_exist")
            return 0  # unreachable: the execute above raises
        with c.cursor() as cur:
            cur.execute("SELECT 1")
        return 1

    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    from data.config import get_settings
    get_settings.cache_clear()

    try:
        with patch("data.cli.connect", return_value=conn), \
             patch("data.cli.ArchiveDownloader"), \
             patch("data.cli.BinanceRest"), \
             patch("data.cli.backfill_symbol", fake_backfill_symbol):
            result = runner.invoke(app, ["backfill"])
    finally:
        get_settings.cache_clear()
        conn.close()

    assert calls == ["BTCUSDT", "ETHUSDT"], "ETHUSDT must still be attempted"
    assert result.exit_code != 0
    assert "BTCUSDT: backfill failed" in result.output
    assert "ETHUSDT: 1 candles written" in result.output
    assert "ETHUSDT: backfill failed" not in result.output


def test_db_refresh_aggregates_materializes_existing_history(migrated_db):
    """Existing 1m history that no policy window reaches (the smoke run's
    SOLUSDT data) is materialized on demand by `tb db refresh-aggregates`."""
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal

    from data.storage.repository import Candle, upsert_candles, upsert_symbol

    t0 = datetime(2022, 3, 1, tzinfo=timezone.utc)
    conn = real_connect(migrated_db)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candles_1m WHERE symbol = 'SOLUSDT'")
        conn.commit()
        upsert_symbol(conn, symbol="SOLUSDT")
        upsert_candles(conn, [
            Candle(symbol="SOLUSDT", open_time=t0 + timedelta(minutes=i),
                   close_time=t0 + timedelta(minutes=i, seconds=59, milliseconds=999),
                   open=Decimal("100"), high=Decimal("101"), low=Decimal("99"),
                   close=Decimal("100.5"), volume=Decimal("1"),
                   quote_volume=Decimal("100"), trade_count=1,
                   taker_buy_base=Decimal("0.5"), taker_buy_quote=Decimal("50"),
                   source="archive")
            for i in range(1440)
        ])

        from data.config import get_settings
        get_settings.cache_clear()
        result = runner.invoke(app, ["db", "refresh-aggregates"])
        get_settings.cache_clear()
        assert result.exit_code == 0, result.output

        counts = {}
        with conn.cursor() as cur:
            for tf in ("5m", "15m", "1h", "4h", "1d"):
                cur.execute(f"SELECT count(*) FROM candles_{tf} WHERE symbol = "
                            "'SOLUSDT' AND open_time >= %s AND open_time < %s",
                            (t0, t0 + timedelta(days=1)))
                counts[tf] = cur.fetchone()[0]
        assert counts == {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candles_1m WHERE symbol = 'SOLUSDT'")
        conn.commit()
        conn.close()
