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
