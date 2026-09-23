from datetime import datetime, timezone
from unittest.mock import patch

from typer.testing import CliRunner

from data.cli import app

runner = CliRunner()


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["db", "symbols", "backfill", "live", "quality", "status"]:
        assert cmd in result.output


def test_quality_command_rejects_unknown_timeframe():
    # Must fail on the argument, not on missing configuration: the
    # validation runs before any settings are loaded.
    result = runner.invoke(app, ["quality", "--timeframe", "7m"])
    assert result.exit_code != 0
    assert "7m" in result.output or "timeframe" in result.output.lower()


def test_backfill_rejects_malformed_from_date():
    # Must fail on the argument itself, before touching settings/DB.
    result = runner.invoke(app, ["backfill", "--from", "not-a-date"])
    assert result.exit_code != 0
    assert "not-a-date" in result.output


def test_backfill_passes_parsed_from_date_through(monkeypatch, tmp_path):
    captured = {}

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_backfill_symbol(conn, symbol, downloader, api, start=None):
        captured["symbol"] = symbol
        captured["start"] = start
        return 0

    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    from data.config import get_settings
    get_settings.cache_clear()

    with patch("data.cli.connect", return_value=FakeConn()), \
         patch("data.cli.ArchiveDownloader"), \
         patch("data.cli.BinanceRest"), \
         patch("data.cli.backfill_symbol", fake_backfill_symbol):
        result = runner.invoke(
            app, ["backfill", "--symbol", "btcusdt", "--from", "2024-06-01"]
        )

    get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert captured["symbol"] == "BTCUSDT"
    assert captured["start"] == datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_backfill_isolates_per_symbol_failures(monkeypatch):
    """One symbol's exception must not stop the others, but the command
    must still exit non-zero so the failure stays visible."""

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = []

    def fake_backfill_symbol(conn, symbol, downloader, api, start=None):
        calls.append(symbol)
        if symbol == "BTCUSDT":
            raise RuntimeError("boom")
        return 1

    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    from data.config import get_settings
    get_settings.cache_clear()

    with patch("data.cli.connect", return_value=FakeConn()), \
         patch("data.cli.ArchiveDownloader"), \
         patch("data.cli.BinanceRest"), \
         patch("data.cli.backfill_symbol", fake_backfill_symbol):
        result = runner.invoke(app, ["backfill"])

    get_settings.cache_clear()
    assert calls == ["BTCUSDT", "ETHUSDT"], "ETHUSDT must still run after BTCUSDT fails"
    assert result.exit_code != 0
    assert "BTCUSDT" in result.output
