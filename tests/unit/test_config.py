from pathlib import Path

from data.config import Settings


def test_settings_parses_symbols_and_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT, ethusdt ,SOLUSDT")
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))

    s = Settings()

    assert s.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert isinstance(s.data_cache_dir, Path)
    assert s.binance_base_url == "https://api.binance.com"


def test_settings_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TB_IGNORE_DOTENV", "1")
    try:
        Settings(_env_file=None)
    except Exception as exc:
        assert "database_url" in str(exc).lower()
    else:
        raise AssertionError("expected a validation error")
