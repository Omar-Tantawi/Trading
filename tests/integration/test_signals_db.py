"""Risk settings and stored signals against the test database."""
import pytest

from signals.settings import RiskSettings, load_settings, save_settings

pytestmark = pytest.mark.db


def test_no_row_gives_defaults(db_conn):
    assert load_settings(db_conn) == RiskSettings()


def test_settings_round_trip(db_conn):
    s = RiskSettings(profile="return", wild_threshold=0.3, capital_usdt=2000.0,
                     max_coin_share=0.5, fee=0.001)
    save_settings(db_conn, s)
    save_settings(db_conn, s)          # still one row
    assert load_settings(db_conn) == s
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM risk_settings")
        assert cur.fetchone()[0] == 1


def test_save_refuses_bad_settings(db_conn):
    with pytest.raises(ValueError):
        save_settings(db_conn, RiskSettings(wild_threshold=2.0))
    assert load_settings(db_conn) == RiskSettings()


# ---- collect and store

from datetime import timedelta  # noqa: E402

from typer.testing import CliRunner  # noqa: E402

from data.cli import app  # noqa: E402
from data.config import get_settings  # noqa: E402
from data.storage.db import connect as real_connect  # noqa: E402
from features.build import build_symbol  # noqa: E402
from features.pipeline import FEATURE_TIMEFRAMES  # noqa: E402
from signals.collect import collect  # noqa: E402
from signals.engine import evaluate  # noqa: E402
from signals.store import latest_signals, save_signal  # noqa: E402
from tests.integration.test_features_cli_db import DAY_MINUTES, T0, _insert  # noqa: E402
from tests.integration.test_ml_cli_db import DAYS  # noqa: E402

runner = CliRunner()


@pytest.fixture
def cli(monkeypatch, migrated_db, tmp_path):
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    get_settings.cache_clear()
    monkeypatch.setattr("data.cli.connect", lambda: real_connect(migrated_db))
    monkeypatch.chdir(tmp_path)          # models/ is written here
    yield runner
    get_settings.cache_clear()


NOW = T0 + timedelta(days=DAYS)


def test_collect_without_candles(db_conn, tmp_path):
    x = collect(db_conn, "BTCUSDT", quality_fail=None, root=tmp_path / "models", now=NOW)
    assert not x.has_data and x.bar_open_time is None and x.p_wild is None


def test_collect_without_model(db_conn, tmp_path):
    _insert(db_conn, "BTCUSDT", range(0, 5 * DAY_MINUTES))
    build_symbol(db_conn, "BTCUSDT", FEATURE_TIMEFRAMES)
    x = collect(db_conn, "BTCUSDT", quality_fail=False, root=tmp_path / "models",
                now=T0 + timedelta(days=5, minutes=10))
    assert x.has_data and x.bar_open_time == T0 + timedelta(days=5, hours=-1)
    assert x.model_problem == "no saved model" and x.p_wild is None
    assert x.close > 0 and x.atr_pct > 0
    assert x.age == timedelta(minutes=10)


def test_collect_with_trained_model_and_other_symbol(db_conn, cli, tmp_path):
    _insert(db_conn, "BTCUSDT", range(0, DAYS * DAY_MINUTES))
    _insert(db_conn, "ETHUSDT", range(0, 5 * DAY_MINUTES))
    assert cli.invoke(app, ["features", "build"]).exit_code == 0
    build_symbol(db_conn, "ETHUSDT", FEATURE_TIMEFRAMES)
    r = cli.invoke(app, ["ml", "train", "--target", "vol3"])
    assert r.exit_code == 0, r.output
    x = collect(db_conn, "BTCUSDT", quality_fail=False, root=tmp_path / "models", now=NOW)
    assert x.model_problem is None and 0 <= x.p_wild <= 1
    eth = collect(db_conn, "ETHUSDT", quality_fail=False, root=tmp_path / "models", now=NOW)
    assert eth.p_wild is None and "not among the training symbols" in eth.model_problem


def test_store_round_trip_and_newest_per_coin(db_conn, tmp_path):
    from signals.settings import RiskSettings
    assert latest_signals(db_conn) == []
    _insert(db_conn, "BTCUSDT", range(0, 5 * DAY_MINUTES))
    build_symbol(db_conn, "BTCUSDT", FEATURE_TIMEFRAMES)
    x = collect(db_conn, "BTCUSDT", quality_fail=False, root=tmp_path, now=NOW)
    s = RiskSettings(capital_usdt=1000.0)
    first = save_signal(db_conn, evaluate(x, s))
    x.model_problem, x.p_wild = None, 0.1
    second = save_signal(db_conn, evaluate(x, s))
    save_signal(db_conn, evaluate(collect(db_conn, "ETHUSDT", None, tmp_path, NOW), s))
    rows = latest_signals(db_conn)
    assert [r["symbol"] for r in rows] == ["BTCUSDT", "ETHUSDT"]
    btc = rows[0]
    assert btc["id"] == second > first
    assert btc["status"] == "WAIT" and btc["p_wild"] == pytest.approx(0.1)
    assert any("STALE" in r for r in btc["reasons"])
    assert btc["settings"]["capital_usdt"] == 1000.0
    assert btc["capital_limit"] == pytest.approx(250.0)
    assert btc["range_low"] < btc["range_high"]
    assert rows[1]["reasons"] == ["no market data"]
