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
    from ml.predict import prediction_state
    state = prediction_state(db_conn, "BTCUSDT", root=tmp_path / "models", now=NOW)
    same = [p for p in state.predictions
            if (p.target, p.model, p.horizon) == ("vol3", "xgb_v1", 24)]
    assert x.p_wild == pytest.approx(same[0].probs[2])
    assert same[0].classes[2] == "wild"
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


# ---- commands

import re  # noqa: E402

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)


def test_signals_without_data_is_wait_and_stored(db_conn, cli):
    out = cli.invoke(app, ["signals"])
    assert out.exit_code == 0, out.output
    assert "BTCUSDT  WAIT" in out.output and "no market data" in out.output
    assert not ADVICE.search(out.output)
    assert [r["symbol"] for r in latest_signals(db_conn)] == ["BTCUSDT"]


def test_signals_with_trained_model(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, DAYS * DAY_MINUTES))
    assert cli.invoke(app, ["features", "build"]).exit_code == 0
    assert cli.invoke(app, ["ml", "train", "--target", "vol3"]).exit_code == 0
    assert cli.invoke(app, ["risk", "set", "--capital", "2000"]).exit_code == 0
    out = cli.invoke(app, ["signals"])
    assert out.exit_code == 0, out.output
    assert "BTCUSDT  WAIT" in out.output                  # the test data is old
    assert "data STALE" in out.output and "P(wild) next 24h:" in out.output
    assert "capital limit: 500 USDT (25% of 2,000 USDT)" in out.output
    assert "note: data quality not checked" not in out.output
    assert not ADVICE.search(out.output)
    nb = cli.invoke(app, ["signals", "--no-build"])
    assert nb.exit_code == 0 and "data quality not checked (--no-build)" in nb.output
    assert latest_signals(db_conn)[0]["p_wild"] is not None


def test_signals_reports_quality_fail(db_conn, cli):
    _insert(db_conn, "BTCUSDT", [m for m in range(0, DAY_MINUTES) if not 600 <= m < 780])
    out = cli.invoke(app, ["signals"])
    assert out.exit_code == 1
    assert "data quality FAIL" in out.output and "BTCUSDT  WAIT" in out.output


def test_risk_show_and_set(db_conn, cli):
    out = cli.invoke(app, ["risk", "show"])
    assert out.exit_code == 0 and "profile:         defensive" in out.output
    assert "capital:         not set" in out.output
    out = cli.invoke(app, ["risk", "set", "--profile", "return", "--wild-threshold",
                           "0.3", "--capital", "1500", "--fee", "0.001"])
    assert out.exit_code == 0, out.output
    s = load_settings(db_conn)
    assert (s.profile, s.wild_threshold, s.capital_usdt, s.fee) == ("return", 0.3, 1500, 0.001)
    assert s.max_coin_share == 0.25
    assert not ADVICE.search(out.output)
    out = cli.invoke(app, ["risk", "set", "--no-capital"])
    assert out.exit_code == 0 and load_settings(db_conn).capital_usdt is None


@pytest.mark.parametrize("args", [
    ["--wild-threshold", "0.99"], ["--profile", "bold"], ["--max-coin-share", "0"],
    ["--fee", "0.5"], ["--capital", "-5"], ["--capital", "10", "--no-capital"],
])
def test_risk_set_rejects_bad_values(db_conn, cli, args):
    out = cli.invoke(app, ["risk", "set", *args])
    assert out.exit_code == 1, out.output
    assert load_settings(db_conn) == RiskSettings()


def test_one_coin_error_does_not_stop_the_others(db_conn, cli, monkeypatch):
    import signals.collect
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
    get_settings.cache_clear()
    real = signals.collect.collect

    def flaky(conn, symbol, *a, **k):
        if symbol == "ETHUSDT":
            with conn.cursor() as cur:
                cur.execute("SELECT 1/0")          # leaves the transaction aborted
        return real(conn, symbol, *a, **k)
    monkeypatch.setattr(signals.collect, "collect", flaky)
    out = cli.invoke(app, ["signals"])
    assert out.exit_code == 1, out.output
    assert "ETHUSDT  WAIT" in out.output and "could not evaluate:" in out.output
    assert "SOLUSDT  WAIT" in out.output and "no market data" in out.output
    assert [r["symbol"] for r in latest_signals(db_conn)] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_quality_gate_error_is_not_reported_as_passed(db_conn, cli, monkeypatch):
    _insert(db_conn, "BTCUSDT", range(0, 5 * DAY_MINUTES))
    build_symbol(db_conn, "BTCUSDT", FEATURE_TIMEFRAMES)

    def boom(*a, **k):
        raise RuntimeError("quality check broke")
    monkeypatch.setattr("data.cli.build_symbol", boom)
    out = cli.invoke(app, ["signals"])
    assert out.exit_code == 1, out.output
    assert "feature build failed (error: quality check broke)" in out.output
    assert latest_signals(db_conn)[0]["status"] == "WAIT"


def test_database_refuses_out_of_range_settings(db_conn):
    import psycopg
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute("INSERT INTO risk_settings (profile, wild_threshold, "
                        "max_coin_share, fee) VALUES ('defensive', 0.25, 0.25, 0.5)")
    db_conn.rollback()
