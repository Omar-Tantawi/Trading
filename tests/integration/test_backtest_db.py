"""Backtest inputs from the database and tb backtest grid."""
import re
from datetime import timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from backtest.load import check_run, load_coin
from data.cli import app
from tests.integration.test_features_cli_db import DAY_MINUTES, T0, _insert
from tests.integration.test_ml_cli_db import DAYS, SMALL_FOLDS, cli  # noqa: F401

pytestmark = pytest.mark.db
ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)


@pytest.fixture
def runs(db_conn, cli):  # noqa: F811
    _insert(db_conn, "BTCUSDT", range(0, DAYS * DAY_MINUTES))
    assert cli.invoke(app, ["features", "build"]).exit_code == 0
    for target, h in (("vol3", "24"), ("dir2", "24"), ("vol3", "4")):
        r = cli.invoke(app, ["ml", "evaluate", "--target", target, "--horizon", h])
        assert r.exit_code == 0, r.output
    with db_conn.cursor() as cur:
        cur.execute("SELECT target, horizon, run_id FROM ml_runs ORDER BY run_id")
        return {(t, h): i for t, h, i in cur.fetchall()}


def test_load_coin_decision_rows(db_conn, runs):
    run_id = runs[("vol3", 24)]
    data = load_coin(db_conn, "BTCUSDT", run_id)
    assert len(data.days) > 0
    assert (data.days.index == data.days.index.floor("1D")).all()
    assert data.days["p_wild"].between(0, 1).all() and data.days["atr_pct"].notna().all()
    first = data.days.index.min()
    assert data.minutes.index.min() == first
    with db_conn.cursor() as cur:
        cur.execute("SELECT p2 FROM ml_predictions WHERE run_id = %s AND model = 'xgb_v1' "
                    "AND symbol = 'BTCUSDT' AND open_time = %s",
                    (run_id, (first - pd.Timedelta(hours=1)).to_pydatetime()))
        assert cur.fetchone()[0] == pytest.approx(data.days["p_wild"].iloc[0])


def test_check_run_refuses_wrong_runs(db_conn, runs):
    assert check_run(db_conn, runs[("vol3", 24)]) is None
    assert "vol3 run with horizon 24" in check_run(db_conn, runs[("dir2", 24)])
    assert "vol3 run with horizon 24" in check_run(db_conn, runs[("vol3", 4)])
    assert check_run(db_conn, 999999) == "no run 999999"


def test_cli_backtest_grid(db_conn, runs, cli):  # noqa: F811
    bad = cli.invoke(app, ["backtest", "grid", "--run", str(runs[("dir2", 24)])])
    assert bad.exit_code == 1 and "horizon 24" in bad.output
    r = cli.invoke(app, ["backtest", "grid"])
    assert r.exit_code == 0, r.output
    for name in ("cash", "hold", "grid_always", "grid_vol3_30", "grid_random_30"):
        assert name in r.output
    assert "grid_vol3_30 vs grid_always" in r.output and "Not advice" in r.output
    assert not ADVICE.search(r.output)
