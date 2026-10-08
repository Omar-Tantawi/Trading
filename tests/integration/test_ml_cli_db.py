"""tb ml evaluate / runs / report against the test database."""
import re
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from data.cli import app
from data.config import get_settings
from data.storage.db import connect as real_connect
from ml.folds import FoldConfig
from tests.integration.test_features_cli_db import DAY_MINUTES, T0, _insert

pytestmark = pytest.mark.db

runner = CliRunner()
DAYS = 50
SMALL_FOLDS = FoldConfig(first_test=T0 + timedelta(days=30), fold_months=1,
                         holdout_start=T0 + timedelta(days=45), cal_fraction=0.2)


@pytest.fixture
def cli(monkeypatch, migrated_db):
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    get_settings.cache_clear()
    monkeypatch.setattr("data.cli.connect", lambda: real_connect(migrated_db))
    monkeypatch.setattr("data.cli.ML_FOLDS", SMALL_FOLDS)
    yield runner
    get_settings.cache_clear()


def test_evaluate_stale_features_exits_1(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, 2 * DAY_MINUTES))
    result = cli.invoke(app, ["ml", "evaluate", "--horizon", "4"])
    assert result.exit_code == 1
    assert "run `tb features build` first" in result.output


def test_evaluate_quality_fail_exits_1(db_conn, cli):
    _insert(db_conn, "BTCUSDT",
            [m for m in range(0, DAY_MINUTES) if not 600 <= m < 780])
    result = cli.invoke(app, ["ml", "evaluate"])
    assert result.exit_code == 1
    assert "refusing to train" in result.output


def test_evaluate_rejects_unknown_horizon(cli):
    result = cli.invoke(app, ["ml", "evaluate", "--horizon", "2"])
    assert result.exit_code != 0


def test_evaluate_runs_report_and_holdout(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, DAYS * DAY_MINUTES))
    assert cli.invoke(app, ["features", "build"]).exit_code == 0

    result = cli.invoke(app, ["ml", "evaluate", "--horizon", "4"])
    assert result.exit_code == 0, result.output
    assert "next 4h" in result.output and "xgb_v1: skill" in result.output
    assert "data loaded" in result.output and "period 1/" in result.output
    assert re.search(r"\[\d\d:\d\d\] done", result.output)

    listed = cli.invoke(app, ["ml", "runs"])
    assert listed.exit_code == 0 and "walk_forward" in listed.output
    with db_conn.cursor() as cur:
        cur.execute("SELECT max(run_id), max(data_end) FROM ml_runs")
        run_id, data_end = cur.fetchone()
    # a walk-forward run never reaches into the holdout
    assert data_end <= SMALL_FOLDS.holdout_start
    again = cli.invoke(app, ["ml", "report", str(run_id)])
    assert again.exit_code == 0 and f"Run {run_id}" in again.output
    assert cli.invoke(app, ["ml", "report", str(run_id + 99)]).exit_code == 1
    diag = cli.invoke(app, ["ml", "diagnose", str(run_id)])
    assert diag.exit_code == 0, diag.output
    assert "xgb_v1" in diag.output and "direction (up vs down" in diag.output
    assert cli.invoke(app, ["ml", "diagnose", str(run_id + 99)]).exit_code == 1

    for n in (1, 2):
        held = cli.invoke(app, ["ml", "evaluate", "--horizon", "4", "--holdout"])
        assert held.exit_code == 0, held.output
        assert f"Holdout evaluations for this horizon so far: {n}" in held.output
    assert "no longer an unbiased estimate" in held.output
