"""tb ml train and tb predict against the test database."""
import re
from datetime import timedelta

import pytest
from typer.testing import CliRunner

from data.cli import app
from data.config import get_settings
from data.storage.db import connect as real_connect
from tests.integration.test_features_cli_db import DAY_MINUTES, _insert
from tests.integration.test_ml_cli_db import DAYS, SMALL_FOLDS

pytestmark = pytest.mark.db

runner = CliRunner()
ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)


@pytest.fixture
def cli(monkeypatch, migrated_db, tmp_path):
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")
    get_settings.cache_clear()
    monkeypatch.setattr("data.cli.connect", lambda: real_connect(migrated_db))
    monkeypatch.setattr("data.cli.ML_FOLDS", SMALL_FOLDS)
    monkeypatch.chdir(tmp_path)          # models/ is written here
    yield runner
    get_settings.cache_clear()


def test_predict_without_models_exits_1(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, 2 * DAY_MINUTES))
    result = cli.invoke(app, ["predict", "BTCUSDT"])
    assert result.exit_code == 1 and "run `tb ml train` first" in result.output


def test_train_then_predict_prints_every_horizon(db_conn, cli, tmp_path):
    _insert(db_conn, "BTCUSDT", range(0, DAYS * DAY_MINUTES))
    assert cli.invoke(app, ["features", "build"]).exit_code == 0
    assert cli.invoke(app, ["ml", "evaluate", "--horizon", "4"]).exit_code == 0

    trained = cli.invoke(app, ["ml", "train"])
    assert trained.exit_code == 0, trained.output
    assert (tmp_path / "models" / "xgb_v1_h24" / "meta.json").exists()

    result = cli.invoke(app, ["predict", "btcusdt", "--no-build"])
    assert result.exit_code == 0, result.output
    for h in (1, 4, 24):
        for model in ("logreg_v2", "xgb_v1"):
            assert f"next {h}h" in result.output and f"{model}, move: down" in result.output
    # only the 4h models have a walk-forward run to quote
    assert "no walk-forward run recorded" in result.output
    assert "95% CI" in result.output
    assert "STALE: these probabilities are for a bar that closed" in result.output
    assert not ADVICE.search(result.output)


def test_predict_without_recent_features_exits_1(db_conn, cli):
    _insert(db_conn, "BTCUSDT", range(0, 2 * DAY_MINUTES))
    result = cli.invoke(app, ["predict", "BTCUSDT", "--no-build"])
    assert result.exit_code == 1, result.output
    assert "run `tb features build` first" in result.output


def test_train_vol3_then_predict_shows_it_and_skips_dir2(db_conn, cli, tmp_path):
    _insert(db_conn, "BTCUSDT", range(0, DAYS * DAY_MINUTES))
    assert cli.invoke(app, ["features", "build"]).exit_code == 0
    for target in ("move3", "vol3"):
        r = cli.invoke(app, ["ml", "evaluate", "--target", target, "--horizon", "4"])
        assert r.exit_code == 0, r.output
        r = cli.invoke(app, ["ml", "train", "--target", target])
        assert r.exit_code == 0, r.output
    assert (tmp_path / "models" / "vol3_xgb_v1_h4" / "meta.json").exists()
    assert (tmp_path / "models" / "xgb_v1_h4" / "meta.json").exists()
    out = cli.invoke(app, ["predict", "BTCUSDT", "--no-build"])
    assert out.exit_code == 0, out.output
    assert "xgb_v1, volatility: quiet" in out.output and "wild" in out.output
    assert "direction: no saved models; run `tb ml train --target dir2`" in out.output
    assert "xgb_v1, move: down" in out.output
    assert not ADVICE.search(out.output)
