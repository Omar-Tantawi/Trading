"""Dashboard routes against the test database."""
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.app import create_app
from data.config import get_settings
from data.storage.db import connect as real_connect
from features.build import build_features
from ml import store
from tests.integration.test_features_cli_db import DAY_MINUTES, _insert

pytestmark = pytest.mark.db


@pytest.fixture
def client(db_conn, migrated_db, monkeypatch):
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    get_settings.cache_clear()
    _insert(db_conn, "BTCUSDT", range(0, 10 * DAY_MINUTES))
    build_features(db_conn, "BTCUSDT", "1h")
    yield TestClient(create_app(lambda: real_connect(migrated_db)))
    get_settings.cache_clear()


def _run(conn):
    times = pd.date_range("2024-05-01", periods=4, freq="1h", tz="UTC")
    pred = pd.concat([pd.DataFrame({
        "model": m, "symbol": "BTCUSDT", "open_time": times, "fold": 0,
        "label": [0, 1, 2, 1], "p_down": .2, "p_flat": .5, "p_up": .3})
        for m in ("base_rate_v1", "xgb_v1")])
    model = {"log_loss": 1.0, "brier": .6, "accuracy": .5, "ece": .01, "skill": .01,
             "skill_lo": .0, "skill_hi": .02,
             "reliability": [{"lo": .2, "hi": .3, "n": 4, "mean_p": .25, "freq": .25}],
             "per_fold": [{"fold": 0, "test_start": "2024-05-01T01:00:00+00:00",
                           "n": 4, "skill": .01}],
             "per_symbol": {"BTCUSDT": .01}}
    metrics = {"horizon": 4, "n_test": 4,
               "class_shares": {"down": .25, "flat": .5, "up": .25},
               "models": {"base_rate_v1": {**model, "skill": 0.0}, "xgb_v1": model}}
    run_id = store.save_run(conn, kind="walk_forward", horizon=4, symbols=["BTCUSDT"],
                            data_end=datetime(2024, 5, 2, tzinfo=timezone.utc),
                            config={}, metrics=metrics, predictions=pred)
    conn.commit()
    return run_id


def test_candles_and_health(client):
    out = client.get("/api/candles?symbol=btcusdt&timeframe=1h&bars=50").json()
    assert len(out["bars"]) == 50 and out["latest"]["trend_regime"]
    empty = client.get("/api/candles?symbol=ETHUSDT&timeframe=4h").json()
    assert empty == {"bars": [], "latest": None}
    health = client.get("/api/health").json()
    assert [h["symbol"] for h in health] == ["BTCUSDT", "ETHUSDT"]
    assert health[0]["stale"] is True and health[1]["last_1m"] is None


def test_runs_report_svgs_and_diagnose(client, db_conn):
    run_id = _run(db_conn)
    runs = client.get("/api/runs").json()
    assert runs[-1]["run_id"] == run_id
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["report"].startswith(f"Run {run_id}")
    for kind in ("calibration", "skill"):
        r = client.get(f"/api/runs/{run_id}/{kind}/xgb_v1.svg")
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")
        assert "<svg" in r.text
    assert client.get(f"/api/runs/{run_id}/skill/logreg_v2.svg").status_code == 404
    assert client.get(f"/api/runs/{run_id + 99}").status_code == 404
    d = client.get(f"/api/runs/{run_id}/diagnose").json()
    assert set(d) == {"xgb_v1"} and set(d["xgb_v1"]) == {"size", "direction"}
