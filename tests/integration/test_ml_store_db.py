"""ML run storage (spec section 7.1) and freshness checks."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from ml import store
from ml.load import StaleFeaturesError, check_fresh

pytestmark = pytest.mark.db

END = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _pred():
    times = pd.date_range("2023-06-01", periods=3, freq="1h", tz="UTC")
    rows = []
    for model in ("base_rate_v1", "xgb_v1"):
        for t in times:
            rows.append({"model": model, "symbol": "BTCUSDT", "open_time": t,
                         "fold": 0, "label": 1, "p_down": .2, "p_flat": .5,
                         "p_up": .3})
    return pd.DataFrame(rows)


def _save(conn, kind="walk_forward", horizon=4):
    metrics = {"models": {"xgb_v1": {"skill": 0.01}}, "horizon": horizon}
    run_id = store.save_run(conn, kind=kind, horizon=horizon, symbols=["BTCUSDT"],
                            data_end=END, config={"k": 0.5}, metrics=metrics,
                            predictions=_pred())
    conn.commit()
    return run_id


def test_tables_exist(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('ml_runs'), to_regclass('ml_predictions')")
        assert all(cur.fetchone())


def test_save_and_load_round_trip(db_conn):
    run_id = _save(db_conn)
    run = store.load_run(db_conn, run_id)
    assert run["metrics"]["models"]["xgb_v1"]["skill"] == 0.01
    assert run["config"] == {"k": 0.5} and run["symbols"] == ["BTCUSDT"]
    assert run["holdout_count"] == 0
    runs = store.list_runs(db_conn)
    assert runs[-1]["n_predictions"] == 6 and runs[-1]["xgb_skill"] == 0.01
    assert store.latest_run_id(db_conn, 4) == run_id
    assert store.load_run(db_conn, run_id + 1000) is None


def test_holdout_count_increments(db_conn):
    _save(db_conn, "holdout")
    _save(db_conn, "holdout")
    _save(db_conn, "holdout", horizon=1)
    assert store.holdout_count(db_conn, 4) == 2
    assert store.latest_run_id(db_conn, 4) is None


def test_delete_cascades(db_conn):
    run_id = _save(db_conn)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM ml_runs WHERE run_id = %s", (run_id,))
        cur.execute("SELECT count(*) FROM ml_predictions WHERE run_id = %s", (run_id,))
        assert cur.fetchone()[0] == 0


def test_check_fresh_raises_when_empty(db_conn):
    with pytest.raises(StaleFeaturesError, match="BTCUSDT 1h: no features"):
        check_fresh(db_conn, ["BTCUSDT"])


def test_holdout_count_is_as_of_the_run(db_conn):
    first = _save(db_conn, "holdout")
    _save(db_conn, "holdout")
    assert store.load_run(db_conn, first)["holdout_count"] == 1
