"""Reading and writing prediction runs (spec section 7.1).

Like features/store.py, the write function does not commit: the caller owns
the transaction.
"""
from datetime import datetime

import pandas as pd
from psycopg.types.json import Jsonb

from features.pipeline import FEATURE_SET
from ml.labels import LABEL_SET

_PRED_COLUMNS = ("model", "symbol", "open_time", "fold", "label",
                 "p_down", "p_flat", "p_up")


def save_run(conn, *, kind: str, horizon: int, symbols: list[str],
             data_end: datetime, config: dict, metrics: dict,
             predictions: pd.DataFrame) -> int:
    """Insert one run and COPY its predictions. Returns the run_id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ml_runs (kind, horizon, label_set, feature_set, "
            "symbols, data_end, config, metrics) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id",
            (kind, horizon, LABEL_SET, FEATURE_SET, list(symbols), data_end,
             Jsonb(config), Jsonb(metrics)),
        )
        run_id = cur.fetchone()[0]
        cols = ", ".join(_PRED_COLUMNS)
        with cur.copy(f"COPY ml_predictions (run_id, {cols}) FROM STDIN") as copy:
            for row in predictions[list(_PRED_COLUMNS)].itertuples(index=False):
                copy.write_row((run_id, row.model, row.symbol,
                                pd.Timestamp(row.open_time).to_pydatetime(),
                                int(row.fold), int(row.label),
                                float(row.p_down), float(row.p_flat),
                                float(row.p_up)))
    return run_id


def holdout_count(conn, horizon: int) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ml_runs "
                    "WHERE kind = 'holdout' AND horizon = %s", (horizon,))
        return cur.fetchone()[0]


def latest_run_id(conn, horizon: int, kind: str = "walk_forward") -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(run_id) FROM ml_runs WHERE kind = %s "
                    "AND horizon = %s", (kind, horizon))
        return cur.fetchone()[0]


def list_runs(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.run_id, r.kind, r.horizon, r.created_at, "
            "(SELECT count(*) FROM ml_predictions p WHERE p.run_id = r.run_id), "
            "(r.metrics -> 'models' -> 'xgb_v1' ->> 'skill')::float8 "
            "FROM ml_runs r ORDER BY r.run_id")
        keys = ("run_id", "kind", "horizon", "created_at", "n_predictions",
                "xgb_skill")
        return [dict(zip(keys, row)) for row in cur.fetchall()]


def load_run(conn, run_id: int) -> dict | None:
    """Every ml_runs column, plus holdout_count for the run's horizon."""
    with conn.cursor() as cur:
        cur.execute("SELECT run_id, created_at, kind, horizon, label_set, "
                    "feature_set, symbols, data_end, config, metrics "
                    "FROM ml_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        names = [d.name for d in cur.description]
    run = dict(zip(names, row))
    run["holdout_count"] = holdout_count(conn, run["horizon"])
    return run
