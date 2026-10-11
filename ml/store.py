"""Reading and writing prediction runs (spec section 7.1).

Like features/store.py, the write function does not commit: the caller owns
the transaction.
"""
from datetime import datetime

import pandas as pd
from psycopg.types.json import Jsonb

from features.pipeline import FEATURE_SET
from ml.targets import get_target

_PRED_COLUMNS = ("model", "symbol", "open_time", "fold", "label",
                 "p0", "p1", "p2")


def save_run(conn, *, kind: str, horizon: int, symbols: list[str],
             data_end: datetime, config: dict, metrics: dict,
             predictions: pd.DataFrame, target: str = "move3") -> int:
    """Insert one run and COPY its predictions (columns p0, p1 and, for a
    three-class target, p2). Returns the run_id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ml_runs (kind, horizon, label_set, feature_set, "
            "symbols, data_end, config, metrics, target) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id",
            (kind, horizon, get_target(target).label_set, FEATURE_SET,
             list(symbols), data_end, Jsonb(config), Jsonb(metrics), target),
        )
        run_id = cur.fetchone()[0]
        cols = ", ".join(_PRED_COLUMNS)
        pred = predictions if "p2" in predictions else predictions.assign(p2=None)
        with cur.copy(f"COPY ml_predictions (run_id, {cols}) FROM STDIN") as copy:
            for row in pred[list(_PRED_COLUMNS)].itertuples(index=False):
                copy.write_row((run_id, row.model, row.symbol,
                                pd.Timestamp(row.open_time).to_pydatetime(),
                                int(row.fold), int(row.label),
                                float(row.p0), float(row.p1),
                                None if row.p2 is None or pd.isna(row.p2)
                                else float(row.p2)))
    return run_id


def holdout_count(conn, horizon: int, up_to_run: int | None = None) -> int:
    """Holdout runs for the horizon; with up_to_run, only those up to and
    including that run (what was known when it ran)."""
    sql = "SELECT count(*) FROM ml_runs WHERE kind = 'holdout' AND horizon = %s"
    params = [horizon]
    if up_to_run is not None:
        sql += " AND run_id <= %s"
        params.append(up_to_run)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def latest_run_id(conn, horizon: int, kind: str = "walk_forward",
                  target: str = "move3") -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(run_id) FROM ml_runs WHERE kind = %s "
                    "AND horizon = %s AND target = %s", (kind, horizon, target))
        return cur.fetchone()[0]


def list_runs(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.run_id, r.kind, r.horizon, r.target, r.created_at, "
            "(SELECT count(*) FROM ml_predictions p WHERE p.run_id = r.run_id), "
            "(r.metrics -> 'models' -> 'xgb_v1' ->> 'skill')::float8 "
            "FROM ml_runs r ORDER BY r.run_id")
        keys = ("run_id", "kind", "horizon", "target", "created_at", "n_predictions",
                "xgb_skill")
        return [dict(zip(keys, row)) for row in cur.fetchall()]


def load_run(conn, run_id: int) -> dict | None:
    """Every ml_runs column, plus holdout_count: the holdout runs for the
    run's horizon up to and including this one."""
    with conn.cursor() as cur:
        cur.execute("SELECT run_id, created_at, kind, horizon, target, label_set, "
                    "feature_set, symbols, data_end, config, metrics "
                    "FROM ml_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        names = [d.name for d in cur.description]
    run = dict(zip(names, row))
    run["holdout_count"] = holdout_count(conn, run["horizon"], run["run_id"])
    return run


def load_predictions(conn, run_id: int) -> pd.DataFrame:
    """Every stored prediction of a run, in the columns evaluate() returns."""
    cols = ", ".join(_PRED_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(f"SELECT {cols} FROM ml_predictions WHERE run_id = %s",
                    (run_id,))
        rows = cur.fetchall()
    out = pd.DataFrame(rows, columns=list(_PRED_COLUMNS))
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    return out
