"""Reading a coin's backtest inputs (spec section 3). Read-only.

Day D starts at τ = D 00:00 UTC. Its inputs are the ones known at τ: the 1h
feature row and the stored vol3 prediction whose open_time is τ − 1h (that
bar closes at τ).
"""
import pandas as pd

from backtest.engine import CoinData
from ml import store

MODEL = "xgb_v1"


def default_run(conn) -> int | None:
    """The newest walk-forward vol3 run with horizon 24."""
    return store.latest_run_id(conn, 24, target="vol3")


def check_run(conn, run_id: int) -> str | None:
    """None if the run can drive the grid backtest, else the reason."""
    run = store.load_run(conn, run_id)
    if run is None:
        return f"no run {run_id}"
    if run["target"] != "vol3" or run["horizon"] != 24 or run["kind"] != "walk_forward":
        return (f"run {run_id} is {run['kind']} {run['target']} next {run['horizon']}h; "
                "the grid backtest needs a walk-forward vol3 run with horizon 24")
    return None


def load_coin(conn, symbol: str, run_id: int) -> CoinData:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.open_time + interval '1 hour', p.p2, f.atr_pct "
            "FROM ml_predictions p LEFT JOIN features_1h f "
            "ON f.symbol = p.symbol AND f.open_time = p.open_time "
            "WHERE p.run_id = %s AND p.model = %s AND p.symbol = %s "
            "AND extract(hour FROM p.open_time AT TIME ZONE 'UTC') = 23 "
            "ORDER BY 1", (run_id, MODEL, symbol))
        rows = cur.fetchall()
        days = pd.DataFrame(rows, columns=["day", "p_wild", "atr_pct"])
        days["day"] = pd.to_datetime(days["day"], utc=True)
        days = days.set_index("day")[["atr_pct", "p_wild"]].astype(float)
        if days.empty:
            return CoinData(minutes=pd.DataFrame(columns=["open", "high", "low", "close"]),
                            days=days)
        cur.execute(
            "SELECT open_time, open::float8, high::float8, low::float8, close::float8 "
            "FROM candles_1m WHERE symbol = %s AND open_time >= %s AND open_time < %s "
            "ORDER BY open_time",
            (symbol, days.index.min().to_pydatetime(),
             (days.index.max() + pd.Timedelta(days=1)).to_pydatetime()))
        minutes = pd.DataFrame(cur.fetchall(),
                               columns=["open_time", "open", "high", "low", "close"])
    minutes["open_time"] = pd.to_datetime(minutes["open_time"], utc=True)
    return CoinData(minutes=minutes.set_index("open_time"), days=days)
