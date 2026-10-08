"""The dashboard's web routes (spec sections 3 and 4). Read-only.

create_app takes the connection factory so tests can point it at the test
database. Every parameter that reaches SQL is checked first: symbols against
the configured list, timeframes against FEATURE_TIMEFRAMES, model names
against the known models.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from dashboard import queries
from dashboard.svg import calibration_svg, skill_svg
from data.config import get_settings
from features.pipeline import FEATURE_TIMEFRAMES
from ml import store
from ml.diagnose import split_skill
from ml.report import MODEL_ORDER, render_report

STATIC = Path(__file__).parent / "static"


def create_app(connect: Callable[[], psycopg.Connection]) -> FastAPI:
    app = FastAPI(title="AI Trading Buddy dashboard", docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @contextmanager
    def db():
        try:
            conn = connect()
        except psycopg.OperationalError as exc:
            raise HTTPException(503, f"database not reachable: {exc}") from exc
        try:
            yield conn
        finally:
            conn.close()

    def check_symbol(symbol: str) -> str:
        symbol = symbol.upper()
        if symbol not in get_settings().symbols:
            raise HTTPException(404, f"unknown symbol {symbol}")
        return symbol

    def check_model(model: str) -> str:
        if model not in MODEL_ORDER:
            raise HTTPException(404, f"unknown model {model}")
        return model

    def run_or_404(conn, run_id: int) -> dict:
        run = store.load_run(conn, run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id}")
        return run

    def model_metrics(run: dict, model: str) -> dict:
        m = run["metrics"]["models"].get(model)
        if m is None:
            raise HTTPException(404, f"run {run['run_id']} has no model {model}")
        return m

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/symbols")
    def symbols():
        return {"symbols": list(get_settings().symbols),
                "timeframes": list(FEATURE_TIMEFRAMES)}

    @app.get("/api/candles")
    def candles(symbol: str, timeframe: str,
                bars: int = Query(500, ge=50, le=5000)):
        symbol = check_symbol(symbol)
        if timeframe not in FEATURE_TIMEFRAMES:
            raise HTTPException(404, f"unknown timeframe {timeframe}")
        with db() as conn:
            return queries.candles(conn, symbol, timeframe, bars)

    @app.get("/api/health")
    def health():
        with db() as conn:
            return queries.health(conn, list(get_settings().symbols),
                                  datetime.now(timezone.utc))

    @app.get("/api/runs")
    def runs():
        with db() as conn:
            return store.list_runs(conn)

    @app.get("/api/runs/{run_id}")
    def run(run_id: int):
        with db() as conn:
            r = run_or_404(conn, run_id)
        return {**r, "report": render_report(r)}

    @app.get("/api/runs/{run_id}/calibration/{model}.svg")
    def calibration(run_id: int, model: str):
        check_model(model)
        with db() as conn:
            m = model_metrics(run_or_404(conn, run_id), model)
        svg = calibration_svg(m.get("reliability") or [], m.get("ece"),
                              f"{model}: calibration")
        return Response(svg, media_type="image/svg+xml")

    @app.get("/api/runs/{run_id}/skill/{model}.svg")
    def skill(run_id: int, model: str):
        check_model(model)
        with db() as conn:
            m = model_metrics(run_or_404(conn, run_id), model)
        svg = skill_svg(m.get("per_fold") or [], m.get("skill", 0.0),
                        m.get("skill_lo"), m.get("skill_hi"),
                        f"{model}: skill per test period")
        return Response(svg, media_type="image/svg+xml")

    @app.get("/api/runs/{run_id}/diagnose")
    def diagnose(run_id: int):
        with db() as conn:
            run_or_404(conn, run_id)
            pred = store.load_predictions(conn, run_id)
        return split_skill(pred)

    return app
