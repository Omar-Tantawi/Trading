"""Dashboard routes that need no database (spec sections 3 and 6)."""
import re
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from dashboard.app import create_app

STATIC = Path(__file__).parents[2] / "dashboard" / "static"
ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)


def _down():
    raise psycopg.OperationalError("connection refused")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused@127.0.0.1:1/none")
    from data.config import get_settings
    get_settings.cache_clear()
    yield TestClient(create_app(_down), base_url="http://127.0.0.1")
    get_settings.cache_clear()


def test_page_and_vendored_library(client):
    page = client.get("/")
    assert page.status_code == 200 and "<html" in page.text
    assert not re.search(r'(src|href)="https?://', page.text)
    js = client.get("/static/vendor/lightweight-charts.standalone.production.js")
    assert js.status_code == 200 and "Lightweight Charts" in js.text
    assert client.get("/static/app.js").status_code == 200


def test_symbols(client):
    out = client.get("/api/symbols").json()
    assert out == {"symbols": ["BTCUSDT", "ETHUSDT"],
                   "timeframes": ["5m", "15m", "1h", "4h", "1d"]}


def test_database_down_is_503(client):
    r = client.get("/api/health")
    assert r.status_code == 503 and "database" in r.json()["detail"]


def test_unknown_symbol_and_timeframe_404(client):
    assert client.get("/api/candles?symbol=DOGEUSDT&timeframe=1h").status_code == 404
    assert client.get("/api/candles?symbol=BTCUSDT&timeframe=2h").status_code == 404


def test_bars_limits_422(client):
    for n in (10, 6000):
        assert client.get(f"/api/candles?symbol=BTCUSDT&timeframe=1h&bars={n}").status_code == 422


def test_model_name_tricks_404(client):
    assert client.get("/api/runs/1/calibration/..%2Fx.svg").status_code == 404
    assert client.get("/api/runs/1/skill/not_a_model.svg").status_code == 404


def test_no_advice_words_in_page():
    for name in ("index.html", "app.js"):
        text = (STATIC / name).read_text()
        assert not ADVICE.search(text), (name, ADVICE.search(text))


def test_foreign_host_header_refused(client):
    assert client.get("/api/symbols", headers={"host": "evil.example"}).status_code == 400
    assert client.get("/api/symbols", headers={"host": "127.0.0.1:8050"}).status_code == 200
