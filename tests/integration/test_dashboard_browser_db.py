"""Browser smoke test (spec section 6): the real page in Chromium against the
test database. Skipped where Playwright or Chromium is not installed."""
import os
import threading
import time

import pytest

from data.config import get_settings
from data.storage.db import connect as real_connect
from features.build import build_features
from tests.integration.test_dashboard_app_db import _run
from tests.integration.test_features_cli_db import DAY_MINUTES, _insert

pytestmark = [pytest.mark.db, pytest.mark.slow]

sync_api = pytest.importorskip("playwright.sync_api")
PORT = 8765
CHROMIUM = os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


@pytest.fixture
def server(db_conn, migrated_db, monkeypatch):
    import uvicorn

    from dashboard.app import create_app
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    get_settings.cache_clear()
    _insert(db_conn, "BTCUSDT", range(0, 10 * DAY_MINUTES))
    build_features(db_conn, "BTCUSDT", "1h")
    run_id = _run(db_conn)
    srv = uvicorn.Server(uvicorn.Config(create_app(lambda: real_connect(migrated_db)),
                                        host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield run_id
    srv.should_exit = True
    thread.join(5)
    get_settings.cache_clear()


def test_every_tab_renders_without_errors(server):
    run_id = server
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=CHROMIUM if os.path.exists(CHROMIUM) else None)
        except Exception as exc:  # no browser on this machine
            pytest.skip(f"Chromium not available: {exc}")
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(f"http://127.0.0.1:{PORT}/")
        page.wait_for_function("document.getElementById('chart-status').textContent.includes('newest bar closed')")
        assert page.locator("#price-chart canvas").count() > 0
        assert "trend" in page.inner_text("#chart-status")
        # the RSI scale always shows the 30 and 70 guide lines
        lo, hi = page.evaluate("(() => { const r = rsiSeries.priceScale().getVisibleRange(); return [r.from, r.to]; })()")
        assert lo <= 0.5 and hi >= 99.5   # the fixed 0-100 scale
        # charts follow the window size: no sideways scrolling
        page.set_viewport_size({"width": 700, "height": 900})
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth") <= 700

        page.click("button[data-tab=models]")
        page.click(f"#runs tbody tr:has-text('{run_id}')")
        page.wait_for_selector("#run-charts img")
        page.click("#diagnose")
        assert page.is_disabled("#diagnose")
        page.wait_for_selector("#diagnosis table")
        assert not page.is_disabled("#diagnose")
        page.wait_for_selector("#run-charts img")
        assert page.inner_text("#run-report").startswith(f"Run {run_id}")
        page.wait_for_function("[...document.querySelectorAll('#run-charts img')].every(i => i.complete && i.naturalWidth > 0)")

        page.click("button[data-tab=health]")
        page.wait_for_selector("#health-table tbody tr")
        assert "STALE" in page.inner_text("#health-table")
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            assert f"{tf} features" in page.inner_text("#health-table thead")
        page.screenshot(path=os.environ.get("DASHBOARD_SHOT", "/tmp/dashboard.png"), full_page=True)
        browser.close()
    assert errors == []
