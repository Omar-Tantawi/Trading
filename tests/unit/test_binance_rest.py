from datetime import timezone
from decimal import Decimal

import httpx
import pytest
import respx

from data.collectors.binance_rest import (
    BinanceRest,
    RateLimitedError,
    backoff_delays,
)

BASE = "https://api.binance.com"


def test_backoff_grows_and_is_capped():
    # Without jitter the growth is deterministic and must be monotonic.
    delays = backoff_delays(6, base=0.5, cap=8.0, jitter=False)
    assert delays[0] == pytest.approx(0.5)
    assert delays == sorted(delays)
    assert max(delays) <= 8.0


def test_backoff_jitter_never_exceeds_the_cap():
    # Jitter must be applied inside the cap, not on top of it: a delay
    # above the cap is what turns a retry storm into a ban.
    for _ in range(200):
        assert max(backoff_delays(8, base=0.5, cap=8.0)) <= 8.0


@respx.mock
def test_retries_then_succeeds_on_500():
    route = respx.get(f"{BASE}/api/v3/ping").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    assert api.get("/api/v3/ping", {}) == {"ok": True}
    assert route.call_count == 3


@respx.mock
def test_no_sleep_after_final_exhausted_attempt():
    # Every attempt returns a retryable 5xx, so all attempts are used up.
    # There must be no sleep after the last one: nothing follows it, so
    # sleeping there is pure wasted latency before raising.
    route = respx.get(f"{BASE}/api/v3/ping").mock(
        return_value=httpx.Response(500)
    )
    sleeps: list[float] = []
    api = BinanceRest(base_url=BASE, sleep=sleeps.append, attempts=4)
    with pytest.raises(httpx.HTTPStatusError):
        api.get("/api/v3/ping", {})
    assert route.call_count == 4
    assert len(sleeps) == route.call_count - 1


@respx.mock
def test_429_raises_rate_limited_with_retry_after():
    respx.get(f"{BASE}/api/v3/ping").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "120"})
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    with pytest.raises(RateLimitedError) as exc:
        api.get("/api/v3/ping", {})
    assert exc.value.retry_after == 120.0


@respx.mock
def test_418_ban_raises_immediately_without_retrying():
    route = respx.get(f"{BASE}/api/v3/ping").mock(
        return_value=httpx.Response(418, headers={"Retry-After": "300"})
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    with pytest.raises(RateLimitedError):
        api.get("/api/v3/ping", {})
    assert route.call_count == 1


@respx.mock
def test_first_candle_time_is_utc():
    respx.get(f"{BASE}/api/v3/klines").mock(
        return_value=httpx.Response(
            200,
            json=[[1502942400000, "4261.48", "4745.42", "4200.74", "4724.89",
                   "1000.5", 1502942459999, "4000000.0", 100, "500.2",
                   "2000000.0", "0"]],
        )
    )
    api = BinanceRest(base_url=BASE, sleep=lambda _: None)
    ts = api.first_candle_time("BTCUSDT")
    assert ts.tzinfo is timezone.utc
    assert ts.year == 2017


@respx.mock
def test_symbol_metadata_extracts_filters():
    from data.collectors.binance_rest import parse_symbol_info

    payload = {"symbols": [{
        "symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT",
        "status": "TRADING",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.00001000"},
            {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
        ],
    }]}
    info = parse_symbol_info(payload)["BTCUSDT"]
    assert info["tick_size"] == Decimal("0.01")
    assert info["step_size"] == Decimal("0.00001")
    assert info["min_notional"] == Decimal("5")
    assert info["status"] == "TRADING"
