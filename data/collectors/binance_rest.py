import logging
import random
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

import httpx

from data.config import get_settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {500, 502, 503, 504}


class RateLimitedError(Exception):
    def __init__(self, status: int, retry_after: float):
        super().__init__(f"rate limited (HTTP {status}); retry after {retry_after}s")
        self.status = status
        self.retry_after = retry_after


def backoff_delays(attempts: int, base: float = 0.5, cap: float = 30.0,
                   jitter: bool = True) -> list[float]:
    """Exponential backoff with jitter, capped.

    The jitter is applied *inside* the cap: a delay longer than the cap
    would be a surprise, and the cap is what keeps a retry storm bounded.
    """
    out = []
    for i in range(attempts):
        raw = base * (2 ** i)
        if jitter and i:
            raw *= 0.8 + 0.4 * random.random()
        out.append(min(cap, raw))
    return out


class BinanceRest:
    def __init__(self, base_url: str | None = None,
                 client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 attempts: int = 5):
        self.base_url = (base_url or get_settings().binance_base_url).rstrip("/")
        self.client = client or httpx.Client(timeout=30.0)
        self.sleep = sleep
        self.attempts = attempts

    def get(self, path: str, params: dict) -> Any:
        delays = backoff_delays(self.attempts)
        last_exc: Exception | None = None
        for attempt, delay in enumerate(delays):
            more_attempts = attempt < len(delays) - 1
            try:
                resp = self.client.get(self.base_url + path, params=params)
            except httpx.TransportError as exc:
                # Timeouts, refused connections, dropped sockets: transient
                # by nature, so they get the same backoff as a 5xx.
                last_exc = exc
                if more_attempts:
                    log.warning("binance %s -> %s, retrying in %.1fs",
                                path, type(exc).__name__, delay)
                    self.sleep(delay)
                continue
            if resp.status_code in (429, 418):
                # 418 means we are already banned: never retry into a ban.
                retry_after = float(resp.headers.get("Retry-After", 60))
                raise RateLimitedError(resp.status_code, retry_after)
            if resp.status_code in RETRYABLE_STATUS:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                if more_attempts:
                    # Only sleep if another attempt follows; sleeping before
                    # giving up on the last attempt is pure wasted latency.
                    log.warning("binance %s -> %s, retrying in %.1fs",
                                path, resp.status_code, delay)
                    self.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise last_exc if last_exc else RuntimeError("request failed")

    def exchange_info(self, symbols: list[str]) -> dict:
        quoted = "[" + ",".join(f'"{s}"' for s in symbols) + "]"
        return self.get("/api/v3/exchangeInfo", {"symbols": quoted})

    def klines(self, symbol: str, start_ms: int, limit: int = 1000) -> list[list]:
        return self.get("/api/v3/klines", {
            "symbol": symbol, "interval": "1m",
            "startTime": start_ms, "limit": limit,
        })

    def first_candle_time(self, symbol: str) -> datetime:
        rows = self.klines(symbol, start_ms=0, limit=1)
        if not rows:
            raise ValueError(f"no klines returned for {symbol}")
        return datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc)


def parse_symbol_info(payload: dict) -> dict[str, dict]:
    """Flatten exchangeInfo into one dict per symbol."""
    out: dict[str, dict] = {}
    for s in payload.get("symbols", []):
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        out[s["symbol"]] = {
            "symbol": s["symbol"],
            "base_asset": s.get("baseAsset"),
            "quote_asset": s.get("quoteAsset"),
            "status": s.get("status"),
            "tick_size": Decimal(filters["PRICE_FILTER"]["tickSize"])
            if "PRICE_FILTER" in filters else None,
            "step_size": Decimal(filters["LOT_SIZE"]["stepSize"])
            if "LOT_SIZE" in filters else None,
            "min_notional": Decimal(filters["NOTIONAL"]["minNotional"])
            if "NOTIONAL" in filters else None,
        }
    return out
