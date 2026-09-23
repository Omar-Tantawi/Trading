from datetime import datetime, timedelta, timezone

import pytest

from data.collectors.live import minutes_missing


def test_minutes_missing_computes_the_gap():
    last = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    now = datetime(2024, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
    # candles at 12:01..12:04 are missing; 12:05 has not closed yet
    assert minutes_missing(last, now) == 4


def test_no_gap_when_current():
    last = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    now = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    assert minutes_missing(last, now) == 0


def test_requires_aware_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        minutes_missing(datetime(2024, 1, 1, 12, 0),
                        datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc))


# --- I3: rate limits and reconnect backoff ----------------------------------

import asyncio
import json
import logging
from types import SimpleNamespace

from data.collectors import live
from data.collectors.binance_rest import RateLimitedError


class _FakeConn:
    def close(self):
        pass


def _collector(gap_fill):
    settings = SimpleNamespace(symbols=["BTCUSDT"], binance_ws_url="wss://example")
    collector = live.LiveCollector(settings=settings, api=object(),
                                   connect_fn=_FakeConn)
    collector.gap_fill = gap_fill
    return collector


def _record_pauses(monkeypatch, stop, after: int):
    """Record every pause the run loop takes, whichever way it sleeps, and
    stop the collector after `after` of them."""
    pauses = []

    async def fake(*args):
        pauses.append(args[-1])
        if len(pauses) >= after:
            stop.set()

    monkeypatch.setattr(live, "_pause", fake, raising=False)
    monkeypatch.setattr(asyncio, "sleep", fake)
    return pauses


async def test_rate_limit_pauses_for_retry_after_and_logs_loudly(monkeypatch, caplog):
    stop = asyncio.Event()
    pauses = _record_pauses(monkeypatch, stop, after=1)
    calls = []

    def gap_fill(conn, symbol):
        calls.append(symbol)
        raise RateLimitedError(429, 120)

    with caplog.at_level(logging.WARNING, logger="data.collectors.live"):
        await asyncio.wait_for(_collector(gap_fill).run(stop), timeout=5)

    assert pauses == [120], "must wait exactly what Binance asked for"
    assert calls == ["BTCUSDT"]
    loud = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("rate limit" in r.getMessage().lower() and "120" in r.getMessage()
               for r in loud)


async def test_reconnects_back_off_exponentially(monkeypatch):
    stop = asyncio.Event()
    pauses = _record_pauses(monkeypatch, stop, after=5)

    def gap_fill(conn, symbol):
        raise RuntimeError("network down")

    await asyncio.wait_for(_collector(gap_fill).run(stop), timeout=5)

    assert pauses[0] == live.RECONNECT_BASE_SECONDS
    assert all(b > a for a, b in zip(pauses, pauses[1:])), pauses
    assert max(pauses) <= live.RECONNECT_CAP_SECONDS


def test_reconnect_delay_is_capped_and_never_overflows():
    assert live.reconnect_delay(1) == live.RECONNECT_BASE_SECONDS
    for failures in (10, 100, 5000):
        assert live.reconnect_delay(failures) <= live.RECONNECT_CAP_SECONDS


async def test_backoff_resets_once_a_connection_delivers_messages(monkeypatch):
    """Three failures in a row: gap fill, gap fill, then a connection that
    delivers a message before dropping. The third pause starts over."""
    stop = asyncio.Event()
    pauses = []

    async def fake_pause(stop_event, seconds):
        pauses.append(seconds)
        if len(pauses) >= 3:
            stop.set()

    monkeypatch.setattr(live, "_pause", fake_pause, raising=False)

    attempts = []

    def gap_fill(conn, symbol):
        attempts.append(symbol)
        if len(attempts) <= 2:
            raise RuntimeError("REST down")
        return 0

    open_kline = {"stream": "btcusdt@kline_1m", "data": {
        "e": "kline", "s": "BTCUSDT", "k": {"t": 0, "T": 59_999, "x": False}}}

    class FakeWs:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            return self._messages()

        async def _messages(self):
            yield json.dumps(open_kline)
            raise ConnectionError("dropped")

    monkeypatch.setattr(live.websockets, "connect", lambda *a, **k: FakeWs())

    await asyncio.wait_for(_collector(gap_fill).run(stop), timeout=5)

    assert pauses[0] == live.RECONNECT_BASE_SECONDS
    assert pauses[1] > pauses[0]
    assert pauses[2] == live.RECONNECT_BASE_SECONDS
