"""One opt-in test that proves the REST path works against the real
Binance API and lands verified rows in the database end to end.

Deliberately excluded from the default run (`-m "not network"` in
pyproject.toml). Run it on purpose with:

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_network_e2e.py -m network
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.collectors.archive import parse_timestamp
from data.collectors.binance_rest import BinanceRest
from data.storage.repository import Candle, upsert_candles, upsert_symbol

pytestmark = [pytest.mark.network, pytest.mark.db]

SYMBOL = "BTCUSDT"


def test_rest_klines_load_end_to_end_into_the_database(db_conn):
    upsert_symbol(db_conn, symbol=SYMBOL)
    api = BinanceRest()
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=3)

    rows = api.klines(SYMBOL, int(start.timestamp() * 1000), limit=200)
    assert rows, "Binance returned no klines for the requested window"

    candles = [
        Candle(
            symbol=SYMBOL,
            open_time=parse_timestamp(r[0]),
            close_time=parse_timestamp(r[6]),
            open=Decimal(r[1]), high=Decimal(r[2]), low=Decimal(r[3]),
            close=Decimal(r[4]), volume=Decimal(r[5]),
            quote_volume=Decimal(r[7]), trade_count=int(r[8]),
            taker_buy_base=Decimal(r[9]), taker_buy_quote=Decimal(r[10]),
            source="rest",
        )
        for r in rows
    ]
    # Only closed candles are ever stored.
    candles = [c for c in candles if c.close_time < now]
    assert candles, "every candle in the window was still open"

    written = upsert_candles(db_conn, candles)
    assert written == len(candles)

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT source, open, high, low, close FROM candles_1m "
            "WHERE symbol = %s ORDER BY open_time", (SYMBOL,)
        )
        stored = cur.fetchall()

    assert stored, "no rows landed in candles_1m"
    for source, o, h, l, c in stored:
        assert source == "rest"
        assert h >= l
        assert h >= o
        assert h >= c
        assert l <= o
        assert l <= c
