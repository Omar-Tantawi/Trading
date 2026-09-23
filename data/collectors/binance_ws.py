from datetime import datetime
from decimal import Decimal

from data.collectors.archive import parse_timestamp
from data.storage.repository import Candle


def stream_url(ws_base: str, symbols: list[str]) -> str:
    parts = []
    for s in symbols:
        low = s.lower()
        parts.append(f"{low}@kline_1m")
        parts.append(f"{low}@bookTicker")
    return f"{ws_base.rstrip('/')}/stream?streams=" + "/".join(parts)


def parse_kline_message(msg: dict, symbol_hint: str | None = None) -> Candle | None:
    """Returns a Candle only for CLOSED candles; None otherwise.

    A candle that is still forming will change before it closes, so storing it
    would put provisional data in the warehouse.
    """
    k = msg.get("k")
    if not k or not k.get("x"):
        return None
    return Candle(
        symbol=k.get("s") or msg.get("s") or symbol_hint,
        open_time=parse_timestamp(k["t"]),
        close_time=parse_timestamp(k["T"]),
        open=Decimal(k["o"]), high=Decimal(k["h"]), low=Decimal(k["l"]),
        close=Decimal(k["c"]), volume=Decimal(k["v"]),
        quote_volume=Decimal(k["q"]), trade_count=int(k["n"]),
        taker_buy_base=Decimal(k["V"]), taker_buy_quote=Decimal(k["Q"]),
        source="ws",
    )


def parse_book_ticker_message(msg: dict, received_at: datetime) -> tuple:
    return (
        msg["s"], received_at,
        Decimal(msg["b"]), Decimal(msg["B"]),
        Decimal(msg["a"]), Decimal(msg["A"]),
    )
