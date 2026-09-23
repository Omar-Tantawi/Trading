import csv
import hashlib
import io
import zipfile
from datetime import date, datetime, timezone
from decimal import Decimal

from data.storage.repository import Candle

EXPECTED_COLUMNS = 12
# Binance switched archive timestamps from milliseconds to microseconds for
# files written from 2025 onward. 1e14 ms is year 5138, so any value above it
# is microseconds, not a plausible millisecond timestamp.
MICROSECOND_THRESHOLD = 1e14


def monthly_url(base: str, symbol: str, year: int, month: int) -> str:
    return (f"{base.rstrip('/')}/data/spot/monthly/klines/{symbol}/1m/"
            f"{symbol}-1m-{year:04d}-{month:02d}.zip")


def daily_url(base: str, symbol: str, day: date) -> str:
    return (f"{base.rstrip('/')}/data/spot/daily/klines/{symbol}/1m/"
            f"{symbol}-1m-{day:%Y-%m-%d}.zip")


def verify_sha256(data: bytes, checksum_text: str) -> bool:
    expected = checksum_text.strip().split()[0].lower()
    return hashlib.sha256(data).hexdigest() == expected


def parse_timestamp(raw: str | int) -> datetime:
    value = int(raw)
    seconds = value / 1_000_000 if value > MICROSECOND_THRESHOLD else value / 1_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def parse_kline_csv(text: str, symbol: str, source: str = "archive") -> list[Candle]:
    candles: list[Candle] = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0].strip().lower() == "open_time":
            continue  # header row, present in 2025+ archives
        if len(row) != EXPECTED_COLUMNS:
            raise ValueError(
                f"expected {EXPECTED_COLUMNS} columns, got {len(row)}: {row!r}"
            )
        candles.append(Candle(
            symbol=symbol,
            open_time=parse_timestamp(row[0]),
            close_time=parse_timestamp(row[6]),
            open=Decimal(row[1]),
            high=Decimal(row[2]),
            low=Decimal(row[3]),
            close=Decimal(row[4]),
            volume=Decimal(row[5]),
            quote_volume=Decimal(row[7]),
            trade_count=int(row[8]),
            taker_buy_base=Decimal(row[9]),
            taker_buy_quote=Decimal(row[10]),
            source=source,
        ))
    return candles


def rows_from_zip(blob: bytes, symbol: str, source: str = "archive") -> list[Candle]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in archive, found {names}")
        text = zf.read(names[0]).decode("utf-8")
    return parse_kline_csv(text, symbol, source=source)
