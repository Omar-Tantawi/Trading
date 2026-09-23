import hashlib
import io
import zipfile
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from data.collectors.archive import (
    daily_url,
    monthly_url,
    parse_kline_csv,
    parse_timestamp,
    rows_from_zip,
    verify_sha256,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
BASE = "https://data.binance.vision"


def test_monthly_url():
    assert monthly_url(BASE, "BTCUSDT", 2024, 1) == (
        f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )


def test_daily_url():
    assert daily_url(BASE, "BTCUSDT", date(2024, 3, 5)) == (
        f"{BASE}/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03-05.zip"
    )


def test_verify_sha256_accepts_matching_and_rejects_tampered():
    blob = b"hello binance"
    digest = hashlib.sha256(blob).hexdigest()
    checksum_text = f"{digest}  BTCUSDT-1m-2024-01.zip\n"
    assert verify_sha256(blob, checksum_text) is True
    assert verify_sha256(b"tampered", checksum_text) is False


def test_parse_timestamp_handles_milliseconds_and_microseconds():
    # 2024-01-01T00:00:00Z in ms and in us must give the same instant.
    ms = parse_timestamp(1704067200000)
    us = parse_timestamp(1704067200000000)
    assert ms == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert us == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_parses_legacy_millisecond_file():
    text = (FIXTURES / "BTCUSDT-1m-2024-01-sample.csv").read_text()
    candles = parse_kline_csv(text, "BTCUSDT")
    assert len(candles) == 2
    first = candles[0]
    assert first.open_time == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert first.close == Decimal("42475.23000000")
    assert first.trade_count == 52891
    assert first.source == "archive"


def test_parses_2025_header_and_microsecond_file():
    text = (FIXTURES / "BTCUSDT-1m-2025-01-sample.csv").read_text()
    candles = parse_kline_csv(text, "BTCUSDT")
    assert len(candles) == 1
    assert candles[0].open_time == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert candles[0].close == Decimal("93480.00000000")


def test_preserves_decimal_precision_exactly():
    text = (FIXTURES / "BTCUSDT-1m-2024-01-sample.csv").read_text()
    candles = parse_kline_csv(text, "BTCUSDT")
    # 300.00000001 must survive; a float round-trip would not.
    assert candles[1].volume == Decimal("300.00000001")
    assert isinstance(candles[1].volume, Decimal)


def test_rows_from_zip_reads_the_single_member():
    text = (FIXTURES / "BTCUSDT-1m-2024-01-sample.csv").read_text()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1m-2024-01.csv", text)
    candles = rows_from_zip(buf.getvalue(), "BTCUSDT")
    assert len(candles) == 2


def test_rejects_row_with_wrong_column_count():
    with pytest.raises(ValueError, match="columns"):
        parse_kline_csv("1,2,3\n", "BTCUSDT")
