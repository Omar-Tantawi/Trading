from datetime import datetime, timezone

import httpx
import pytest
import respx

from data.collectors.backfill import ArchiveDownloader, months_between

BASE = "https://data.binance.vision"


def test_months_between_is_inclusive():
    start = datetime(2023, 11, 15, tzinfo=timezone.utc)
    end = datetime(2024, 2, 3, tzinfo=timezone.utc)
    assert months_between(start, end) == [
        (2023, 11), (2023, 12), (2024, 1), (2024, 2)
    ]


@respx.mock
def test_missing_month_returns_none_and_does_not_raise():
    """A month before the symbol was listed 404s. That is normal."""
    url = f"{BASE}/data/spot/monthly/klines/SOLUSDT/1m/SOLUSDT-1m-2017-01.zip"
    route = respx.get(url).mock(return_value=httpx.Response(404))
    dl = ArchiveDownloader(BASE, sleep=lambda _: None)

    assert dl.fetch_month("SOLUSDT", 2017, 1) is None
    assert route.call_count == 1  # not retried


@respx.mock
def test_checksum_mismatch_raises():
    url = f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    respx.get(url).mock(return_value=httpx.Response(200, content=b"payload"))
    respx.get(url + ".CHECKSUM").mock(
        return_value=httpx.Response(200, text="deadbeef  BTCUSDT-1m-2024-01.zip")
    )
    dl = ArchiveDownloader(BASE, sleep=lambda _: None)

    with pytest.raises(ValueError, match="checksum"):
        dl.fetch_month("BTCUSDT", 2024, 1)


@respx.mock
def test_archive_download_retries_timeouts():
    url = f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    route = respx.get(url).mock(side_effect=[
        httpx.ReadTimeout("read timed out"),
        httpx.Response(200, content=b"payload"),
    ])
    sleeps: list[float] = []
    dl = ArchiveDownloader(BASE, sleep=sleeps.append)
    assert dl._get(url) == b"payload"
    assert route.call_count == 2
    assert len(sleeps) == 1


@respx.mock
def test_archive_download_does_not_sleep_after_final_attempt():
    url = f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    route = respx.get(url).mock(return_value=httpx.Response(503))
    sleeps: list[float] = []
    dl = ArchiveDownloader(BASE, sleep=sleeps.append, attempts=4)
    with pytest.raises(RuntimeError, match="gave up"):
        dl._get(url)
    assert route.call_count == 4
    assert len(sleeps) == 3, "no sleep may follow the last attempt"


@respx.mock
def test_archive_download_gives_up_on_persistent_transport_error():
    url = f"{BASE}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    route = respx.get(url).mock(side_effect=httpx.ConnectError("refused"))
    sleeps: list[float] = []
    dl = ArchiveDownloader(BASE, sleep=sleeps.append, attempts=3)
    with pytest.raises(RuntimeError, match="gave up"):
        dl._get(url)
    assert route.call_count == 3
    assert len(sleeps) == 2
