import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    # Host port 5433, not the container's internal 5432: this machine also
    # runs a native PostgreSQL 18 Windows service on 5432, which intercepts
    # connections meant for the tb-timescaledb container. See docker-compose.yml
    # and task-2-report.md for the full story.
    # Host 127.0.0.1, not localhost: the container is published on IPv4
    # loopback only, and on Windows localhost tries ::1 first, which never
    # answers, so the connection hangs.
    "postgresql://tb:tb_local_dev@127.0.0.1:5433/trading_buddy_test",
)


@pytest.fixture(scope="session")
def migrated_db():
    """Apply migrations once against the test database."""
    os.environ["DATABASE_URL"] = TEST_DSN
    from data.config import get_settings
    from data.storage.db import run_migrations

    get_settings.cache_clear()
    run_migrations(TEST_DSN)
    return TEST_DSN


@pytest.fixture
def db_conn(migrated_db):
    """A connection to a database emptied before each test.

    DELETE rather than TRUNCATE: TimescaleDB restricts TRUNCATE on a
    hypertable that has continuous aggregates attached.
    """
    from data.storage.db import connect

    conn = connect(migrated_db)
    try:
        with conn.cursor() as cur:
            for table in ("candles_1m", "book_ticker", "ingestion_runs",
                          "data_quality_reports", "symbols"):
                cur.execute(f"DELETE FROM {table}")
        conn.commit()
        yield conn
    finally:
        conn.close()


@pytest.fixture
def autocommit_conn(migrated_db):
    """Some TimescaleDB commands (CALL refresh_continuous_aggregate) refuse
    to run inside a transaction block."""
    import psycopg

    conn = psycopg.connect(migrated_db, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "taker_buy_base"]
_DEFAULT_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _bar_index(n, step, start):
    return pd.DatetimeIndex(
        [start + i * step for i in range(n)], name="open_time"
    ).tz_convert("UTC")


@pytest.fixture
def make_bars():
    """Factory for a seeded random-walk bar series (float64, UTC index)."""

    def _make(n=3000, *, step=timedelta(hours=1), seed=0, start=_DEFAULT_START):
        rng = np.random.default_rng(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        open_ = np.concatenate(([100.0], close[:-1]))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.003, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.003, n)))
        volume = np.exp(rng.normal(3.0, 0.5, n))
        taker = volume * rng.uniform(0.3, 0.7, n)
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume, "taker_buy_base": taker},
            index=_bar_index(n, step, start),
            columns=_BAR_COLUMNS,
        ).astype("float64")

    return _make


@pytest.fixture
def bars_from():
    """Factory for explicit bars. opens default to closes, volumes to 1.0,
    taker-buy volume to half the volume."""

    def _from(*, highs, lows, closes, opens=None, volumes=None, taker=None,
              step=timedelta(hours=1), start=_DEFAULT_START):
        n = len(closes)
        closes = np.asarray(closes, dtype="float64")
        opens = closes.copy() if opens is None else np.asarray(opens, dtype="float64")
        volumes = (np.ones(n) if volumes is None
                   else np.asarray(volumes, dtype="float64"))
        taker = volumes / 2 if taker is None else np.asarray(taker, dtype="float64")
        return pd.DataFrame(
            {"open": opens,
             "high": np.asarray(highs, dtype="float64"),
             "low": np.asarray(lows, dtype="float64"),
             "close": closes, "volume": volumes, "taker_buy_base": taker},
            index=_bar_index(n, step, start),
            columns=_BAR_COLUMNS,
        )

    return _from


@pytest.fixture
def assert_features_match():
    """Compare two feature frames the way spec section 5.3 defines equal:
    same index and columns; floats within numpy.isclose(rtol=1e-9, atol=1e-9)
    with NaN == NaN; integers and text exactly, with NA == NA."""

    def _assert(a, b):
        pd.testing.assert_index_equal(a.index, b.index)
        assert list(a.columns) == list(b.columns)
        for col in a.columns:
            x, y = a[col], b[col]
            null_x, null_y = x.isna().to_numpy(), y.isna().to_numpy()
            assert (null_x == null_y).all(), f"{col}: nulls differ"
            keep = ~null_x
            if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
                xv = x.to_numpy(dtype="float64")[keep]
                yv = y.to_numpy(dtype="float64")[keep]
                bad = ~np.isclose(xv, yv, rtol=1e-9, atol=1e-9)
                assert not bad.any(), (
                    f"{col}: {bad.sum()} values differ, e.g. {xv[bad][:3]} vs {yv[bad][:3]}")
            else:
                xv, yv = x.to_numpy(dtype=object)[keep], y.to_numpy(dtype=object)[keep]
                assert (xv == yv).all(), f"{col}: values differ"

    return _assert
