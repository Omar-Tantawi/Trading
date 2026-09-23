import os

import pytest

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    # Host port 5433, not the container's internal 5432: this machine also
    # runs a native PostgreSQL 18 Windows service on 5432, which intercepts
    # connections meant for the tb-timescaledb container. See docker-compose.yml
    # and task-2-report.md for the full story.
    "postgresql://tb:tb_local_dev@localhost:5433/trading_buddy_test",
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
