import pytest

pytestmark = pytest.mark.db


def test_migrations_create_hypertables(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT hypertable_name FROM timescaledb_information.hypertables"
        )
        names = {r[0] for r in cur.fetchall()}
    assert {"candles_1m", "book_ticker"} <= names


def test_migrations_create_continuous_aggregates(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT view_name FROM timescaledb_information.continuous_aggregates"
        )
        views = {r[0] for r in cur.fetchall()}
    assert {"candles_5m", "candles_15m", "candles_1h", "candles_4h", "candles_1d"} <= views


def test_migrations_are_idempotent(db_conn):
    from data.storage.db import run_migrations

    applied_again = run_migrations()
    assert applied_again == []
