from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from data.storage.repository import upsert_symbol
from features import store
from features.pipeline import (
    FEATURE_COLUMNS, FEATURE_SET, FEATURE_TIMEFRAMES, INT32_COLUMNS,
    INT_COLUMNS, TEXT_COLUMNS, compute_features,
)

pytestmark = pytest.mark.db

SYMBOL = "BTCUSDT"


def _expected_type(col):
    if col in INT_COLUMNS:
        return "smallint"
    if col in INT32_COLUMNS:
        return "integer"
    if col in TEXT_COLUMNS:
        return "text"
    return "double precision"


def _frame(make_bars, n=300):
    return compute_features(make_bars(n, step=timedelta(hours=1)),
                            timedelta(hours=1))


def _one_row_frame():
    idx = pd.DatetimeIndex([pd.Timestamp("2022-01-01", tz="UTC")], name="open_time")
    data = {}
    for col in FEATURE_COLUMNS:
        if col in INT_COLUMNS or col in INT32_COLUMNS:
            data[col] = pd.array([1], dtype="Int64")
        elif col in TEXT_COLUMNS:
            data[col] = pd.Series(["x"], dtype=object)
        else:
            data[col] = np.array([1.0])
    return pd.DataFrame(data, index=idx)


def _scalar(db_conn, sql, params=()):
    with db_conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def test_tables_match_feature_columns(db_conn):
    expected_names = ["symbol", "open_time", "feature_set", *FEATURE_COLUMNS,
                      "computed_at"]
    for tf in FEATURE_TIMEFRAMES:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s "
                "ORDER BY ordinal_position",
                (f"features_{tf}",),
            )
            cols = cur.fetchall()
        assert [c[0] for c in cols] == expected_names, tf
        types = dict(cols)
        for col in FEATURE_COLUMNS:
            assert types[col] == _expected_type(col), f"{tf}.{col}"
        assert types["feature_set"] == "smallint"


def test_feature_tables_are_compressed_hypertables(db_conn):
    names = {f"features_{tf}" for tf in FEATURE_TIMEFRAMES}
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT hypertable_name, compression_enabled "
            "FROM timescaledb_information.hypertables "
            "WHERE hypertable_name = ANY(%s)", (list(names),))
        rows = cur.fetchall()
        assert {r[0] for r in rows} == names
        assert all(r[1] for r in rows)

        cur.execute(
            "SELECT hypertable_name, config->>'compress_after' "
            "FROM timescaledb_information.jobs "
            "WHERE proc_name = 'policy_compression' "
            "AND hypertable_name = ANY(%s)", (list(names),))
        jobs = cur.fetchall()
        assert len(jobs) == 5 and {j[0] for j in jobs} == names
        assert all(j[1] == "30 days" for j in jobs)

        cur.execute(
            "SELECT hypertable_name, attname, segmentby_column_index, "
            "orderby_column_index, orderby_asc "
            "FROM timescaledb_information.compression_settings "
            "WHERE hypertable_name = ANY(%s)", (list(names),))
        settings = {(r[0], r[1]): r for r in cur.fetchall()}
    for name in names:
        assert settings[(name, "symbol")][2] is not None
        assert settings[(name, "open_time")][3] is not None
        assert settings[(name, "open_time")][4] is False  # DESC


def test_symbol_foreign_key_on_every_table(db_conn):
    frame = _one_row_frame()
    for tf in FEATURE_TIMEFRAMES:
        with pytest.raises(Exception, match="foreign key"):
            store.upsert_features(db_conn, "NOPE", tf, frame)
        db_conn.rollback()


def test_feature_table_rejects_unknown_timeframe():
    assert store.feature_table("1h") == "features_1h"
    with pytest.raises(ValueError):
        store.feature_table("1m")
    with pytest.raises(ValueError):
        store.feature_table("1h; DROP TABLE symbols")


def test_round_trip_preserves_values(db_conn, make_bars, assert_features_match):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    assert store.upsert_features(db_conn, SYMBOL, "1h", frame) == len(frame)
    db_conn.commit()

    back = store.read_features(db_conn, SYMBOL, "1h")
    assert_features_match(frame, back)
    assert str(back.index.tz) == "UTC"
    assert back.index.name == "open_time"
    for col in FEATURE_COLUMNS:
        assert back[col].dtype == frame[col].dtype, col


def test_read_features_range_is_start_inclusive_end_exclusive(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    db_conn.commit()
    start, end = frame.index[10], frame.index[20]
    back = store.read_features(db_conn, SYMBOL, "1h", start=start, end=end)
    assert list(back.index) == list(frame.index[10:20])
    assert len(store.read_features(db_conn, SYMBOL, "1h", start=start)) == 290
    assert len(store.read_features(db_conn, SYMBOL, "1h", end=end)) == 20


def test_read_features_empty_has_columns_and_dtypes(db_conn):
    upsert_symbol(db_conn, symbol=SYMBOL)
    back = store.read_features(db_conn, SYMBOL, "1h")
    assert back.empty
    assert list(back.columns) == list(FEATURE_COLUMNS)
    assert back.index.name == "open_time" and str(back.index.tz) == "UTC"
    assert str(back["structure"].dtype) == "Int64"
    assert str(back["trend_duration"].dtype) == "Int64"
    assert str(back["ret_1"].dtype) == "float64"
    assert back["trend_regime"].dtype == object


def test_nan_is_stored_as_null(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    nan_count = int(frame["rsi_14"].isna().sum())
    assert nan_count == 42
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    db_conn.commit()
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h "
                            "WHERE rsi_14 IS NULL") == nan_count
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h "
                            "WHERE rsi_14 = 'NaN'::float8") == 0
    # every column: NULL count equals the frame's missing count, no 'NaN'
    for col in FEATURE_COLUMNS:
        assert _scalar(db_conn, f"SELECT count(*) FROM features_1h "
                                f"WHERE {col} IS NULL") == int(frame[col].isna().sum()), col
        assert _scalar(db_conn, f"SELECT count(*) FROM features_1h "
                                f"WHERE {col}::text = 'NaN'") == 0, col


def test_upsert_is_idempotent_and_updates(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    db_conn.commit()
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h") == len(frame)

    changed = frame.copy()
    row = changed.index[-1]
    changed.loc[row, "ret_1"] = 0.123456
    changed.loc[row, "trend_regime"] = "changed"
    store.upsert_features(db_conn, SYMBOL, "1h", changed)
    db_conn.commit()
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h") == len(frame)
    assert _scalar(db_conn, "SELECT ret_1 FROM features_1h WHERE open_time = %s",
                   (row,)) == pytest.approx(0.123456)
    assert _scalar(db_conn, "SELECT trend_regime FROM features_1h "
                            "WHERE open_time = %s", (row,)) == "changed"


def test_upsert_can_turn_a_value_into_null(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    row = frame.index[-1]
    assert _scalar(db_conn, "SELECT rsi_14 FROM features_1h WHERE open_time = %s",
                   (row,)) is not None
    blanked = frame.copy()
    blanked.loc[row, "rsi_14"] = float("nan")
    store.upsert_features(db_conn, SYMBOL, "1h", blanked)
    assert _scalar(db_conn, "SELECT rsi_14 FROM features_1h WHERE open_time = %s",
                   (row,)) is None


def test_upsert_empty_frame_writes_nothing(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    assert store.upsert_features(db_conn, SYMBOL, "1h", frame.iloc[0:0]) == 0
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h") == 0


def test_upsert_stamps_the_feature_set(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    store.upsert_features(db_conn, SYMBOL, "1h", _frame(make_bars))
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h "
                            "WHERE feature_set = %s", (FEATURE_SET,)) == 300


def test_last_built_latest_and_stale_feature_set(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    assert store.last_built(db_conn, SYMBOL, "1h") is None
    assert store.latest_features(db_conn, SYMBOL, "1h") is None
    assert store.has_stale_feature_set(db_conn, SYMBOL, "1h") is False

    frame = _frame(make_bars)
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    db_conn.commit()
    last = frame.index[-1]
    assert store.last_built(db_conn, SYMBOL, "1h") == last
    latest = store.latest_features(db_conn, SYMBOL, "1h")
    assert latest["open_time"] == last
    assert latest["close_pos"] == pytest.approx(frame["close_pos"].iloc[-1])
    assert latest["trend_regime"] == frame["trend_regime"].iloc[-1]
    assert store.has_stale_feature_set(db_conn, SYMBOL, "1h") is False

    with db_conn.cursor() as cur:
        cur.execute("UPDATE features_1h SET feature_set = 0 WHERE open_time = %s",
                    (frame.index[5],))
    db_conn.commit()
    assert store.has_stale_feature_set(db_conn, SYMBOL, "1h") is True
    # scoped to the symbol and the timeframe asked about
    assert store.has_stale_feature_set(db_conn, "ETHUSDT", "1h") is False
    assert store.has_stale_feature_set(db_conn, SYMBOL, "4h") is False


def test_latest_features_maps_missing_values_to_none(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    store.upsert_features(db_conn, SYMBOL, "1h", frame.iloc[:1])
    latest = store.latest_features(db_conn, SYMBOL, "1h")
    assert latest["rsi_14"] is None
    assert latest["structure"] is None


def test_infinite_value_is_refused(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    frame.loc[frame.index[-1], "ret_1"] = float("inf")
    with pytest.raises(ValueError, match="ret_1"):
        store.upsert_features(db_conn, SYMBOL, "1h", frame)


def test_two_timeframes_in_one_transaction(db_conn, make_bars):
    upsert_symbol(db_conn, symbol=SYMBOL)
    frame = _frame(make_bars)
    store.upsert_features(db_conn, SYMBOL, "1h", frame)
    store.upsert_features(db_conn, SYMBOL, "4h", frame.iloc[:50])
    db_conn.commit()
    assert _scalar(db_conn, "SELECT count(*) FROM features_1h") == 300
    assert _scalar(db_conn, "SELECT count(*) FROM features_4h") == 50
