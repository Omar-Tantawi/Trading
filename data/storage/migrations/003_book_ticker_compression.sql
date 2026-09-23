-- book_ticker measured ~393 rows/s (~5 GB/day) uncompressed. The writer now
-- keeps at most one row per symbol per second (see upsert_book_ticker);
-- compression bounds what is kept. There is deliberately NO retention
-- policy: deleting recorded market data is the user's decision.
ALTER TABLE book_ticker SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('book_ticker', INTERVAL '1 day', if_not_exists => TRUE);
