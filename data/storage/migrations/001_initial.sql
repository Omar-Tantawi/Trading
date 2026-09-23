CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS symbols (
    symbol        text PRIMARY KEY,
    base_asset    text,
    quote_asset   text,
    tick_size     numeric,
    step_size     numeric,
    min_notional  numeric,
    listed_at     timestamptz,
    status        text,
    is_active     boolean DEFAULT true,
    updated_at    timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS candles_1m (
    symbol          text NOT NULL REFERENCES symbols(symbol),
    open_time       timestamptz NOT NULL,
    close_time      timestamptz NOT NULL,
    open            numeric(20,8) NOT NULL,
    high            numeric(20,8) NOT NULL,
    low             numeric(20,8) NOT NULL,
    close           numeric(20,8) NOT NULL,
    volume          numeric(30,8) NOT NULL,
    quote_volume    numeric(30,8) NOT NULL,
    trade_count     integer NOT NULL,
    taker_buy_base  numeric(30,8) NOT NULL,
    taker_buy_quote numeric(30,8) NOT NULL,
    source          text NOT NULL,
    inserted_at     timestamptz DEFAULT now(),
    PRIMARY KEY (symbol, open_time)
);

SELECT create_hypertable('candles_1m', 'open_time',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS book_ticker (
    symbol     text NOT NULL,
    ts         timestamptz NOT NULL,
    bid_price  numeric(20,8) NOT NULL,
    bid_qty    numeric(30,8) NOT NULL,
    ask_price  numeric(20,8) NOT NULL,
    ask_qty    numeric(30,8) NOT NULL,
    spread     numeric(20,8) GENERATED ALWAYS AS (ask_price - bid_price) STORED,
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('book_ticker', 'ts',
                         chunk_time_interval => INTERVAL '1 day',
                         if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id            bigserial PRIMARY KEY,
    component     text NOT NULL,
    symbol        text,
    period_start  timestamptz,
    period_end    timestamptz,
    status        text NOT NULL,
    rows_written  bigint DEFAULT 0,
    error         text,
    started_at    timestamptz DEFAULT now(),
    finished_at   timestamptz
);

CREATE INDEX IF NOT EXISTS ingestion_runs_lookup
    ON ingestion_runs (component, symbol, status, period_start);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    id              bigserial PRIMARY KEY,
    symbol          text NOT NULL,
    timeframe       text NOT NULL,
    checked_from    timestamptz,
    checked_to      timestamptz,
    total_candles   bigint,
    duplicates      bigint,
    invalid         bigint,
    missing         bigint,
    completeness_pct numeric(9,6),
    verdict         text NOT NULL,
    details         jsonb,
    created_at      timestamptz DEFAULT now()
);

-- Source precedence: archive (2) > rest (1) > ws (0).
CREATE OR REPLACE FUNCTION source_priority(s text) RETURNS int AS $$
    SELECT CASE s WHEN 'archive' THEN 2 WHEN 'rest' THEN 1 ELSE 0 END;
$$ LANGUAGE sql IMMUTABLE;

CREATE MATERIALIZED VIEW IF NOT EXISTS candles_5m
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket(INTERVAL '5 minutes', open_time) AS open_time,
       first(open, open_time)  AS open,
       max(high)               AS high,
       min(low)                AS low,
       last(close, open_time)  AS close,
       sum(volume)             AS volume,
       sum(quote_volume)       AS quote_volume,
       sum(trade_count)        AS trade_count,
       sum(taker_buy_base)     AS taker_buy_base,
       sum(taker_buy_quote)    AS taker_buy_quote
FROM candles_1m
GROUP BY symbol, 2
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS candles_15m
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket(INTERVAL '15 minutes', open_time) AS open_time,
       first(open, open_time)  AS open,
       max(high)               AS high,
       min(low)                AS low,
       last(close, open_time)  AS close,
       sum(volume)             AS volume,
       sum(quote_volume)       AS quote_volume,
       sum(trade_count)        AS trade_count,
       sum(taker_buy_base)     AS taker_buy_base,
       sum(taker_buy_quote)    AS taker_buy_quote
FROM candles_1m
GROUP BY symbol, 2
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS candles_1h
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket(INTERVAL '1 hour', open_time) AS open_time,
       first(open, open_time)  AS open,
       max(high)               AS high,
       min(low)                AS low,
       last(close, open_time)  AS close,
       sum(volume)             AS volume,
       sum(quote_volume)       AS quote_volume,
       sum(trade_count)        AS trade_count,
       sum(taker_buy_base)     AS taker_buy_base,
       sum(taker_buy_quote)    AS taker_buy_quote
FROM candles_1m
GROUP BY symbol, 2
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS candles_4h
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket(INTERVAL '4 hours', open_time) AS open_time,
       first(open, open_time)  AS open,
       max(high)               AS high,
       min(low)                AS low,
       last(close, open_time)  AS close,
       sum(volume)             AS volume,
       sum(quote_volume)       AS quote_volume,
       sum(trade_count)        AS trade_count,
       sum(taker_buy_base)     AS taker_buy_base,
       sum(taker_buy_quote)    AS taker_buy_quote
FROM candles_1m
GROUP BY symbol, 2
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS candles_1d
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket(INTERVAL '1 day', open_time) AS open_time,
       first(open, open_time)  AS open,
       max(high)               AS high,
       min(low)                AS low,
       last(close, open_time)  AS close,
       sum(volume)             AS volume,
       sum(quote_volume)       AS quote_volume,
       sum(trade_count)        AS trade_count,
       sum(taker_buy_base)     AS taker_buy_base,
       sum(taker_buy_quote)    AS taker_buy_quote
FROM candles_1m
GROUP BY symbol, 2
WITH NO DATA;

SELECT add_continuous_aggregate_policy('candles_5m',
    start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_15m',
    start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '15 minutes', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_1h',
    start_offset => INTERVAL '30 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_4h',
    start_offset => INTERVAL '90 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('candles_1d',
    start_offset => INTERVAL '365 days', end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE);

ALTER TABLE candles_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('candles_1m', INTERVAL '7 days', if_not_exists => TRUE);
