-- Feature tables (spec 5.1): one hypertable per timeframe, identical columns.
-- Column order is FEATURE_COLUMNS in features/pipeline.py; a DB test pins it.
-- Feature rows are derived data and are never deleted; compression only
-- bounds their size. A missing feature is NULL, never 'NaN'.
CREATE TABLE IF NOT EXISTS features_5m (
    symbol      text NOT NULL REFERENCES symbols(symbol),
    open_time   timestamptz NOT NULL,
    feature_set smallint NOT NULL,
    ret_1 double precision,
    ret_3 double precision,
    ret_6 double precision,
    ret_12 double precision,
    ret_24 double precision,
    dist_high_20 double precision,
    dist_high_100 double precision,
    dist_low_20 double precision,
    dist_low_100 double precision,
    accel_6 double precision,
    range_ratio_20 double precision,
    ema_9 double precision,
    ema_20 double precision,
    ema_50 double precision,
    ema_100 double precision,
    ema_200 double precision,
    dist_ema_9 double precision,
    dist_ema_20 double precision,
    dist_ema_50 double precision,
    dist_ema_100 double precision,
    dist_ema_200 double precision,
    sma_20 double precision,
    sma_50 double precision,
    sma_200 double precision,
    dist_sma_20 double precision,
    dist_sma_50 double precision,
    dist_sma_200 double precision,
    rsi_14 double precision,
    macd_pct double precision,
    macd_signal_pct double precision,
    macd_hist_pct double precision,
    atr_14 double precision,
    atr_pct double precision,
    plus_di_14 double precision,
    minus_di_14 double precision,
    adx_14 double precision,
    bb_upper double precision,
    bb_lower double precision,
    bb_pct_b double precision,
    bb_width double precision,
    stoch_k_14 double precision,
    stoch_d_3 double precision,
    roc_10 double precision,
    swing_high double precision,
    swing_low double precision,
    dist_swing_high double precision,
    dist_swing_low double precision,
    swing_high_dir smallint,
    swing_low_dir smallint,
    structure smallint,
    ema_slope_20 double precision,
    ema_slope_50 double precision,
    trend_duration integer,
    trend_accel double precision,
    bos smallint,
    bars_since_bos integer,
    ret_std_20 double precision,
    hist_vol_20 double precision,
    range_pct double precision,
    vol_pct_365d double precision,
    volume_ratio_20 double precision,
    volume_change double precision,
    volume_z_20 double precision,
    buy_volume double precision,
    sell_volume double precision,
    buy_share double precision,
    body_frac double precision,
    upper_wick_frac double precision,
    lower_wick_frac double precision,
    open_pos double precision,
    close_pos double precision,
    body_atr double precision,
    size_atr double precision,
    trend_regime text,
    volatility_regime text,
    computed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, open_time)
);

-- LIKE copies columns, defaults and the primary key, but not foreign keys.
CREATE TABLE IF NOT EXISTS features_15m (LIKE features_5m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS features_1h  (LIKE features_5m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS features_4h  (LIKE features_5m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS features_1d  (LIKE features_5m INCLUDING ALL);

ALTER TABLE features_15m ADD CONSTRAINT features_15m_symbol_fkey
    FOREIGN KEY (symbol) REFERENCES symbols(symbol);
ALTER TABLE features_1h ADD CONSTRAINT features_1h_symbol_fkey
    FOREIGN KEY (symbol) REFERENCES symbols(symbol);
ALTER TABLE features_4h ADD CONSTRAINT features_4h_symbol_fkey
    FOREIGN KEY (symbol) REFERENCES symbols(symbol);
ALTER TABLE features_1d ADD CONSTRAINT features_1d_symbol_fkey
    FOREIGN KEY (symbol) REFERENCES symbols(symbol);

-- Chunk width grows with the bar width so a chunk holds a useful number of rows.
SELECT create_hypertable('features_5m', 'open_time',
                         chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);
SELECT create_hypertable('features_15m', 'open_time',
                         chunk_time_interval => INTERVAL '90 days', if_not_exists => TRUE);
SELECT create_hypertable('features_1h', 'open_time',
                         chunk_time_interval => INTERVAL '365 days', if_not_exists => TRUE);
SELECT create_hypertable('features_4h', 'open_time',
                         chunk_time_interval => INTERVAL '365 days', if_not_exists => TRUE);
SELECT create_hypertable('features_1d', 'open_time',
                         chunk_time_interval => INTERVAL '365 days', if_not_exists => TRUE);

ALTER TABLE features_5m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('features_5m', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE features_15m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('features_15m', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE features_1h SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('features_1h', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE features_4h SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('features_4h', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE features_1d SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);
SELECT add_compression_policy('features_1d', INTERVAL '30 days', if_not_exists => TRUE);

