-- Sub-project 6: the user's Binance spot grid bots, typed in, and their
-- profit readings. Comparisons with holding are computed when shown.
CREATE TABLE IF NOT EXISTS bots (
    id          bigserial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    name        text NOT NULL,
    symbol      text NOT NULL CHECK (symbol ~ '^[A-Z0-9]+USDT$'),
    invest_usdt double precision NOT NULL CHECK (invest_usdt > 0),
    start_time  timestamptz NOT NULL,
    fee         double precision NOT NULL CHECK (fee BETWEEN 0 AND 0.01),
    range_low   double precision,
    range_high  double precision,
    grids       integer CHECK (grids BETWEEN 2 AND 500),
    stopped_at  timestamptz,
    CHECK ((range_low IS NULL) = (range_high IS NULL)),
    CHECK (range_low IS NULL OR (range_low > 0 AND range_low < range_high))
);

CREATE TABLE IF NOT EXISTS bot_updates (
    id           bigserial PRIMARY KEY,
    bot_id       bigint NOT NULL REFERENCES bots (id) ON DELETE CASCADE,
    created_at   timestamptz NOT NULL DEFAULT now(),
    at           timestamptz NOT NULL,
    total_profit double precision NOT NULL,
    final        boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS bot_updates_bot_at ON bot_updates (bot_id, at, id);
