-- Sub-project 5: the user's risk settings (one row) and every grid-conditions
-- signal shown, with the settings it used.
CREATE TABLE IF NOT EXISTS risk_settings (
    id             smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    profile        text NOT NULL CHECK (profile IN ('defensive', 'return')),
    wild_threshold double precision NOT NULL,
    capital_usdt   double precision,
    max_coin_share double precision NOT NULL,
    fee            double precision NOT NULL,
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS signals (
    id            bigserial PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    symbol        text NOT NULL,
    bar_open_time timestamptz,
    status        text NOT NULL CHECK (status IN ('WAIT', 'NORMAL')),
    p_wild        double precision,
    reasons       text[] NOT NULL,
    range_low     double precision,
    range_high    double precision,
    step_pct      double precision,
    capital_limit double precision,
    settings      jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS signals_symbol_created ON signals (symbol, created_at DESC);
