-- Prediction runs and their out-of-sample predictions (sub-project 3 spec 7.1).
-- One row in ml_runs per (evaluation, horizon); deleting a run deletes its
-- predictions. Plain tables: written once per run and read by run.
CREATE TABLE IF NOT EXISTS ml_runs (
    run_id      bigserial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    kind        text NOT NULL CHECK (kind IN ('walk_forward', 'holdout')),
    horizon     smallint NOT NULL,
    label_set   smallint NOT NULL,
    feature_set smallint NOT NULL,
    symbols     text[] NOT NULL,
    data_end    timestamptz NOT NULL,
    config      jsonb NOT NULL,
    metrics     jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    run_id    bigint NOT NULL REFERENCES ml_runs (run_id) ON DELETE CASCADE,
    model     text NOT NULL,
    symbol    text NOT NULL,
    open_time timestamptz NOT NULL,
    fold      smallint NOT NULL,
    label     smallint NOT NULL,
    p_down    double precision NOT NULL,
    p_flat    double precision NOT NULL,
    p_up      double precision NOT NULL,
    PRIMARY KEY (run_id, model, symbol, open_time)
);
