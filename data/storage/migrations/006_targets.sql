-- Sub-project 3b: runs record their target, and predictions are stored by
-- class position so a two-class target fits (p2 is NULL then).
-- Existing rows are move3: p0 = down, p1 = flat, p2 = up.
ALTER TABLE ml_runs ADD COLUMN IF NOT EXISTS target text NOT NULL DEFAULT 'move3';
ALTER TABLE ml_predictions RENAME COLUMN p_down TO p0;
ALTER TABLE ml_predictions RENAME COLUMN p_flat TO p1;
ALTER TABLE ml_predictions RENAME COLUMN p_up TO p2;
ALTER TABLE ml_predictions ALTER COLUMN p2 DROP NOT NULL;
