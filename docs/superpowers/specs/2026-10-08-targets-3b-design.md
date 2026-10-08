# Design: AI Trading Buddy — Sub-project 3b: Separate Volatility and Direction Targets

Date: 2026-10-08
Status: Approved for planning
Builds on: `docs/superpowers/specs/2026-10-07-prediction-ml-3-design.md` (sub-project 3)

## 1. Purpose

Sub-project 3 measured one three-way question (up / flat / down) and found
that its skill is almost all about **size**, not **direction**
(`tb ml diagnose`). 3b asks the two questions separately, each with its own
models and its own honest score:

- **Volatility (`vol3`):** over the next H hours, will the price move
  **quietly**, **normally** or **wildly**, in either direction?
- **Direction (`dir2`):** at the end of the next H hours, will the price be
  **up** or **down**?

The original question stays as target **`move3`**, so old runs, saved
models and `tb predict` keep working. Nothing here recommends or trades.

### Success criteria

1. `tb ml evaluate --target vol3` and `--target dir2` run the same
   walk-forward evaluation, baselines, calibration and confidence intervals
   as sub-project 3, and store and report their runs.
2. Each run records its target; reports, `tb ml runs`, the dashboard and
   `tb predict` name the classes of that target.
3. Leakage canaries cover both new targets: no skill on a random walk; a
   planted volatility pattern is found by `vol3`.
4. `tb ml train --target vol3|dir2` and `tb predict` show the new targets
   beside `move3` when their models exist.
5. The holdout stays unused.

### Agreed with the user (2026-10-08)

- Volatility as three buckets (option A); direction as up / down.
- Same coins, 1h bars, horizons 1h, 4h, 24h; no new features.

### Non-goals

- New features (Market Profile is 2b), tuning, new model types.
- A volatility *number* (option B) or a yes/no big-move target (option C).
- Changing `move3`'s definition: it keeps `LABEL_SET = 1`.

## 2. Targets

A target is a `Target` (in `ml/targets.py`): `name`, `classes` (tuple of
names), `label_set` (version), and `labels(sd: SymbolData, horizon) ->
pd.Series` (Int8, NA where undefined). `TARGETS = {"move3", "vol3", "dir2"}`.

Timing is as in sub-project 3 §3.1: decision at τ = t + 1h, reference price
`c0 = close_t`; the horizon covers the bars keyed t+1h … t+H·1h, which must
all exist (otherwise NA), and `atr_pct_t` must exist (otherwise NA).

### 2.1 `move3` (unchanged)

Classes `down, flat, up`; sub-project 3 §3.2; `label_set = 1`.

### 2.2 `vol3`

The **largest move from c0 during the horizon**, either way, in log terms:

```
m = max( ln(max high over the H bars / c0),  ln(c0 / min low over the H bars) )
u = atr_pct_t · √H                      (the coin's usual move for H hours)
quiet  (0) if m <  VOL_QUIET · u         VOL_QUIET = 0.5
normal (1) if VOL_QUIET · u ≤ m < VOL_WILD · u
wild   (2) if m ≥  VOL_WILD · u          VOL_WILD  = 1.25
```

`label_set = 1`. It uses highs and lows, so `SymbolData` gains `high` and
`low` series (1h, same index as `close`).

*Ruling:* the cutoffs are fixed constants, not quantiles of the data, so a
label never depends on which rows a fold trained on. ½ and 1¼ of the usual
move are a first guess; the report prints the actual class shares, and the
base-rate comparison stays fair whatever they are. Cost if wrong: unbalanced
classes (e.g. few "wild" rows), which makes "wild" probabilities noisier;
changing them means `label_set = 2` and a new evaluation, nothing else.

### 2.3 `dir2`

```
r = ln(close_{t+H} / close_t)
down (0) if r < 0,  up (1) if r > 0,  NA if r = 0
```

`label_set = 1`. No ATR involved, but rows without `atr_pct_t` are NA too,
so `dir2` and `vol3` use exactly the same rows as `move3`.

## 3. Changes to the sub-project 3 code

- **Labels and datasets:** `build_dataset(data, horizon, symbols, target)`
  uses `target.labels`. `ml/labels.py` keeps `make_labels` for `move3`.
- **Models and baselines** take the number of classes from the target
  (`make_models(n_classes)`): base rate, rule baselines and calibration are
  already generic; logistic regression maps its classes into `n` columns;
  XGBoost uses `num_class = n`.
- **Predictions** are stored positionally. Migration `006_targets.sql`:
  `ml_runs.target text NOT NULL DEFAULT 'move3'`; in `ml_predictions`
  rename `p_down, p_flat, p_up` to `p0, p1, p2` and let `p2` be NULL (a
  two-class target has no third column). Existing rows keep their meaning
  (move3: p0 = down, p1 = flat, p2 = up).
- **Metrics, reports and summaries** use the target's class names.
- **`tb ml diagnose`** applies to `move3` runs only; it says so for others.
- **Artifacts:** `move3` models keep their paths (`models/<model>_h<H>`);
  other targets use `models/<target>_<model>_h<H>`. `meta.json` gains
  `target` and `label_set` of that target. Loading checks both.
- **Run config** records the target, its constants and its `label_set`.

## 4. Commands

- `tb ml evaluate [--target move3|vol3|dir2] [--horizon H] [--holdout]`:
  default target `move3` (unchanged behaviour). One target per run; about
  50 min per target on the PC.
- `tb ml train [--target …]`: default `move3`.
- `tb ml runs` shows the target column; `tb ml report` names the classes.
- `tb predict SYMBOL`: for each horizon, every target whose models are saved:

  ```
  next 4h (to 13:00 UTC), xgb_v1, move: down 22% | flat 54% | up 24% ...
  next 4h, xgb_v1, volatility: quiet 31% | normal 51% | wild 18%
      (quiet = stays within 0.42% of 83,747; wild = moves 1.05% or more)
  next 4h, xgb_v1, direction: down 49% | up 51%
  ```

  each followed by its skill line. A target with no saved models is skipped
  with one line ("volatility: no saved models; run `tb ml train --target
  vol3`").

## 5. Dashboard

The runs table gains a **target** column; the run view titles its charts
with the target; the size-vs-direction button shows only for `move3` runs.
Calibration and skill charts are already generic.

## 6. Testing

- `vol3` labels: hand-made bars hit each class; the excursion uses highs and
  lows of the H future bars only (a spike in bar t itself does not count);
  gaps and NULL `atr_pct` give NA; thresholds scale with √H.
- `dir2` labels: up, down, NA on r = 0; NA where `move3` is NA.
- Models with 2 classes: shape n × 2, rows sum to 1, missing class still 2
  columns.
- Store: migration renames columns; a 2-class run round-trips with `p2`
  NULL; old move3 rows read back unchanged.
- Leakage canaries (`slow`): `vol3` and `dir2` on a random walk with
  constant volatility find no skill; `vol3` on a synthetic series with
  volatility clustering finds skill; the leaked-future canary holds for
  `dir2`.
- CLI and dashboard: `--target` validated; runs show their target;
  `tb predict` prints each target and skips missing ones; no advice words.
- Real data (PC): `tb ml evaluate --target vol3`, then `--target dir2`; read
  the reports together and record them in the handoff.
