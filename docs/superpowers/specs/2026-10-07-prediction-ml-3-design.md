# Design: AI Trading Buddy — Sub-project 3: Prediction ML

Date: 2026-10-07
Status: Approved for planning
Milestone: 3 of the programme (see the data-foundation spec §2)
Source: vision document `docs/vision/2026-09-22-original-vision.md` §18–23, §30–33, §95, §107

## 1. Purpose

Answer one question honestly: **do the 2a features carry measurable
information about where the price goes next?** Sub-project 3 builds labels,
simple baselines, a logistic regression and XGBoost, tests them walk-forward
on data they never saw, calibrates their probabilities, and states the result
with a confidence interval. A model that does not beat the simple baselines
is reported as such; that is a valid outcome, not a failure of the project.

Nothing in 3 recommends, signals or trades. A model outputs three
probabilities (down / flat / up). Turning probabilities into decisions,
including WAIT, is sub-project 5.

### Success criteria

1. `tb ml evaluate` runs a walk-forward evaluation of every model in §5 on
   1h bars of BTCUSDT, ETHUSDT, SOLUSDT and BNBUSDT for the horizons 1h, 4h
   and 24h, stores every out-of-sample prediction, and prints a report.
2. The report gives, per horizon and model, log loss, Brier score, accuracy,
   calibration error, and the log-loss **skill versus the base-rate model
   with a 95% block-bootstrap confidence interval** (§6).
3. Leakage is tested, not assumed: on a pure random walk the full pipeline
   finds no skill, and a deliberately leaked feature is detected (§8.2).
4. A final holdout period is never used for development and every look at it
   is counted (§4.3).
5. `tb ml train` saves versioned models, and `tb predict SYMBOL` prints the
   current calibrated probabilities with the measured skill beside them, in
   plain words and without advice.

### Agreed with the user (2026-10-07)

- Target: **up / down / flat** over a horizon, with a volatility-scaled
  threshold (option A).
- Horizons: **1h, 4h and 24h**, on **1h bars**, all four coins.

### Non-goals

- Trading simulation, fees, buy-and-hold equity curves, Sharpe ratio
  (sub-project 4). Buy-and-hold as a *classifier* is "always up", which the
  base-rate model already beats or matches; its financial form belongs to 4.
- Signals, WAIT logic, position sizing (5). Ensembles (§23 of the vision).
- Hyperparameter search. v1 uses fixed, conservative settings, so no tuning
  can leak into the measurement. A later version may add search inside the
  walk-forward training windows only.
- Deep learning (LSTM, transformers), LightGBM, random forest. The vision
  lists them as later options; the harness here makes adding one a small
  change.
- Market Profile, support/resistance or pattern features (2b).
- Charts (3.5).

## 2. Architecture

A new top-level package `ml/`, beside `data/` and `features/`:

| Module | Responsibility | Touches DB |
|---|---|---|
| `ml/labels.py` | Close series + `atr_pct` → class labels per horizon (§3) | no |
| `ml/dataset.py` | Pure as-of join of 1h, 4h and 1d feature frames; model-input column list (§4.1) | no |
| `ml/load.py` | Loads closes and feature frames for the symbols; checks freshness and `FEATURE_SET` | yes (read) |
| `ml/folds.py` | Walk-forward folds, purging, the holdout boundary (§4.2–4.3) | no |
| `ml/baselines.py` | Base-rate model and the rule baselines (§5.1) | no |
| `ml/models.py` | Logistic regression, XGBoost, temperature calibration (§5.2–5.3) | no |
| `ml/metrics.py` | Log loss, Brier, accuracy, ECE, reliability bins, block bootstrap (§6) | no |
| `ml/evaluate.py` | Runs every model over every fold for one horizon; returns predictions and metrics | no |
| `ml/store.py` | Writes and reads `ml_runs` and `ml_predictions` (§7.1) | yes |
| `ml/report.py` | Metrics → plain-text report | no |
| `ml/artifacts.py` | Saves and loads trained models with their metadata (§7.2) | no (files) |
| `ml/predict.py` | Latest feature rows → current probabilities → `PredictionState` and rendered text | yes (read) |

Every module that does not touch the database takes pandas objects and
returns pandas or plain Python objects, and never reads the clock, so it is
testable on synthetic data without Docker. This is the same split as 2a.

The CLI gains `tb ml evaluate`, `tb ml runs`, `tb ml report`, `tb ml train`
and `tb predict` (§7.3). Migration `005_ml.sql` adds the two tables.
`scikit-learn` and `xgboost` become runtime dependencies; `ml*` joins the
setuptools package list; `models/` is added to `.gitignore`.

## 3. Labels

### 3.1 Timing

A 1h feature row keyed `open_time = t` is known only when that bar closes,
at **decision time τ = t + 1h** (README, "Timing of a feature row"). The
reference price is that bar's close, `c₀ = close_t`. For horizon `H` (in
1h bars: 1, 4 or 24), the outcome price is the close of the bar keyed
`t + H·1h`, which is known at `τ + H·1h`.

### 3.2 Definition

```
r       = ln(close_{t+H} / close_t)
thr     = K · atr_pct_t · √H          (K = LABEL_K = 0.5)
label   = up   (2)  if r >  thr
          down (0)  if r < −thr
          flat (1)  otherwise
```

`atr_pct_t` is the 2a feature at the same row (known at τ). Scaling by
`√H` follows how random-walk spread grows with time, so the three horizons
have comparable class balance. The label is the **end-of-horizon** outcome,
not the first touch of a barrier: it answers "where is the price after H",
which is what the user asked.

*Ruling:* `K = 0.5` is a named constant. It is chosen so that, on a random
walk with per-bar spread near `atr_pct`, roughly a third to a half of rows
are flat. The report prints the actual class shares per horizon so the user
sees them. Cost if wrong: classes are unbalanced, which log loss and the
base-rate comparison still handle correctly; only the feel of "flat" changes.

The label is NULL (row dropped from training and evaluation) when:
`atr_pct_t` is NULL; the bar `t + H·1h` is missing; or any bar between them
is missing (a gap means the 1h series is not continuous over the horizon).

`LABEL_SET = 1` versions this definition. Any change to `K`, the formula or
the gap rule bumps it.

## 4. Data

### 4.1 Model inputs

**Rows:** one per (symbol, 1h bar) with a non-NULL label.

**Columns:**

- The 1h features that are scale-free (2a spec §3.2): every column of
  `FEATURE_COLUMNS` except the price-level ones (`ema_*`, `sma_*`,
  `atr_14`, `bb_upper`, `bb_lower`, `swing_high`, `swing_low`) and the
  base-unit volumes (`buy_volume`, `sell_volume`). `MODEL_FEATURES` in
  `ml/dataset.py` is this list, defined by exclusion so a new scale-free 2a
  column is picked up deliberately (a test pins the list).
- The same columns from 4h and 1d, prefixed `h4_` and `d1_`, joined **as of
  τ**: for a decision at τ, the latest 4h row whose bar has closed by τ,
  i.e. the largest `T` with `T + 4h ≤ τ`; the same for 1d with `T + 1d ≤ τ`.
  This is the multi-timeframe context of vision §6, and the most likely
  place for a leak, so §8.2 tests it directly.
- The two regime labels of each timeframe, one-hot encoded (fixed category
  lists from `features/regime.py`, so every fold has the same columns).
- The symbol, one-hot encoded. One model is trained per horizon on all
  four coins together: four times more data, and the symbol columns let it
  learn coin-specific base rates.

Missing feature values stay NaN. XGBoost handles NaN natively. Logistic
regression imputes each column's median from its training rows and adds no
indicator columns (the NaNs are almost all early warm-up rows).

### 4.2 Walk-forward folds

All rows from all symbols are ordered by τ. Folds are defined on τ:

- **Test windows:** consecutive 6-month windows (`FOLD_MONTHS = 6`), the
  first starting `FIRST_TEST = 2020-01-01`, the last ending at the holdout
  boundary (it may be shorter than 6 months).
- **Training window:** everything before the test window (expanding), minus
  a **purge** (below).
- **Calibration slice:** the last 20 % of the training window by τ
  (`CAL_FRACTION = 0.2`) is held back from fitting and used only to fit the
  calibration temperature (§5.3).
- **Purge:** a row's label is known at `τ + H·1h`. A training row is
  dropped if its label is not known before the next slice starts. So
  between fit and calibration slices, and between the calibration slice and
  the test window, the last `H` hours of rows are removed. Without this,
  a 24h label in training would overlap the first day of test.

Each fold is trained from scratch. A model never sees a row from its test
window or later.

### 4.3 Holdout

`HOLDOUT_START = 2025-10-01` (UTC, on τ). Rows with τ ≥ it are the final
test set (vision §30): never in any walk-forward fold, never looked at while
developing.

`tb ml evaluate --holdout` runs one more fold: train on everything before
`HOLDOUT_START` (with purge and calibration slice as above), test on the
holdout. Every holdout run is recorded with `kind = 'holdout'`; the report
prints **how many times the holdout has been evaluated**, and from the
second time on, warns that its result is no longer an unbiased estimate.

*Ruling:* about 12 months of holdout out of roughly 8 years of BTC/ETH
history. Cost if wrong: a shorter development period; walk-forward still
has about 11 folds.

## 5. Models

Every model implements one interface: `fit(X_fit, y_fit, X_cal, y_cal)` and
`predict_proba(X) → n × 3` (columns down, flat, up; rows sum to 1). Model
names carry a version (`MODEL_VERSIONS`, vision §95), e.g. `xgb_v1`.

### 5.1 Baselines

| Name | What it predicts |
|---|---|
| `base_rate_v1` | The class shares of the training rows, for every row. The "random model" done properly: the best you can do knowing nothing about the market. **All skill is measured against it.** |
| `ema_cross_v1` | P(class \| state) from the training rows, state = sign of `trend_duration` (EMA 20 above / below EMA 50) |
| `rsi_v1` | Same, state = RSI zone: < 30, 30–70, > 70 |
| `macd_v1` | Same, state = sign of `macd_hist_pct` |

The rule baselines are the vision's "EMA crossover, RSI strategy, MACD
strategy" turned into probability models: the frequency of each class in
each rule state, counted on training rows with +1 smoothing. A NULL state
uses the base rate. This makes them directly comparable with the ML models
on the same metrics, with no hand-picked probabilities.

### 5.2 Machine-learning models

- `logreg_v1`: multinomial logistic regression (scikit-learn), columns
  standardised on fit rows, L2 penalty `C = 0.1`, `max_iter = 1000`.
- `xgb_v1`: XGBoost `multi:softprob`, `n_estimators = 300`,
  `max_depth = 4`, `learning_rate = 0.05`, `subsample = 0.8`,
  `colsample_bytree = 0.8`, `min_child_weight = 50`, `tree_method = hist`,
  `random_state = 0`, `n_jobs = 4`. No early stopping (it would need
  another slice and is a form of tuning).

*Ruling:* fixed, conservative settings (shallow trees, slow learning, large
leaves) because daily-noise data overfits easily and no tuning means no
tuning leak. Cost if wrong: XGBoost reads weaker than a tuned version
could; the report says it is untuned.

### 5.3 Calibration

Every model's probabilities are calibrated by **temperature scaling**: one
number `T > 0`, fitted on the calibration slice by minimising log loss of
`softmax(log(p) / T)`. `T > 1` softens over-confident probabilities. It
keeps the ranking of classes and cannot overfit (one parameter). The base
rate is already calibrated by construction and gets `T = 1`.

Reliability is checked, not assumed: §6 measures ECE and reliability bins on
the test windows.

## 6. Metrics and the report

Per (horizon, model), over all walk-forward test rows pooled, and per fold:

- **Log loss** (primary): mean of −ln p(true class), with p clipped to
  [1e-15, 1].
- **Brier score**: mean squared error of the three probabilities against
  the one-hot label.
- **Accuracy**: share where the highest probability is the true class.
- **ECE** (expected calibration error): over all three classes, bins of
  width 0.1 on predicted probability; weighted mean of |mean predicted −
  observed frequency|. Reliability bins are stored for the dashboard.
- **Skill** = 1 − logloss(model) / logloss(base_rate_v1), on the same rows.
  Positive means better than knowing nothing.
- **95 % confidence interval for skill:** moving-block bootstrap over the
  test rows in τ order, block length 1 week (168 hourly decision times,
  all symbols at those times together), 1,000 resamples, fixed seed. Blocks
  keep the autocorrelation that overlapping 24h labels create, which a
  plain bootstrap would ignore and so report false certainty.

The report, per horizon: class shares; a table of models (log loss, skill
with CI, Brier, accuracy, ECE); skill per fold (does it hold up over time?);
skill per symbol. One verdict line per model, worded as measurement only:

- "skill +0.8 % (95 % CI +0.3 % … +1.3 %): better than the base rate on unseen data"
- "skill +0.1 % (95 % CI −0.4 % … +0.6 %): not distinguishable from the base rate"
- "skill −0.5 % (CI entirely below 0): worse than the base rate"

The report never contains buy, sell, long, short, enter or exit (a test
asserts it, as in 2a).

## 7. Storage, artifacts and commands

### 7.1 Tables (migration `005_ml.sql`)

`ml_runs`: `run_id bigserial PRIMARY KEY`, `created_at timestamptz NOT NULL
DEFAULT now()`, `kind text NOT NULL CHECK (kind IN ('walk_forward',
'holdout'))`, `horizon smallint NOT NULL`, `label_set smallint`,
`feature_set smallint`, `symbols text[]`, `data_end timestamptz` (newest
τ used), `config jsonb` (every constant of §3–5), `metrics jsonb` (the
whole §6 result, including reliability bins).

`ml_predictions`: `run_id bigint REFERENCES ml_runs ON DELETE CASCADE`,
`model text`, `symbol text`, `open_time timestamptz`, `fold smallint`,
`label smallint`, `p_down`, `p_flat`, `p_up double precision`;
`PRIMARY KEY (run_id, model, symbol, open_time)`. A plain table (not a
hypertable): it is written once per run and read by run. One full
evaluation stores about 4–5 million rows; `tb ml runs` shows each run's
size. Deleting old runs is left to the user (a `DELETE FROM ml_runs WHERE
run_id = …` cascades).

One `tb ml evaluate` writes one run per horizon, in one transaction per
run.

### 7.2 Model artifacts

`tb ml train` fits each model per horizon on **all labelled rows** available
(this is for current use; its quality was measured by `evaluate`) and saves
to `models/<model>_h<H>/` (git-ignored): the model file (XGBoost JSON or
scikit-learn via `joblib`), the temperature, and `meta.json` with model
name, horizon, `LABEL_SET`, `FEATURE_SET`, the column list, the training
range, the time of training, and the `run_id` of the latest walk-forward
run for that horizon (whose skill `tb predict` quotes). Loading refuses a
model whose `FEATURE_SET`, `LABEL_SET` or column list differs from the code.

### 7.3 Commands

- `tb ml evaluate [--horizon H] [--holdout]`: quality gate
  (`assert_trainable` on 1m for every symbol), freshness check (every
  `features_*` table holds `FEATURE_SET` rows within 2 steps of the 1m
  data; otherwise "run `tb features build` first" and exit 1), then for
  each horizon: build the dataset, run the folds, store, print the report.
- `tb ml runs`: list runs (id, kind, horizon, time, rows, xgb skill).
- `tb ml report RUN_ID`: print a stored run's report again.
- `tb ml train`: as §7.2; prints what was saved.
- `tb predict SYMBOL [--no-build]`: incremental feature build for the
  symbol (as `tb analyze`), then for each horizon and for `xgb_v1` and
  `logreg_v1`: the probabilities for the newest closed 1h bar, what "up" and
  "down" mean in price ("above 84,600 at 13:00 UTC"), and the measured skill
  with its CI from the run named in the artifact. If that skill's CI
  includes 0, the line says the model has not shown skill on unseen data.
  Missing models → "run `tb ml train` first", exit 1.

`tb predict` builds a `PredictionState` dataclass that serialises to a
plain dict (for the LLM in 8 and the dashboard in 3.5), and `render()`
prints it. No advice words (test).

### 7.4 Performance targets

- `tb ml evaluate` for all three horizons on the PC: under 30 minutes.
- `tb predict`: under 30 seconds after a recent build.

## 8. Testing

### 8.1 Pure units (no DB)

- Labels: hand-made closes with known returns hit each class; the threshold
  scales with `√H`; gaps and NULL `atr_pct` give NULL; the last H rows are
  NULL.
- As-of join: a 4h row keyed T is invisible at τ < T + 4h and visible at
  τ = T + 4h; the same for 1d; hand-built frames with exact expected rows.
- Folds: no test τ in training; purge removes exactly the rows whose label
  ends inside the next slice; nothing at or after `HOLDOUT_START` appears in
  walk-forward folds; the holdout fold trains only before it.
- Baselines: lookup tables on hand-made rows; base rate equals class shares.
- Temperature: recovers a known T on synthetic over-confident probabilities;
  T = 1 on calibrated ones.
- Metrics: log loss, Brier, accuracy, ECE on hand-computed examples;
  bootstrap CI is reproducible (fixed seed) and contains the point estimate.
- Models: probabilities have shape n × 3, are in [0, 1], rows sum to 1.

### 8.2 Leakage canaries (the most important tests)

1. **Random walk, no skill.** Run the real 2a pipeline on several thousand
   bars of a synthetic random walk per symbol (1h, 4h and 1d from the same
   1m-equivalent path), build the dataset, run walk-forward with
   `logreg_v1` and `xgb_v1`. The skill CI must include 0 or lie below it.
   A leak anywhere (labels, joins, purging, folds) shows up as skill.
2. **A leak is caught.** Same data plus a column holding the future return
   `r`: skill must be large (> 20 %). This proves canary 1 can fail.
3. **Planted signal is found.** A synthetic series whose next-H return
   depends on a feature: `xgb_v1` skill CI lies above 0. Proves the
   harness can see real information.
4. **Multi-timeframe leak.** A 4h column made from a 4h bar's own close,
   joined with a deliberate off-by-one (as of T instead of T + 4h), must
   produce skill on a random walk; the correct join must not.

### 8.3 Database and CLI (`-m db`)

- migration creates both tables; run and predictions round-trip
- `tb ml evaluate` on a small fixture database stores runs and prints a
  report; refuses on quality FAIL and on stale features
- the holdout counter increments and the warning appears on the second run
- `tb ml train` then `tb predict` prints every horizon; refuses a model
  whose column list differs
- no advice words in any rendered output

### 8.4 Real-data verification (on the user's PC)

1. `tb db upgrade`, `tb features build`, `tb ml evaluate`: all three
   horizons complete; record time against §7.4.
2. Read the report together. Record each model's skill and CI in the
   handoff, whatever they are.
3. Only after that, and with the user's agreement, `tb ml evaluate
   --holdout` once.
4. `tb ml train`, then `tb predict BTCUSDT` reads sensibly.
