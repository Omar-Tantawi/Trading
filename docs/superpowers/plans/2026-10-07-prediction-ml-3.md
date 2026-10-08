# Prediction ML (Sub-project 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Labels, baselines, logistic regression and XGBoost on 1h bars, measured walk-forward with calibrated probabilities and confidence intervals, plus `tb predict`.

**Architecture:** A new `ml/` package. Pure modules (labels, dataset, folds, baselines, models, metrics, evaluate, report) take pandas/NumPy and never touch the DB or clock; `ml/load.py`, `ml/store.py` and `ml/predict.py` are the only DB code. The CLI in `data/cli.py` gains a `tb ml` group and `tb predict`.

**Tech Stack:** Python 3.13, pandas 3, NumPy 2, scikit-learn ≥ 1.6, xgboost ≥ 3.0, joblib, scipy (via scikit-learn), psycopg 3, Typer, Rich, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-prediction-ml-3-design.md`

## Global Constraints

- Labels: `LABEL_K = 0.5`, `thr = LABEL_K · atr_pct_t · √H`, classes `down = 0`, `flat = 1`, `up = 2`; `LABEL_SET = 1`; `HORIZONS = (1, 4, 24)`.
- Decision time τ = `open_time + 1h`; the outcome is the close of the bar keyed `open_time + H·1h`.
- Folds: `FIRST_TEST = 2020-01-01`, `FOLD_MONTHS = 6`, `HOLDOUT_START = 2025-10-01`, `CAL_FRACTION = 0.2`, all UTC, all on τ; purge of H hours before each next slice.
- Models and order: `base_rate_v1`, `ema_cross_v1`, `rsi_v1`, `macd_v1`, `logreg_v1`, `xgb_v1`. Hyperparameters exactly as spec §5.2.
- Every predicted probability row has 3 columns (down, flat, up) in [0, 1] summing to 1.
- Bootstrap: weekly blocks of τ, 1,000 resamples, seed 0, 95 % percentile interval.
- No rendered text contains buy, sell, long, short, enter or exit as whole words.
- Commit with explicit paths; never `git add -A`; never commit `.env` or `models/`.

## Review Focus

1. **A symbol listed later (SOL from 2020-08):** its rows start mid-history; folds with no rows for a symbol must not crash per-symbol metrics → test in Task 6 (`per_symbol` skips symbols with no test rows).
2. **A fold whose test window is empty or whose training set lacks a class:** must skip the fold (empty test) or still output 3 probability columns (missing class) → tests in Task 3 and Task 5.
3. **All-NaN feature columns early in history** (1d EMA 200 for 600 days): logistic median imputation must not produce NaN when a column's training median is NaN → test in Task 5 (impute with 0 when the median is NaN).
4. **Running `tb ml evaluate` with features out of date or never built:** exit 1 with "run `tb features build` first", not a stack trace → test in Task 7.
5. **`tb predict` when the newest 1h feature row has NULL `atr_pct`** or models are missing: a clear line or exit 1, never a crash → test in Task 9.

---

### Task 1: Dependencies, package and labels

**Files:**
- Modify: `pyproject.toml` (dependencies; `include = ["data*", "features*", "ml*"]`; marker `slow`)
- Modify: `.gitignore` (add `models/`)
- Create: `ml/__init__.py`, `ml/labels.py`
- Test: `tests/unit/test_ml_labels.py`

**Interfaces:**
- Produces: `LABEL_K: float = 0.5`, `LABEL_SET: int = 1`, `HORIZONS: tuple[int, ...] = (1, 4, 24)`, `DOWN, FLAT, UP = 0, 1, 2`, `CLASSES = ("down", "flat", "up")`;
  `threshold(atr_pct: float | pd.Series, horizon: int) -> float | pd.Series`;
  `make_labels(close: pd.Series, atr_pct: pd.Series, horizon: int) -> pd.Series` (same index, dtype `Int8`, NA where undefined). `close` is on a UTC 1h `open_time` index that may have gaps.

- [ ] **Step 1: Add dependencies** `scikit-learn>=1.6`, `xgboost>=3.0`, `joblib>=1.4` to runtime; add `"slow: long-running ML tests"` to markers; `pip install -e ".[dev]"`.
- [ ] **Step 2: Write failing tests** in `tests/unit/test_ml_labels.py`:
  - `test_threshold_scales_with_sqrt_h`: `threshold(0.01, 4) == pytest.approx(0.01)` and `threshold(0.01, 1) == pytest.approx(0.005)`.
  - `test_classes`: closes `[100, 101, 100, 99, 100]` on consecutive hours, `atr_pct = 0.01` everywhere, H=1 → `[UP, DOWN, DOWN, UP, NA]` (ln(1.01)=0.00995 > 0.005).
  - `test_flat_inside_threshold`: close moves +0.2 % with atr_pct 0.01, H=1 → `FLAT`.
  - `test_gap_gives_na`: hours 0,1,3,4 (hour 2 missing), H=1 → row at hour 1 is NA; H=4 rows whose window spans the gap are NA.
  - `test_null_atr_gives_na` and `test_last_h_rows_na` (H=4 → last 4 NA).
- [ ] **Step 3: Run** `pytest tests/unit/test_ml_labels.py -v` → FAIL (import error).
- [ ] **Step 4: Implement** `ml/labels.py`. Look up the future close with `close.reindex(close.index + H·1h)`; detect gaps by checking that the positional distance between `t` and `t + H·1h` equals H (`index.get_indexer`).
- [ ] **Step 5: Run** the tests → PASS; run `pytest tests/unit -q` → all pass.
- [ ] **Step 6: Commit** `pyproject.toml .gitignore ml/__init__.py ml/labels.py tests/unit/test_ml_labels.py` — `feat: add ML labels (up/flat/down, ATR-scaled) and dependencies`.

### Task 2: Dataset (as-of join and design matrix)

**Files:**
- Create: `ml/dataset.py`
- Test: `tests/unit/test_ml_dataset.py`

**Interfaces:**
- Consumes: `features.pipeline.FEATURE_COLUMNS`, `TEXT_COLUMNS`; `features.regime` labels; `ml.labels.make_labels`.
- Produces:
  - `MODEL_FEATURES: tuple[str, ...]` = `FEATURE_COLUMNS` minus `EXCLUDED = {ema_9, ema_20, ema_50, ema_100, ema_200, sma_20, sma_50, sma_200, atr_14, bb_upper, bb_lower, swing_high, swing_low, buy_volume, sell_volume}` and minus the two text regime columns.
  - `TREND_REGIMES = ("sideways","weak_bullish","weak_bearish","strong_bullish","strong_bearish")`, `VOL_REGIMES = ("low","normal","high","extreme")`.
  - `@dataclass SymbolData: close: pd.Series; f1h: pd.DataFrame; f4h: pd.DataFrame; f1d: pd.DataFrame` (feature frames as `features.store.read_features` returns).
  - `asof_join(base_index: pd.DatetimeIndex, other: pd.DataFrame, other_step: timedelta, prefix: str) -> pd.DataFrame` — for each base `t`, the `other` row with the largest `T` such that `T + other_step <= t + 1h`; columns renamed `prefix + col`; all-NaN row if none.
  - `symbol_features(sd: SymbolData) -> pd.DataFrame` — 1h model features + `h4_`/`d1_` joins of MODEL_FEATURES and the regime text columns (`trend_regime`, `volatility_regime`, `h4_trend_regime`, … ) on the 1h index.
  - `encode(frame: pd.DataFrame, symbol: str, symbols: tuple[str, ...]) -> pd.DataFrame` — float64; regime text → one-hot `trend_regime=<label>` columns over the fixed lists (all timeframes); `symbol=<S>` one-hot over `symbols`; text columns dropped.
  - `@dataclass Dataset: X: pd.DataFrame; y: np.ndarray; tau: np.ndarray (datetime64[ns, UTC] as pandas DatetimeIndex); symbol: np.ndarray; open_time: pd.DatetimeIndex` — rows sorted by (tau, symbol), RangeIndex.
  - `build_dataset(data: dict[str, SymbolData], horizon: int) -> Dataset` — labels from `make_labels(sd.close, sd.f1h["atr_pct"], horizon)`, rows with NA label dropped.

- [ ] **Step 1: Write failing tests:**
  - `test_model_features_excludes_price_levels`: none of `EXCLUDED` in `MODEL_FEATURES`; `"rsi_14"`, `"atr_pct"`, `"dist_ema_200"` in it; `len(MODEL_FEATURES) == len(FEATURE_COLUMNS) - 17`.
  - `test_asof_4h_visible_only_after_close`: 4h rows keyed 00:00 and 04:00; base 1h index 00:00…08:00. Base `t=02:00` (τ=03:00) → NaN; `t=03:00` (τ=04:00) → the 00:00 row; `t=06:00` → 00:00 row; `t=07:00` (τ=08:00) → the 04:00 row.
  - `test_asof_1d_visible_only_after_close`: same idea with a 1d row keyed day D: invisible at base `t = D+22h`, visible at `t = D+23h`.
  - `test_encode_one_hot_fixed_columns`: a frame where all regimes are `"sideways"`/`"low"` still yields every `trend_regime=…` column (zeros except sideways), and `symbol=ETHUSDT == 1.0` when encoding ETHUSDT among four symbols; no object dtype remains.
  - `test_build_dataset_sorted_and_labelled`: two tiny synthetic symbols → `X` row count equals non-NA labels, `tau == open_time + 1h`, sorted by tau.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** with `pd.merge_asof` on `tau = t + 1h` vs `avail = T + other_step` (`direction="backward"`, `allow_exact_matches=True`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `ml/dataset.py tests/unit/test_ml_dataset.py` — `feat: add the ML dataset with an as-of multi-timeframe join`.

### Task 3: Walk-forward folds

**Files:** Create `ml/folds.py`; Test `tests/unit/test_ml_folds.py`.

**Interfaces:**
- Produces: `@dataclass(frozen=True) FoldConfig: first_test: datetime; fold_months: int; holdout_start: datetime; cal_fraction: float`; `DEFAULT_FOLDS = FoldConfig(2020-01-01 UTC, 6, 2025-10-01 UTC, 0.2)`;
  `@dataclass Fold: number: int; test_start: datetime; test_end: datetime; fit_idx: np.ndarray; cal_idx: np.ndarray; test_idx: np.ndarray` (integer positions into the Dataset);
  `walk_forward_folds(tau: pd.DatetimeIndex, horizon: int, cfg: FoldConfig) -> list[Fold]`;
  `holdout_fold(tau: pd.DatetimeIndex, horizon: int, cfg: FoldConfig) -> Fold | None` (None if no holdout rows).

Algorithm (for one test window `[s, e)`): train candidates = rows with `tau + H h <= s`; split them by τ at the `(1 − cal_fraction)` quantile of their τ into fit/cal with boundary `c`; fit = rows with `tau + H h <= c`; cal = rows with `tau >= c` (and in train candidates); test = rows with `s <= tau < e`. Windows: `s = first_test + k·fold_months` while `s < holdout_start`, `e = min(s + fold_months, holdout_start)`. Skip a window with no test rows or no fit rows. Holdout: `s = holdout_start`, `e = +∞`.

- [ ] **Step 1: Write failing tests** on an hourly τ index 2018-01-01 … 2026-01-01 with `DEFAULT_FOLDS`:
  - `test_window_boundaries`: first fold `test_start == 2020-01-01`, last `test_end == 2025-10-01`, 12 folds (2020H1 … 2025H1 is 11 + the 2025-07…10 window).
  - `test_no_overlap_and_purge` (H=24): for every fold, `max(tau[fit]) + 24h <= min(tau[cal])` and `max(tau[cal]) + 24h <= test_start`; fit, cal, test are disjoint.
  - `test_nothing_from_holdout_in_walk_forward`: all tau of all folds `< 2025-10-01`.
  - `test_holdout_fold`: test τ all `>= 2025-10-01`; train τ all `+ H h <= 2025-10-01`.
  - `test_empty_window_skipped`: τ index with a gap covering 2021H1 → no fold with that `test_start`.
- [ ] **Step 2–4: Run → FAIL, implement, run → PASS.**
- [ ] **Step 5: Commit** `ml/folds.py tests/unit/test_ml_folds.py` — `feat: add purged walk-forward folds and the holdout boundary`.

### Task 4: Metrics

**Files:** Create `ml/metrics.py`; Test `tests/unit/test_ml_metrics.py`.

**Interfaces:**
- Produces (`p`: n × 3 float array, `y`: int array):
  `log_loss(p, y) -> float` (clip 1e-15); `row_log_loss(p, y) -> np.ndarray`; `brier(p, y) -> float` (mean over rows of the sum over 3 classes); `accuracy(p, y) -> float`;
  `reliability(p, y, bins=10) -> list[dict]` (keys `lo, hi, n, mean_p, freq`, pooled over all 3 classes);
  `ece(p, y, bins=10) -> float`;
  `skill(ll_model: float, ll_base: float) -> float = 1 − ll_model / ll_base`;
  `bootstrap_skill_ci(p_model, p_base, y, tau: pd.DatetimeIndex, n=1000, seed=0) -> tuple[float, float]` — group rows by `tau.floor("7D")`-style calendar week (use ISO week start: `tau.to_period("W")`), sum per-week model loss, base loss and row count; resample weeks with replacement; skill per resample from the summed losses; return 2.5/97.5 percentiles.

- [ ] **Step 1: Write failing tests:** `test_log_loss_hand` (p=[[.2,.3,.5]], y=[2] → −ln .5); `test_brier_hand` (same → .2²+.3²+.5² = .38); `test_accuracy`; `test_ece_perfectly_calibrated_is_small` (sample y from p with 200k rows → ece < 0.01); `test_ece_overconfident_is_large`; `test_skill_zero_for_identical`; `test_bootstrap_reproducible_and_contains_point` (same seed → same CI; point skill between lo and hi).
- [ ] **Step 2–4: FAIL, implement, PASS.**
- [ ] **Step 5: Commit** `ml/metrics.py tests/unit/test_ml_metrics.py` — `feat: add ML metrics and the weekly block-bootstrap skill interval`.

### Task 5: Baselines, models and calibration

**Files:** Create `ml/baselines.py`, `ml/models.py`; Test `tests/unit/test_ml_models.py`.

**Interfaces:**
- Produces in `ml/models.py`:
  `fit_temperature(p: np.ndarray, y: np.ndarray) -> float` (minimise mean log loss of `softmax(log(p)/T)` over `log T ∈ [ln 0.05, ln 20]` with `scipy.optimize.minimize_scalar(method="bounded")`); `apply_temperature(p, T) -> np.ndarray`;
  `class Model` base: attributes `name: str`, `temperature: float = 1.0`; `fit(X_fit, y_fit, X_cal, y_cal) -> Model` (calls `_fit` then sets `temperature` from `_raw_proba(X_cal)` unless `calibrate = False`); `predict_proba(X) -> np.ndarray` (= `apply_temperature(_raw_proba(X), temperature)`); subclasses implement `_fit(X, y)` and `_raw_proba(X)`. Raw probabilities always have 3 columns even if a class is absent from `y_fit` (fill the missing class with 1e-6 and renormalise).
  `class LogReg(Model)` name `logreg_v1` (StandardScaler + `LogisticRegression(C=0.1, max_iter=1000)`; median impute from fit rows, 0 where the median is NaN).
  `class XGB(Model)` name `xgb_v1` (spec §5.2 settings, `objective="multi:softprob"`, `num_class=3`).
  `make_models() -> list[Model]` in Global Constraints order.
- Produces in `ml/baselines.py`: `class BaseRate(Model)` name `base_rate_v1`, `calibrate = False`; `class RuleBaseline(Model)` with `state(X) -> pd.Series` (NaN = unknown) and Laplace +1 counts per state; subclasses `EmaCross` (`ema_cross_v1`, sign of `trend_duration`), `RsiZone` (`rsi_v1`: 0 if `rsi_14 < 30`, 1 if `<= 70`, 2 above), `MacdSign` (`macd_v1`, sign of `macd_hist_pct`). Unknown or unseen state → base rate. Rule baselines are calibrated like the ML models.

- [ ] **Step 1: Write failing tests:**
  - `test_temperature_recovers_known_T`: true p from random logits; y sampled from p; over-confident `p_hot = softmax(logits·2)` → `fit_temperature(p_hot, y)` ≈ 2 (±0.2); on calibrated p → ≈ 1 (±0.1).
  - `test_base_rate_is_class_share`: y_fit `[0,0,1,2]` → every row `[.5,.25,.25]`.
  - `test_rsi_zone_lookup`: hand rows → the counted +1-smoothed frequencies.
  - `test_all_models_proba_shape` (parametrised over `make_models()`): random X (with NaNs and an all-NaN column) and y → shape n×3, in [0,1], rows sum to 1 (atol 1e-9), no NaN.
  - `test_missing_class_still_three_columns`: y_fit contains only 0 and 2 → 3 columns.
- [ ] **Step 2–4: FAIL, implement, PASS.**
- [ ] **Step 5: Commit** `ml/baselines.py ml/models.py tests/unit/test_ml_models.py` — `feat: add baselines, logistic regression, XGBoost and temperature calibration`.

### Task 6: Walk-forward evaluation and the leakage canaries

**Files:**
- Create: `ml/evaluate.py`, `tests/unit/ml_synthetic.py` (helper, not a test file)
- Test: `tests/unit/test_ml_evaluate.py`, `tests/unit/test_ml_leakage.py` (marked `slow`)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  `@dataclass EvalResult: predictions: pd.DataFrame` (columns `model, symbol, open_time, fold, label, p_down, p_flat, p_up`) `; metrics: dict`;
  `evaluate(ds: Dataset, horizon: int, folds: list[Fold], model_factory=make_models) -> EvalResult`;
  `summarise(pred: pd.DataFrame, horizon: int) -> dict` with keys `horizon`, `class_shares` (dict class→share over test rows), `models` (name → `{log_loss, brier, accuracy, ece, skill, skill_lo, skill_hi, reliability, per_fold: [{fold, test_start, n, skill}], per_symbol: {sym: skill}}`), `n_test`, `folds` (list of `{number, test_start, test_end, n_fit, n_cal, n_test}`). Skills are relative to `base_rate_v1` on the same rows; a symbol or fold with no rows is omitted.
  In `ml_synthetic.py`: `random_walk_symbol(seed, start, hours) -> SymbolData` (1h OHLCV random walk with ~0.6 % hourly σ; 4h and 1d bars aggregated from the 1h bars; features from `features.pipeline.compute_features` for each); `planted_signal_symbol(seed, start, hours, horizon) -> SymbolData` (returns over the next `horizon` hours are pushed up when `rsi_14` of the current bar is below 40 and down when above 60).
- Canary fold config: `FoldConfig(first_test=start + 180 days, fold_months=2, holdout_start=start + hours − 30 days, cal_fraction=0.2)`; two symbols × 9,000 hours.

- [ ] **Step 1: Write failing tests** (`test_ml_evaluate.py`): on a small random-walk dataset with only `BaseRate` and `LogReg` → predictions have one row per (model, test row); `summarise` gives `skill == 0` for `base_rate_v1`; `per_symbol` omits a symbol with no test rows.
- [ ] **Step 2: Write the canaries** (`test_ml_leakage.py`, `@pytest.mark.slow`, H=4):
  - `test_random_walk_has_no_skill`: for `logreg_v1` and `xgb_v1`, assert `skill_lo <= 0.0` (the interval includes 0 or lies below it).
  - `test_leaked_future_return_is_caught`: add column `leak = ln(close_{t+H}/close_t)` to `X` → `xgb_v1` skill > 0.20.
  - `test_planted_signal_is_found`: `xgb_v1` `skill_lo > 0`.
  - `test_off_by_one_4h_join_leaks`: build `h4_close_ret` = 4h bar return joined at `T + 0` (wrong) → `xgb_v1` skill_lo > 0; joined with `asof_join` (right) → no skill (as in the first canary).
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement** `ml/evaluate.py` (fit each fresh model per fold on `fit_idx`/`cal_idx`, predict `test_idx`). **Step 5: Run** both files → PASS; note the runtime of the slow file in the commit message.
- [ ] **Step 6: Commit** `ml/evaluate.py tests/unit/ml_synthetic.py tests/unit/test_ml_evaluate.py tests/unit/test_ml_leakage.py` — `feat: add walk-forward evaluation and leakage canaries`.

### Task 7: Storage, loading and migration

**Files:**
- Create: `data/storage/migrations/005_ml.sql`, `ml/store.py`, `ml/load.py`
- Test: `tests/integration/test_ml_store_db.py` (`-m db`), `tests/unit/test_ml_load.py` (pure parts)

**Interfaces:**
- Produces in `ml/store.py`: `save_run(conn, *, kind: str, horizon: int, symbols: list[str], data_end: datetime, config: dict, metrics: dict, predictions: pd.DataFrame) -> int` (COPY predictions; does not commit); `list_runs(conn) -> list[dict]` (run_id, kind, horizon, created_at, n_predictions, xgb_skill); `load_run(conn, run_id) -> dict | None` (all `ml_runs` columns + `holdout_count` for its horizon); `holdout_count(conn, horizon) -> int`; `latest_run_id(conn, horizon, kind="walk_forward") -> int | None`.
- Produces in `ml/load.py`: `class StaleFeaturesError(Exception)`; `check_fresh(conn, symbols) -> None` (raises if any `features_<tf>` for 1h/4h/1d has a stale `FEATURE_SET` row, no rows, or a newest row older than `build_cutoff − 2·step`); `load_symbol_data(conn, symbol) -> SymbolData` (1h closes via `features.frame.load_bars(conn, symbol, "1h", None, build_cutoff(...))["close"]`, frames via `read_features`); `run_config(horizon: int, cfg: FoldConfig) -> dict` (all constants: `LABEL_K`, `LABEL_SET`, `FEATURE_SET`, fold settings as ISO strings, model names, XGB/logreg settings).
- Migration exactly as spec §7.1 (index on `ml_predictions (run_id, model)` is the PK prefix; no extra index).

- [ ] **Step 1: Write failing tests:** db: `test_tables_exist`, `test_save_and_load_round_trip` (2 models × 3 rows; metrics dict back equal), `test_holdout_count_increments`, `test_delete_cascades`; `test_check_fresh_raises_when_empty`. Unit: `test_run_config_is_json_serialisable`.
- [ ] **Step 2: Run** unit → FAIL; db tests cannot run in the cloud session without `TEST_DATABASE_URL` — say so; run them on the PC.
- [ ] **Step 3: Implement.** **Step 4: Run** unit → PASS.
- [ ] **Step 5: Commit** the five files — `feat: store ML runs and predictions, and load training data`.

### Task 8: Report and the `tb ml` commands

**Files:**
- Create: `ml/report.py`
- Modify: `data/cli.py` (add `ml_app = typer.Typer(help="Prediction models")`, `app.add_typer(ml_app, name="ml")`; commands `evaluate`, `runs`, `report`)
- Test: `tests/unit/test_ml_report.py`, `tests/integration/test_ml_cli_db.py`

**Interfaces:**
- Produces: `verdict(skill: float, lo: float, hi: float) -> str` (the three phrasings of spec §6 exactly); `render_report(run: dict) -> str` (header with run id, kind, horizon, data end; holdout count and, if `kind == "holdout"` and count ≥ 2, the line `"Warning: the holdout has now been evaluated N times; this result is no longer an unbiased estimate."`; class shares; model table; per-fold skill line per ML model; per-symbol skill; one verdict per model; footer "Models are untuned (fixed settings, spec §5.2).").
- `tb ml evaluate [--horizon H (repeatable; default all)] [--holdout]`: connect, `assert_trainable(conn, symbols, "1m")` (DataQualityError → print, exit 1), `check_fresh` (StaleFeaturesError → print "run `tb features build` first", exit 1), load data once, then per horizon: `build_dataset`, folds (`walk_forward_folds` or `[holdout_fold]`), `evaluate`, `save_run` + commit, print `render_report(load_run(...))`.
- `tb ml runs` (Rich table), `tb ml report RUN_ID` (exit 1 if unknown).

- [ ] **Step 1: Write failing tests:** `test_verdict_phrasings` (three cases, exact strings from spec §6 with values formatted `+0.8 %` style: `f"{v:+.1%}"`), `test_report_has_no_advice_words` (regex `\b(buy|sell|long|short|enter|exit)\b`, case-insensitive, on a report rendered from a fabricated metrics dict covering all models), `test_holdout_warning_from_second_run`. db: `test_evaluate_stale_features_exits_1`, `test_evaluate_quality_fail_exits_1`.
- [ ] **Step 2–4: FAIL, implement, PASS** (unit only here).
- [ ] **Step 5: Commit** `ml/report.py data/cli.py tests/unit/test_ml_report.py tests/integration/test_ml_cli_db.py` — `feat: add tb ml evaluate, runs and report`.

### Task 9: Artifacts, `tb ml train` and `tb predict`

**Files:**
- Create: `ml/artifacts.py`, `ml/predict.py`
- Modify: `data/cli.py` (`tb ml train`, `tb predict SYMBOL [--no-build]`)
- Test: `tests/unit/test_ml_artifacts.py`, `tests/unit/test_ml_predict.py`, `tests/integration/test_ml_predict_db.py`

**Interfaces:**
- Produces in `ml/artifacts.py`: `MODELS_DIR = Path("models")`; `PREDICT_MODELS = ("logreg_v1", "xgb_v1")`; `class ArtifactMismatch(Exception)`; `save_model(model, horizon: int, meta: dict, root: Path = MODELS_DIR) -> Path` (writes `root/<name>_h<H>/model.joblib` and `meta.json`; meta gains `label_set`, `feature_set`, `columns`, `horizon`, `model`); `load_model(name: str, horizon: int, root: Path = MODELS_DIR) -> tuple[Model, dict]` (raises `FileNotFoundError` if absent, `ArtifactMismatch` if label_set/feature_set differ from code).
- Produces in `ml/predict.py`: `@dataclass HorizonPrediction: horizon; model; p_down; p_flat; p_up; up_above: float; down_below: float; outcome_time: datetime; skill: float | None; skill_lo; skill_hi; note: str | None`; `@dataclass PredictionState: symbol; bar_open_time: datetime; close: float; predictions: list[HorizonPrediction]` with `to_dict()`; `predict_rows(model, X_row: pd.DataFrame, meta) -> np.ndarray` (reindex `X_row` to `meta["columns"]`; extra/missing columns → `ArtifactMismatch`); `prediction_state(conn, symbol, root=MODELS_DIR) -> PredictionState` (latest 1h row → `encode(symbol_features(...))` of the last row; `up_above = close·exp(thr)`, `down_below = close·exp(−thr)`, `outcome_time = open_time + (H+1)h`; skill from `load_run(meta["run_id"])` if present; `note = "atr_pct missing; no prediction"` and probabilities None when `atr_pct` is NULL); `render(state) -> str` — per horizon: `"next 4h (to 13:00 UTC), xgb_v1: down 31% · flat 42% · up 27%  (up = close above 84,600; down = below 83,300)"` and the skill line using `verdict`.
- `tb ml train`: quality gate + `check_fresh`, load all symbols, for each horizon `build_dataset`; fit fit/cal split = the last `CAL_FRACTION` of τ with the H-hour purge (reuse `holdout_fold` logic: write `final_split(tau, horizon, cal_fraction) -> (fit_idx, cal_idx)` in `ml/folds.py`), save each of `PREDICT_MODELS` with `run_id = latest_run_id(conn, H)` and the training range.
- `tb predict SYMBOL`: like `tb analyze` (incremental build unless `--no-build`), then print `render`. Missing models → "run `tb ml train` first", exit 1.

- [ ] **Step 1: Write failing tests:** `test_save_load_round_trip` (tmp_path; predictions equal), `test_load_refuses_other_feature_set`, `test_predict_rows_refuses_column_mismatch`, `test_render_no_advice_words`, `test_render_null_atr_note`, `test_final_split_purges`; db: `test_train_then_predict_prints_every_horizon`.
- [ ] **Step 2–4: FAIL, implement, PASS** (unit).
- [ ] **Step 5: Commit** the files — `feat: add tb ml train and tb predict with versioned model artifacts`.

### Task 10: README, handoff and full verification

**Files:** Modify `README.md` (title line, "Prediction models (sub-project 3)" section with every command, the holdout rule in plain words, and "probabilities are measurements, not advice"), `docs/handoff/2026-10-07-session-handoff.md` (state of sub-project 3, branch, what is verified where).

- [ ] **Step 1:** Write the README section and handoff update.
- [ ] **Step 2:** Run `pytest tests/unit -q` (including slow) and record the count; state that `-m db` tests and real-data verification (spec §8.4) are pending on the PC.
- [ ] **Step 3: Commit** `README.md docs/handoff/2026-10-07-session-handoff.md` — `docs: document the prediction commands and update the handoff`.
