# AI Trading Buddy: Session Handoff (for cloud sessions)

**Written:** 2026-10-07. Replaces the 2026-09-29 handoff, which described
sub-project 1 (now merged into `master`).
**Branch:** `master`. Sub-project 2a was merged on 2026-10-07 (`771af81`);
sub-project 3 (prediction ML) was merged on 2026-10-08 (`4f59d75`).

> Trust `git log` and the spec over this document if they ever disagree.

---

## 1. What the project is

A personal **AI Trading Buddy for Binance**. It analyses crypto markets with its
own models, will later monitor and advise on the user's Binance bots, and
explains everything through an LLM assistant and a dashboard.

- The **LLM never makes predictions.** It only explains the output of
  deterministic models and risk engines, and never invents numbers.
- **WAIT is a first-class decision.** No module recommends buy/sell.
- Nothing trades real money until the late stages, and then only with manual
  confirmation.

Vision: `docs/vision/2026-09-22-original-vision.md` (the image/chart-upload
and vision-model parts are out of scope).

## 2. Build order (decided; do not re-open)

1. Data foundation: **done, merged.**
2. Market intelligence. **2a (features) is done and merged**;
   2b (Market Profile, from `docs/research/2026-09-23-market-profile-hypotheses.md`) comes later.
3. Prediction ML (baselines, XGBoost, walk-forward, calibration). **Done and
   merged** (v1: mostly a volatility forecaster; see §5b).
   3.5. Minimal read-only dashboard. **Built on `feat/dashboard-3-5`
   (2026-10-08); PC check and merge pending (§5c).**
4. Backtesting. 5. Signals, risk and portfolio. 6. Bot monitoring.
7. Bot advisor. 8. LLM assistant (the user picks the LLM then).
9. Full dashboard. 10. Demo, then shadow, then controlled real trading.

Each sub-project gets its own brainstorm → spec → plan → build cycle using the
**superpowers skills, following their conventions as written**
(`docs/superpowers/specs/`, `docs/superpowers/plans/`).

## 3. Key documents

| What | Path |
|---|---|
| Sub-project 1 spec / plan | `docs/superpowers/specs/2026-09-23-data-foundation-design.md`, `docs/superpowers/plans/2026-09-23-data-foundation.md` |
| Sub-project 2a spec (binding) / plan | `docs/superpowers/specs/2026-09-30-market-intelligence-2a-design.md`, `docs/superpowers/plans/2026-09-30-market-intelligence-2a.md` |
| Sub-project 3 spec / plan | `docs/superpowers/specs/2026-10-07-prediction-ml-3-design.md`, `docs/superpowers/plans/2026-10-07-prediction-ml-3.md` |
| Market Profile hypotheses (for 2b) | `docs/research/2026-09-23-market-profile-hypotheses.md` |
| Operator runbook | `README.md` |

The SDD ledger (`.superpowers/`) is git-ignored and lives only on the user's
PC. Everything a cloud session needs from it is copied into §5 and §6 below.

## 4. Environment facts

- **Runs on the user's Windows PC.** PostgreSQL 16 + TimescaleDB 2.30.1 in
  Docker (`timescale/timescaledb:2.30.1-pg16`), host port **5433**, bound to
  `127.0.0.1` only. Use `127.0.0.1`, never `localhost` (IPv6 `::1` hangs on
  that PC). A native PostgreSQL 18 on 5432 belongs to the user; leave it alone.
- Python **3.13** in `.venv` (plain venv). Install with `pip install -e ".[dev]"`.
  pandas 3.0 and NumPy 2.x are runtime dependencies.
- **The real market data (~17.5 M 1-minute candles) exists only on the PC.**
  A cloud session has none of it. It can run the offline unit tests
  (`pytest tests/unit`). The `-m db` tests need a TimescaleDB 2.30.1 instance:
  on 2026-10-07 a cloud session ran one by starting `dockerd` and
  `docker run -d -p 127.0.0.1:5433:5432 -e POSTGRES_USER=tb
  -e POSTGRES_PASSWORD=tb_local_dev -e POSTGRES_DB=trading_buddy_test
  timescale/timescaledb:2.30.1-pg16 postgres -c shared_preload_libraries=timescaledb`
  (an empty test database, not the user's data). The Binance network test
  cannot reach Binance from the cloud.
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT. Timeframes 5m, 15m, 1h, 4h, 1d
  are continuous aggregates of `candles_1m`; features live in `features_<tf>`.

## 5. Current state of sub-project 2a

All 9 plan tasks are done and reviewed. The final whole-branch review found
1 Critical + 1 Important + minors; one fix wave fixed them (`72514a1..9fe71b4`)
and a scoped re-review came back clean. Last count: 280 unit+integration tests
pass, 105 `-m db` tests pass.

**Real-data verification (spec §7.6), done 2026-09-30 on the PC:**

1. `tb features build --rebuild`: all 20 symbol×timeframe pairs succeed in
   **16.9 min** (target < 30 min). This also proved the Critical fix (rebuild
   over compressed chunks) on real data.
2. Feature rows = eligible aggregate buckets for all 20 pairs, first and last
   bar equal, `feature_set = 1` everywhere.
3. `tb analyze BTCUSDT` reads sensibly in **17 s** (target < 30 s).
4. The user's own check, done 2026-10-07: Binance's BTCUSDT 1h RSI(14) and
   EMA(200) match `tb analyze` (bar 2026-10-07 07:00 UTC: RSI 32.28,
   EMA 200 84,928.30). Data was caught up to 2026-10-07 08:27 UTC.
   A second read the same day gave RSI 32.4 and close/EMA 200 − 1 = −0.92 %
   (matching `tb analyze`), but quoted prices about 1,000 lower (close
   83,140, EMA 83,912); the user was asked to re-read the digits. Not a code
   question: the indicator values agree either way.

### Rulings made during 2a (each with its cost if wrong)

- `bos` is NULL until both swing highs and lows exist; `bars_since_bos` counts
  across zero bars. Cost: a few early bars read NULL instead of 0.
- `WARMUP["trend_regime"] = 84`, taken from its input `adx_14`. Cost: none
  (only which test bucket checks the column).
- Feature chunk intervals 30 d / 90 d / 365 d by table. Cost: coarse tables
  compress up to ~13 months late; they are small.
- `upsert_features` raises on ±inf (spec §3.3). Cost: a build fails loudly
  instead of storing inf.
- Builds write in batches of 100,000 rows inside one transaction. Cost:
  slightly more round trips.
- Each (symbol, timeframe) pair is isolated (spec §6.1 over the plan's
  per-symbol wording). Cost: none.
- Holes later repaired in the 1m data are not recomputed automatically; the
  README tells the operator to run `tb features build --rebuild` after a
  repair. Cost: features next to a small repaired hole stay stale until then.
- `tb analyze` exits 1 when its build was skipped or failed, but still prints
  the stored state. Cost: a script that treats exit 1 as "no output" would
  ignore that output.
- Incremental lookback is `max(400 days, 365 days + 2,100 bars)` (spec §5.3
  updated). Cost: a slightly longer incremental load.

### Deferred minors (not blocking)

Python-level ±inf check (vectorise with `np.isinf`); a few tests that could be
tighter (dtype pinning, 5m/15m/1d keys, all regime labels in the forbidden-word
test, a "failed mid-build leaves no partial writes" test); unescaped Rich
markup for a user-supplied `--symbol`; the 300k-bar performance test sits in
the default unit suite (~1.2 s).

## 5b. Current state of sub-project 3 (prediction ML)

Agreed with the user: target up / down / flat with an ATR-scaled threshold
(spec option A); horizons 1h, 4h, 24h on 1h bars, all four coins. Spec and
plan are written; all 10 plan tasks are built (executed inline,
superpowers:executing-plans; ledger in the git-ignored `.superpowers/`).

- New package `ml/`, migration `005_ml.sql`, commands `tb ml evaluate`,
  `tb ml runs`, `tb ml report`, `tb ml train`, `tb predict`.
- Final whole-branch review (fresh reviewer): no leakage found, no Critical.
  One fix pass fixed 7 findings (predict crash without recent features;
  base rate smoothing and single-class logistic fallback; `python -m
  data.cli` guard; holdout count as of the run; walk-forward `data_end`;
  walk-forward labels end before the holdout; predict refuses a symbol the
  models were not trained on).
- Last count (cloud session, 2026-10-07): **354 tests pass** in the default
  run (includes the `-m db` tests against a cloud TimescaleDB 2.30.1
  container); only the Binance network test was deselected.
- Reviewer's runtime estimate for `tb ml evaluate` on real data: about
  20–25 min (target < 30), unmeasured.
- Leakage canaries (`tests/unit/test_ml_leakage.py`, ~90 s, marked `slow`):
  random walk shows no skill (logreg −9.4 %, xgb −3.9 %, both CIs below 0:
  overfitting small synthetic data, the safe direction); a leaked future
  return and a 4h join made before the bar closes are both caught; a planted
  signal is found (xgb +7.3 %).
- **First real-data run (walk-forward, 2026-10-07, user's PC; runs 1–3).**
  Test periods 2020-01 to 2025-10, about 195k test rows per horizon. Skill
  vs the base rate with 95 % CI:

  | Horizon | Outcomes (down/flat/up) | xgb_v1 | logreg_v1 | best rule baseline |
  |---|---|---|---|---|
  | 1h | 17 / 65 / 18 % | **+1.3 % (+1.2 … +1.5)**, positive in 12/12 periods | +0.7 % (+0.5 … +0.9) | rsi +0.2 % |
  | 4h | 17 / 65 / 18 % | **+0.8 % (+0.6 … +1.0)**, positive in 11/12 | −0.4 % (−0.8 … −0.1) | ~+0.1 % |
  | 24h | 19 / 60 / 21 % | −0.6 % (−1.5 … +0.2): none | −4.2 % (−5.9 … −2.5); −31.7 % in 2021H1 | ~0 % |

  Reading: small, stable, measurable skill at 1h and 4h; none at 24h.
  Accuracy equals "always flat" (65 %): the gain is in better-shaded
  probabilities, not in more correct top picks. Not yet known whether the
  skill is about direction or only about how much the price moves, nor
  whether it is worth anything after fees (sub-project 4). Holdout not yet
  used.
- **Runtime: over an hour** on the PC (target < 30 min). Cloud profiling at
  full size (synthetic, 4 cores): per fit, logreg ~40 s (≈250 iterations)
  and xgb ~35 s; saving ~20 s per horizon. Fix: `logreg_v2` (`max_iter`
  100, same test score in profiling), XGBoost on all cores, and progress
  lines with timings. Cloud estimate after the fix: ~8.6 min per horizon on
  4 cores (~26 min total); the PC time is not yet measured.
- Fixed after the run: the report printed models in jsonb key order; now a
  fixed order (`ml/report.py`).

## 5c. Current state of sub-project 3.5 (dashboard)

Agreed with the user: approach A (FastAPI + one plain HTML page, TradingView
Lightweight Charts v5.2.1 vendored under `dashboard/static/vendor/`).
Spec `docs/superpowers/specs/2026-10-08-dashboard-3-5-design.md`, plan
`docs/superpowers/plans/2026-10-08-dashboard-3-5.md`. `tb dashboard` serves
http://127.0.0.1:8050 with Chart, Models and Data health tabs. Tested in the
cloud session against the test database, including a Chromium smoke test
(`tests/integration/test_dashboard_browser_db.py`, needs the `playwright`
package, which is not a project dependency; skipped without it). Not yet
opened on the PC with real data.

## 6. Next steps

1. Done: the first `tb ml evaluate` on real data (results above).
   The user chose option A: before any holdout run, find out whether the
   skill is about size (move vs flat) or direction. Added `tb ml diagnose
   RUN_ID` (`ml/diagnose.py`), which reads stored predictions only.
   **Result (2026-10-08, runs 1 and 2): the skill is almost all size.**

   | | 1h size | 1h direction | 4h size | 4h direction |
   |---|---|---|---|---|
   | xgb_v1 | +1.7 % (+1.5 … +1.9) | +0.2 % (+0.0 … +0.4) | +1.2 % (+1.0 … +1.4) | −0.1 % (−0.6 … +0.3) |
   | logreg_v1 | +1.1 % (+1.0 … +1.3) | −0.5 % (−0.8 … −0.2) | −0.0 % | −1.5 % (−2.2 … −0.9) |
   | rule baselines | +0.1 … +0.3 % | ~0 % | +0.1 … +0.2 % | ~0 % |

   The models know when the market will be busy or quiet (volatility), not
   which way it will go. xgb picks the right side in 52.8 % of moves at
   both 1h and 4h (base rate's side: 50.8 % / 49.8 %), but its direction
   probabilities do not beat the base rate in log loss (1h only borderline),
   so that hit rate is not an established signal.
   Re-timed `tb ml evaluate` (2026-10-08, runs 4–6, with `logreg_v2`):
   **51 min** total (target < 30): data load 1:41, then about 17–18 min per
   horizon. Per fit: xgb 16–86 s, logreg_v2 12–164 s, with no link to
   training size (CPU contention or line-search cost; not diagnosed).
   Results are the same as runs 1–3: logreg_v2 +0.7 % / −0.4 % / −4.4 %
   (1h / 4h / 24h), xgb_v1 unchanged (+1.3 % / +0.8 % / −0.6 %).
   **Decided with the user (2026-10-08):** ~50 min is accepted (spec
   target revised to < 60 min), and the **holdout stays unused** for v1;
   it is saved for the model the project will rely on.
2. Holdout: not for v1 (decided 2026-10-08).
3. Done 2026-10-08: `tb ml train` saved all 6 models; `tb predict BTCUSDT`
   printed every horizon with its skill line. It showed that predict did
   not warn when the newest bar was old (21 h; `tb live` was not running).
   Fixed: predict prints the bar's age and a STALE warning (same rule as
   `tb analyze`).
4. Done 2026-10-08: rulings given to the user; merged into `master` with
   the user's approval (merged result: 242 unit + 117 db + 4 leakage
   canaries pass).
5. The user chose 3.5 (dashboard): built, see §5c. **User:** `git pull` on
   `feat/dashboard-3-5`, `pip install -e ".[dev]"`, `tb dashboard`, check
   the three tabs; then merge (ask first). After that: 2b Market Profile or
   "3b" (separate volatility and direction targets). The holdout (from
   2025-10-01) is still unused; keep it so.

## 7. Open questions waiting on the user

- **book_ticker retention.** It currently keeps one quote per symbol per second,
  compressed after a day, with no automatic deletion. Offered and not yet
  answered: keep full detail for a rolling ~7 days and 1-per-second for older
  data. That means automatic deletion, which is the user's call.
- **Market Profile book gaps.** Pages ~121–171 and ~191–216 did not extract as
  text (probably scanned images). They need a text copy or OCR; never guess at
  their content.

## 8. Working agreements

- Use the superpowers skills and keep their conventions as written.
- Once a direction is agreed, skip per-section design approvals; decide the
  details, write the spec and plan, and execute. **Still stop before merges.**
- Explain in plain language; the user is learning Docker and the tooling.
  Explain a command before asking the user to run it.
- Git: commit with explicit paths, never `git add -A`, never commit `.env`;
  end commit messages with the Co-Authored-By line the harness specifies.
- Book material: extract ideas as testable hypotheses in our own words; never
  reproduce the book's text.
- Honesty: say when something is unverified.
