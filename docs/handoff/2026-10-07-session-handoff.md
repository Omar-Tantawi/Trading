# AI Trading Buddy: Session Handoff (for cloud sessions)

**Written:** 2026-10-07. Replaces the 2026-09-29 handoff, which described
sub-project 1 (now merged into `master`).
**Branch:** `feat/market-intelligence-2a` (sub-project 2a, not yet merged).
**Base:** `master` at `17ee56c`.

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
2. Market intelligence. **2a (features) is built and verified on this branch**;
   2b (Market Profile, from `docs/research/2026-09-23-market-profile-hypotheses.md`) comes later.
3. Prediction ML (baselines, XGBoost, walk-forward, calibration).
   3.5. Minimal read-only dashboard.
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
  (`pytest tests/unit`); the `-m db` tests need a TimescaleDB 2.30.1 instance
  reachable through `TEST_DATABASE_URL` (see `.env.example`).
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
4. **Still open:** the user's own check: compare Binance's BTCUSDT 1h chart
   (RSI 14, EMA 200) with `tb analyze` on the last closed bar. Needs fresh
   data first (`tb backfill` catches up the tail), so it must run on the PC.

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

## 6. Next steps, in order

1. **User:** the Binance RSI/EMA comparison (§5, item 4), on the PC.
2. Give the user the rulings above, then use
   **superpowers:finishing-a-development-branch** to merge
   `feat/market-intelligence-2a` into `master`. **Ask the user first**; never
   merge or push without asking.
3. Start the next piece (2b Market Profile, or sub-project 3) with
   **superpowers:brainstorming**.

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
