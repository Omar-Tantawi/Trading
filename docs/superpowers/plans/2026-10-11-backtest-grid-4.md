# Backtesting Spot Grid Bots (Sub-project 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tb backtest grid` compares daily spot grid bots that run always, pause on forecast-wild days, or pause at random, against cash and hold, with fees and a bootstrap interval.

**Architecture:** New `backtest/` package: pure `grid.py` (one day), `engine.py` (strategies over days), `metrics.py`; `load.py` reads the DB; `report.py` renders; CLI `tb backtest grid`.

**Tech Stack:** Python 3.13, NumPy, pandas, psycopg, Typer.

**Spec:** `docs/superpowers/specs/2026-10-11-backtest-grid-4-design.md`

## Global Constraints

- `RANGE_K = 3`, `N_GRIDS = 10`, `A = 0.995 · equity / N`, `FEE = 0.001`, `SLIPPAGE = 0.0005`, start 1,000 USDT, day = 00:00–24:00 UTC.
- Intraminute path: O→L→H→C if close ≥ open else O→H→L→C.
- Decision inputs for day τ: feature row and `xgb_v1` vol3 prediction with `open_time = τ − 1h`, horizon 24.
- Thresholds 0.25, 0.30, 0.35; random control seed 0, same pause count as 0.30.
- No advice words in the report.

## Review Focus

1. **A day with missing minutes or no prediction** → cash for all strategies and counted, not a crash → test in Task 3.
2. **Price gaps past several levels in one minute** → each crossed grid fills at its level once → test in Task 1.
3. **Coin quantity too large for the cash (fees)** → never negative USDT → test in Task 1.
4. **A run that is not vol3/24h** → exit 1 with reason → test in Task 4.
5. **SOL's later start** → its test period starts with its first prediction → test in Task 3.

---

### Task 1: One grid day (`backtest/grid.py`)
**Produces:** `levels(p0, atr_pct) -> np.ndarray` (N+1 levels); `path(open, high, low, close) -> np.ndarray` (4 points per minute); `@dataclass DayResult: cash, coin, round_trips, fees, fills`; `run_day(prices: np.ndarray, levels, equity: float, coin: float, fee: float, slippage: float) -> DayResult` (rebalances at prices[0], then fills).
- [ ] Tests: oscillation across one grid → 1 round trip, profit as spec; full fall → all coin; full rise → all USDT; gap across 3 levels fills 3 grids; cash never < 0; rebalance trades only the difference.
- [ ] Implement (per grid: marks −1/+1 at ≤ L_j / ≥ L_{j+1}, state = last mark, count changes); commit.

### Task 2: Metrics (`backtest/metrics.py`)
**Produces:** `summary(equity: pd.Series, active: pd.Series, round_trips, fees, start) -> dict`; `diff_ci(ret_a, ret_b, n=1000, seed=0) -> (mean_bp, lo_bp, hi_bp)`.
- [ ] Tests on hand series (max drawdown, annual return, Sharpe sign, longest losing streak, win share); bootstrap reproducible and contains the mean.
- [ ] Implement; commit.

### Task 3: Engine and loader
**Files:** `backtest/engine.py`, `backtest/load.py`.
**Produces:** `@dataclass CoinData: minutes: pd.DataFrame (UTC index, open/high/low/close), days: pd.DataFrame (index day start; atr_pct, p_wild)`; `STRATEGIES` names; `run_strategy(data: CoinData, name, fee) -> StrategyResult(equity: pd.Series, active: pd.Series, round_trips, fees)`; `load_coin(conn, symbol, run_id) -> CoinData`; `default_run(conn) -> int | None`.
- [ ] Tests: cash flat; hold follows price; paused day flat except exit cost; incomplete day → cash, counted; prefix invariance; random control pauses the same number of days as `_30`. DB: decision rows at τ − 1h; full days only.
- [ ] Implement; commit.

### Task 4: Report and CLI, docs
**Files:** `backtest/report.py`, `data/cli.py` (`backtest_app`, `tb backtest grid`), README, handoff.
- [ ] Tests: report rows per strategy, comparison verdict lines, no advice words; CLI refuses a dir2 or 1h run; runs end to end on the fixture DB.
- [ ] Implement; full suite; commit; push.
