# Design: AI Trading Buddy — Sub-project 4: Backtesting Spot Grid Bots

Date: 2026-10-11
Status: Approved for planning
Source: vision §28–29, §56–63 (grid bots in the advisor), §98 (advisor tests)
Builds on: sub-projects 1 (1m candles), 2a (features), 3/3b (stored `vol3` predictions)

## 1. Purpose

Answer, with fees and honest statistics: **does the `vol3` volatility
forecast make a Binance spot grid bot better?** The test compares a grid
that runs every day with grids that pause (WAIT) on days the model expects
a wild move, plus simple baselines. Nothing here trades or recommends; the
output is a measurement of the past.

### Success criteria

1. `tb backtest grid` simulates spot grid bots minute by minute on the
   stored 1m candles of each coin, over the test periods where walk-forward
   `vol3` predictions exist (2020-01 … 2025-09; never the holdout).
2. It compares the strategies of §4 on return, annual return, maximum
   drawdown, Sharpe, Sortino, active days, round trips, fees, winning days
   and the longest losing streak (vision §29), per coin and averaged.
3. It states whether pausing on forecast-wild days beats running every
   day, as a mean daily-return difference with a 95 % block-bootstrap
   interval, and whether a random pause schedule with the same pause rate
   does as well (so timing, not just trading less, is tested).
4. No look-ahead: each day's decision uses only the prediction and
   features known at that day's start; a prefix test proves it.

### Agreed with the user (2026-10-11)

- First backtest: grid bots steered by the volatility forecast (option A).
- The user's bots: **spot grid**.

### Non-goals

- Futures grids, leverage, DCA, rebalancing bots (later versions).
- Direction strategies from `dir2` (a later backtest).
- Optimising grid settings (number of grids, range): fixed settings, so
  no tuning leaks into the result.
- Live or paper trading; storing backtests in the database; dashboard
  charts (a later addition once the report is trusted).

## 2. The simulated grid bot

A Binance-style **arithmetic spot grid**, rebuilt each day at 00:00 UTC.

- **Range:** centred on the day's opening price `p0`; lower `p0·(1 − W)`,
  upper `p0·(1 + W)`, with `W = RANGE_K · u24`, `u24 = atr_pct · √24`
  (the coin's usual 24 h move, from the 1h feature row that closed at
  00:00), `RANGE_K = 3`.
- **Grids:** `N_GRIDS = 10` equal price steps; levels `L_0 … L_10`.
- **Money:** each grid gets an equal share of the day's equity
  (`A = 0.995 · equity / N`; the 0.5 % keeps a fee cushion). Grid j trades
  a fixed quantity `q_j = A / L_j` of the coin.
- **Start:** grids entirely above `p0` hold coin (bought at `p0`), the
  others hold USDT waiting to buy at their lower level. Only the
  difference from the coin already held is traded (a market order).
- **During the day**, each grid alternates: holding USDT, it buys `q_j`
  when the price touches `L_j`; holding coin, it sells `q_j` when the
  price touches `L_{j+1}`. A completed buy-then-sell is a **round trip**.
- **Price path within a minute:** open → low → high → close when the
  candle closed up, open → high → low → close otherwise. Grid orders are
  limit orders and fill at their level.
- **End of the day:** the coin held is kept into the next day's grid if
  the next day also runs; otherwise it is sold at the next day's open.
  Equity is valued at each day's close.
- **Costs:** `FEE = 0.1 %` per fill (Binance spot base rate; the CLI takes
  `--fee`), and `SLIPPAGE = 0.05 %` on market orders (start-of-day
  rebalancing and exits). Limit fills have no slippage.
- If the price leaves the range, the grid simply stops filling until it
  comes back or the day ends (as on Binance).

*Ruling:* rebuilding the grid daily matches the 24 h forecast and keeps the
range sized to current volatility; a long-lived grid would drift out of
range and make results depend on its start date. Cost if wrong: results
describe a "re-centred daily" grid, not one you leave alone for weeks; that
can be added later as another strategy.

## 3. Data

- **1m candles** (`candles_1m`, as float) for each test day.
- **Decision inputs** for day D (start τ = D 00:00 UTC): the 1h feature row
  with `open_time = τ − 1h` (its `atr_pct`) and the stored `vol3`
  prediction of `xgb_v1` with `open_time = τ − 1h` and horizon 24 (its
  `p2` = P(wild)), from the chosen run (default: the latest walk-forward
  `vol3` run with horizon 24). Both are known at τ.
- A day is in the test only when its prediction, its `atr_pct` and a full
  1440 minutes of candles exist; other days are cash for every strategy
  (and counted).

## 4. Strategies (all start with 1,000 USDT)

| Name | Each day |
|---|---|
| `cash` | Hold USDT (0 %). |
| `hold` | Buy the coin on the first test day and keep it. |
| `grid_always` | Run the grid. |
| `grid_vol3_25`, `_30`, `_35` | Run the grid unless P(wild) ≥ 0.25 / 0.30 / 0.35; then WAIT in USDT. |
| `grid_random_30` | Pause on random days (fixed seed), as many as `grid_vol3_30` pauses. |

*Ruling:* three fixed thresholds chosen in advance (around the ~20 % base
rate of wild days), all reported; no "best" threshold is picked from the
results. The random control shows whether the forecast's timing matters
or only the fewer trading days.

## 5. Metrics and the report

Per coin and strategy, from the daily equity series: total return, annual
return (365-day years), maximum drawdown, Sharpe and Sortino (daily, ×√365,
risk-free 0), active days, round trips, fees paid (% of start), share of
winning active days, longest losing streak (active days).

Comparisons per coin and pooled over the four coins:
`grid_vol3_30 − grid_always` and `grid_random_30 − grid_always`, as mean
daily return difference in basis points with a 95 % interval
(weekly-block bootstrap, 1,000 resamples, seed 0), with a verdict line in
the words of the ML reports ("better / worse / not distinguishable").

The report is measurement: it never contains buy, sell, long, short, enter
or exit (test), and ends with "Past simulated results; real fills, fees and
outages differ. Not advice."

## 6. Commands

`tb backtest grid [--symbol S] [--run RUN_ID] [--fee 0.001]`: checks the run
is a walk-forward `vol3` run with horizon 24 (else exit 1 with the reason),
loads data per coin, simulates every strategy, prints the report.

## 7. Architecture

New package `backtest/`:

| Module | Responsibility | DB |
|---|---|---|
| `backtest/grid.py` | One day of one grid: levels, path, fills, cash/coin, round trips, fees | no |
| `backtest/engine.py` | Runs a strategy day by day over a coin's test days; equity series | no |
| `backtest/metrics.py` | Financial metrics, block-bootstrap difference | no |
| `backtest/load.py` | Minutes, decision inputs per day | yes (read) |
| `backtest/report.py` | Text report | no |

## 8. Testing

- Grid day, hand-made prices: one oscillation across one grid gives one
  round trip with profit `q·(L_{j+1} − L_j)` minus two fees; a fall through
  all levels ends fully in coin; a rise ends fully in USDT; the intrabar
  path order is applied; start-of-day rebalancing trades only the
  difference; no fills outside the range.
- Engine: `cash` stays at 1,000; `hold` follows the price minus one fee and
  slippage; a paused day keeps equity flat except the exit cost.
- No look-ahead: results up to day d are identical whether or not later
  minutes and predictions exist (prefix test), and a day's decision changes
  only when that day's own inputs change.
- Metrics on hand series (drawdown, Sharpe sign, streaks); bootstrap
  reproducible.
- DB: loader returns the right decision rows (τ − 1h) and full days only;
  CLI refuses a non-vol3 or non-24h run; report has no advice words.
