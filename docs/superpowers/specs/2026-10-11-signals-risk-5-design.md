# Design: AI Trading Buddy — Sub-project 5: Grid Conditions and Risk Settings

Date: 2026-10-11
Status: Approved for planning
Source: vision §24 (signal engine), §25–27 (risk, risk/reward, sizing)
Builds on: 3b (`vol3` 24h model), 4 (grid backtest: pause rules and profiles)

## 1. Purpose

Each day, for each coin, state plainly whether conditions look like the
days a spot grid ran in the backtest, or like the days it paused (WAIT),
and why; apply the user's risk settings; suggest the grid range and the
capital limit those settings allow. The user decides; nothing here places,
changes or recommends orders, and nothing says buy or sell.

### Success criteria

1. `tb signals` prints, per coin: a status (**WAIT** or **GRID CONDITIONS
   NORMAL**), every reason behind it, the forecast P(wild) next 24 h with
   the threshold it was compared with, the suggested grid range and step,
   and the capital limit for the coin.
2. `tb risk show` / `tb risk set` keep the user's settings (§3) in the
   database; every signal records the settings it used.
3. Gates that make the status WAIT regardless of the forecast: no 1m data,
   data quality FAIL, newest 1h bar STALE, no saved `vol3` 24 h model or a
   model that no longer fits, missing `atr_pct`.
4. Every signal is stored (history for the dashboard and for checking
   later how WAIT days actually went) and shown on a dashboard tab.
5. No advice words in any output (test).

### Agreed with the user (2026-10-11)

- Option A: daily grid-conditions per coin with risk settings.

### Non-goals (and where they go)

- **Daily/weekly loss limits**: they need the user's real bot results,
  which sub-project 6 (bot monitoring) records. Not faked here.
- Direction signals (`dir2` edge not shown to survive fees), any buy/sell
  wording, placing or changing orders.
- Choosing the profile or threshold from the data; they are the user's
  settings, with the backtest's evidence shown beside them.

## 2. The signal

For each configured coin, at the newest closed 1h bar (`t`, closing at
τ = t + 1h):

1. **Gates**, in order; each failing gate adds a reason and forces WAIT:
   - no 1m candles stored → "no market data"
   - the incremental feature build was skipped for 1m quality FAIL →
     "data quality FAIL"
   - newest 1h bar older than 2 h 5 min (the `tb predict` / `tb analyze`
     rule) → "data STALE (… old)"
   - no saved `vol3` `xgb_v1` 24 h model, or `ArtifactMismatch` →
     "no forecast model (run `tb ml train --target vol3`)"
   - `atr_pct` NULL → "volatility measure missing"
2. **Forecast**: P(wild) = the `vol3` `xgb_v1` 24 h model's probability of
   "wild" for the bar `t` (as `tb predict`). If P(wild) ≥ the user's
   `wild_threshold` → WAIT, reason "wild day expected (P(wild) 38 % ≥ 25 %)".
3. Otherwise → **GRID CONDITIONS NORMAL**, reason "P(wild) 12 % < 25 %".

Also shown, whatever the status:

- **Suggested range** from the backtest's grid: lower/upper
  `close · (1 ∓ 3 · atr_pct · √24)`, 10 grids, the step in % and in price,
  and the step compared with two fees at the user's fee rate (a step that
  cannot pay two fees is flagged).
- **Capital limit per coin**: `capital_usdt × max_coin_share` (if capital
  is set), else just the share.
- **Profile note**: `defensive` → "in the backtest, WAIT days were spent in
  USDT (smallest drops: 2022 −19 % vs −45 % always-on)"; `return` → "in
  the backtest, WAIT days kept the coin and only cancelled orders (higher
  return, drops like always-on)". Numbers quoted are from the 2026-10-11
  backtest (run 10, fee 0.075 %), stated as such.

## 3. Risk settings

One row in a new table `risk_settings` (migration `007_signals.sql`), with
defaults applied when absent:

| Setting | Default | Allowed |
|---|---|---|
| `profile` | `defensive` | `defensive`, `return` |
| `wild_threshold` | 0.25 | 0.05 … 0.95 |
| `capital_usdt` | none | ≥ 0 |
| `max_coin_share` | 0.25 | 0.01 … 1 |
| `fee` | 0.00075 | 0 … 0.01 |

*Ruling:* defaults are the backtest's best-protected setting (threshold
0.25, defensive) and the user's BNB fee rate. Cost if wrong: the user
changes them with `tb risk set`; every stored signal records what it used.

## 4. Storage

`signals` table: `id`, `created_at`, `symbol`, `bar_open_time` (t),
`status` (`WAIT` / `NORMAL`), `p_wild` (nullable), `reasons` (text[]),
`range_low`, `range_high`, `step_pct` (nullable), `capital_limit`
(nullable), `settings` (jsonb snapshot). `tb signals` writes one row per
coin per run (a re-run for the same bar adds another row; the dashboard
shows the newest per coin).

## 5. Commands and dashboard

- `tb signals [--no-build]`: like `tb analyze`, builds features first
  (unless `--no-build`), then evaluates, stores and prints every coin.
- `tb risk show`, `tb risk set [--profile] [--wild-threshold] [--capital]
  [--max-coin-share] [--fee]`: validated; prints the new settings.
- Dashboard: a **Signals** tab with the newest signal per coin (status,
  reasons, P(wild), range, limit, time) and the current settings. Read-only.

## 6. Architecture

New package `signals/`:

| Module | Responsibility | DB |
|---|---|---|
| `signals/settings.py` | `RiskSettings` dataclass, defaults, validation, load/save | yes |
| `signals/engine.py` | Pure: inputs → `Signal` (status, reasons, range, limit) | no |
| `signals/collect.py` | Per coin: gates' inputs, P(wild) from the saved model | yes (read) |
| `signals/store.py` | Write and read `signals` | yes |
| `signals/render.py` | Text output | no |

## 7. Testing

- Engine (pure): each gate alone forces WAIT with its reason; several
  gates list all reasons; threshold boundary (P = threshold → WAIT);
  range and step maths; flag when step ≤ 2 × fee; capital limit with and
  without capital.
- Settings: defaults, validation errors, round trip.
- Collect/store/CLI (`-m db`): a coin without data → WAIT "no market
  data"; with a trained vol3 model → a P(wild) and a status; without the
  model → WAIT; stored rows; `tb risk set` rejects bad values.
- Dashboard: `/api/signals` returns the newest per coin; tab renders.
- No advice words in `tb signals`, `tb risk show` and the tab.
