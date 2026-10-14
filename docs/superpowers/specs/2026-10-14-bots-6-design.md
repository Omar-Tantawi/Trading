# Design: AI Trading Buddy — Sub-project 6: Bot Monitoring (v1: bots vs holding)

Date: 2026-10-14
Status: Approved for planning
Source: vision §51–55 (bot monitoring, registry, performance, comparison), §110
Builds on: 1 (stored 1m candles), 5 (`risk_settings.fee`)

## 1. Purpose

The user runs Binance **spot grid bots**, started and stopped at any time,
always funded with **USDT only**. Today they check each bot on its profit
and history page in the Binance app. The question that page cannot answer:
**did this bot actually beat simply holding the coin?** — buying the coin
with the same USDT at the bot's start and keeping it.

v1 answers that per bot and for all bots together, from numbers the user
types in, and keeps a history. Nothing here recommends or places orders;
it measures.

### Success criteria

1. `tb bots add` records a bot (coin, start time, USDT invested; optional
   name, price range, grid count).
2. `tb bots update ID --profit X` records the bot's total profit (USDT) as
   the app shows it, at a time (default now); `tb bots stop ID --profit X`
   records the final profit and marks the bot stopped.
3. For every update the system computes what holding would have given at
   that time, from the stored 1m prices, and shows bot vs hold in USDT and
   percent, with a plain verdict line ("bot ahead of holding by 12.30 USDT
   (+2.5 pts)" / "behind" / "level").
4. `tb bots list` shows every bot at its latest update plus totals over all
   bots; `tb bots show ID` shows one bot's full update history.
5. A read-only dashboard tab **Bots** shows the same as `tb bots list`.
6. No advice words in any output (test).

### Agreed with the user (2026-10-14)

- Sub-project 6, starting with bots vs holding.
- Bots are checked in the Binance app (profit and history page); the number
  of bots varies.
- Numbers are **typed in** (the user's choice after automatic reading was
  raised; whether Binance's API exposes spot grid bots was not checked).
- Bots are funded with USDT only.
- The figure typed in is the app's **Total profit** (not Grid profit).

### Non-goals (and where they go)

- Automatic reading from Binance (later, if its API allows; vision §51
  "manual at first").
- Per-trade data, fees split, drawdown, health checks (vision §53, §58) —
  they need trade-level data the user does not type in.
- **Daily/weekly loss limits** (deferred from sub-project 5): profit
  updates arrive whenever the user types them, not daily, so a daily limit
  cannot be checked reliably. Revisit with automatic reading.
- Futures, DCA or other bot types.

## 2. Bots and updates

A **bot**: `id` (assigned), `name` (optional, default `<SYMBOL> grid`),
`symbol` (e.g. `BTCUSDT`; must end in `USDT`), `invest_usdt` (> 0),
`start_time`, optional `range_low`, `range_high` (both or neither,
`0 < low < high`), `grids` (2 … 500), `stopped_at` (NULL while running),
`fee` (the user's `risk_settings` fee when the bot was added).

*Ruling (builder check):* the fee is fixed per bot at `add`, so a later
`tb risk set --fee` does not rewrite past comparisons.

An **update**: `bot_id`, `at` (time of the reading), `total_profit` (USDT,
may be negative, must be > −invest), `final` (true for the stop update).
`total_profit` is the app's **Total profit** field, not **Grid profit**
(confirmed with the user 2026-10-14: the page shows both). Total profit
includes the gain or loss on the coin the bot holds, which the comparison
with holding needs; the help text of `--profit` says so.
Rules: `start_time ≤ at ≤ now`; no update after a final one; `at` not
before the bot's previous update. Several updates may share a day.

**Times typed by the user** (`--start`, `--at`) are read as the PC's local
time (the app shows local time; the user's PC is UTC+3), format
`YYYY-MM-DD HH:MM`, or with `--utc` as UTC; stored in UTC; every command
echoes the UTC time it stored. *Ruling:* local time because it is what the
app shows; cost if wrong: a 3-hour shift of the hold comparison's start
price, visible in the echo and fixable by deleting and re-adding the bot.

## 3. The comparison

For a bot and an update at time `T`:

- **Price at a time t** = the open of the stored 1m candle with the latest
  `open_time ≤ t`, if that candle is at most 10 minutes before `t`;
  otherwise the comparison is unavailable with the reason "no stored price
  for BTCUSDT near 2026-10-14 06:30 UTC (run `tb backfill`, or add the coin
  to SYMBOLS)".
- **Hold value** at T = `invest × (1 − fee) × P(T) / P(start)`, with `fee`
  = the user's fee from `risk_settings` (one buy, default 0.075 %).
- **Bot value** at T = `invest + total_profit`.
- Shown: bot profit and %, hold profit and %, difference
  `bot − hold` in USDT and in percentage points of the investment, days
  running, and a verdict: ahead / behind / level (|diff| < 0.01 USDT).
- **All bots**: sums over every bot's latest update (running and stopped)
  of invested, bot profit, hold profit and difference; bots without a
  comparison are counted and excluded from the sums.

*Ruling:* the hold side pays one buy fee and no sell fee (a held coin is
not sold); the bot's profit from the app is already after its fees.

## 4. Commands

```
tb bots add --symbol BTCUSDT --invest 500 --start "2026-10-14 09:30"
            [--utc] [--name N] [--low P --high P] [--grids N]
tb bots update ID --profit 12.34 [--at "YYYY-MM-DD HH:MM"] [--utc]
tb bots stop   ID --profit 15.00 [--at ...] [--utc]
tb bots list
tb bots show ID
tb bots undo   ID                (removes the bot's latest update; if it
                                  was the stop update, the bot runs again)
tb bots delete ID --yes          (removes the bot and its updates)
```

Invalid input → exit 1 with the reason, nothing stored.

*Ruling (builder check):* `undo` exists so a mistyped update or stop can be
fixed without re-typing a bot's whole history; the rules of §2 then make
the corrected entry possible again.

## 5. Storage

Migration `008_bots.sql`: tables `bots` and `bot_updates` (`bot_id`
references `bots` with ON DELETE CASCADE; CHECKs for the rules that fit a
row). Comparisons are computed when shown, never stored (the price data is
the source of truth).

## 6. Dashboard

Tab **Bots**: one row per bot at its latest update (name, coin, status,
started, invested, bot profit %, hold profit %, difference, verdict,
last update time), a totals row, and the empty state "no bots yet (tb bots
add)". Endpoint `/api/bots`. Read-only; entry stays on the command line.

## 7. Architecture

New package `bots/`:

| Module | Responsibility | DB |
|---|---|---|
| `bots/store.py` | Bots and updates: add, update, stop, delete, read; validation | yes |
| `bots/compare.py` | Pure: bot + update + two prices + fee → comparison; totals | no |
| `bots/prices.py` | Price at a time from `candles_1m` | yes (read) |
| `bots/report.py` | Text for `list` and `show` | no |
| `bots/times.py` | Parse typed times (local or UTC) | no |

## 8. Testing

- Compare (pure): hand numbers for ahead / behind / level; fee applied
  once; loss bigger than hold's; totals skip unavailable bots.
- Times: local parsing with a fixed UTC+3 zone, `--utc`, bad formats.
- Store (db): add/update/stop/delete round trip; rules rejected (update
  before start, after stop, future time, profit ≤ −invest, bad range).
- Prices (db): exact minute, 9 minutes old, 11 minutes old → unavailable,
  unknown coin.
- CLI (db): full flow add → update → list/show → stop; error exits; no
  advice words.
- Dashboard: `/api/bots` empty and filled; the browser test opens the tab.
