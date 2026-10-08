# Design: AI Trading Buddy — Sub-project 3.5: Minimal Read-only Dashboard

Date: 2026-10-08
Status: Approved for planning
Milestone: 3.5 of the programme (data-foundation spec §2, user-approved 2026-09-23)
Source: vision document §85, §87 (partial)

## 1. Purpose

Let the user **see** the data and the models instead of reading text
reports: candle charts with a few features, the evaluation results of
sub-project 3, and the health of the data. Deliberately small and plain;
sub-project 9 builds the real dashboard.

Read-only: nothing on the page changes data, trains, fetches from Binance or
trades. Nothing on it recommends buy or sell.

### Success criteria

1. `tb dashboard` starts a local web server on `127.0.0.1` (default port
   8050) and prints the address; opening it in a browser shows three tabs:
   **Chart**, **Models** and **Data health**.
2. **Chart:** candlesticks for a chosen symbol and timeframe (5m … 1d) with
   EMA 200 on the price, RSI 14 in a pane below, and the newest bar's trend
   and volatility regime; times are labelled UTC.
3. **Models:** the list of stored runs; for a chosen run, the report's model
   table, skill per test period as a bar chart, a calibration chart per
   model, and on request the size-versus-direction split.
4. **Data health:** per symbol the newest 1m candle and its age (STALE as in
   `tb status`), the newest feature bar per timeframe, and the latest
   quality verdict.
5. Works offline: no file is loaded from the internet.

### Agreed with the user (2026-10-08)

- Approach A: a small Python web server (FastAPI) plus one plain HTML page;
  candle charts with TradingView Lightweight Charts.

### Non-goals

- Logins, users, remote access (it binds to 127.0.0.1 only; a VPS setup
  would need its own design).
- Anything that writes: no buttons that build features, train or backfill.
- Live streaming updates; the page has a Refresh button.
- Predictions on the chart (`tb predict` stays the place for them, with its
  skill and STALE lines).
- Mobile layout, theming, polish (sub-project 9).

## 2. Architecture

A new top-level package `dashboard/`:

| File | Responsibility | Touches DB |
|---|---|---|
| `dashboard/queries.py` | All reads: symbols, candles with features, health, runs | yes (read) |
| `dashboard/svg.py` | Server-side SVG for the calibration chart and the per-period skill bars | no |
| `dashboard/app.py` | `create_app(connect)`: FastAPI routes; validates parameters; returns JSON or SVG | via queries |
| `dashboard/static/index.html`, `app.js`, `style.css` | The page; fetches the JSON API | no |
| `dashboard/static/vendor/lightweight-charts.standalone.production.js` | TradingView Lightweight Charts v5.2.1 (Apache-2.0), with its `LICENSE` beside it | no |

`create_app` takes the connection factory as an argument, so tests pass one
pointing at the test database. Each request opens and closes its own
connection (one user, local). The CLI gains `tb dashboard [--port N]`,
which runs uvicorn on `127.0.0.1` only. `fastapi` and `uvicorn` become
runtime dependencies; `dashboard*` joins the package list and the static
files are package data.

*Ruling:* the calibration and skill charts are SVG drawn in Python rather
than with a second JavaScript chart library: they are simple, need no other
download, and are unit-testable. Lightweight Charts is time-axis only, so it
draws just the candle tab.

## 3. API (JSON unless stated)

| Route | Returns |
|---|---|
| `GET /` | the page |
| `GET /api/symbols` | configured symbols and the five timeframes |
| `GET /api/candles?symbol=S&timeframe=TF&bars=N` | the newest N closed bars (default 500, 50 ≤ N ≤ 5,000): time (unix seconds, bar open), OHLC, volume, `ema_200`, `rsi_14` (null where missing); plus the newest row's `trend_regime`, `volatility_regime`, `adx_14`, `vol_pct_365d` and its bar close time |
| `GET /api/health` | per symbol: newest 1m candle, age in minutes, `stale` (age > 5 min, as `tb status`), newest feature bar per timeframe, latest quality verdict and its time |
| `GET /api/runs` | `ml.store.list_runs` |
| `GET /api/runs/{id}` | `ml.store.load_run` plus `report` (the `render_report` text) |
| `GET /api/runs/{id}/calibration/{model}.svg` | SVG (image/svg+xml) |
| `GET /api/runs/{id}/skill/{model}.svg` | SVG |
| `GET /api/runs/{id}/diagnose` | `ml.diagnose.split_skill` of the stored predictions (slow: loads ~1.2 M rows; the page asks before calling it) |

Errors: an unknown symbol, timeframe, run or model → 404 with a short
message; `bars` out of range → 422 (FastAPI validation). Symbols are checked
against `get_settings().symbols` and timeframes against
`FEATURE_TIMEFRAMES` before any SQL, since timeframes pick table names.
A symbol with no candles returns an empty list, not an error.

Candles come from `candles_<tf>` (cast to float8, as `features/frame.py`)
joined on `open_time` to `features_<tf>`; only bars before the build cutoff
(`features.frame.build_cutoff`) are returned, so the forming bar never
shows.

## 4. The page

One HTML page, three tabs, plain CSS, no framework.

- **Chart:** symbol and timeframe selectors, a Refresh button, a line
  "newest bar closed … UTC (age …) · trend strong_bearish (ADX 30.4) ·
  volatility normal (pct 23)"; the candle chart with EMA 200 as a line, a
  second chart below with RSI 14 and guide lines at 30 and 70, both with
  synchronised time scales; "Times are UTC" under the chart.
- **Models:** the runs table (id, kind, horizon, time, xgb skill); clicking
  a run shows the report text in a `<pre>`, then for each model its
  skill-per-period SVG and calibration SVG; a "Compute size vs direction
  (may take a minute)" button fetches `/diagnose` and shows its table.
- **Data health:** a table per symbol; STALE cells in red.

A footer on every tab: "Read-only. Numbers describe the past; nothing here
is advice."

## 5. SVG charts

- **Calibration** (per run and model): reliability bins pooled over the
  three classes (already in the run's metrics): x = mean predicted
  probability, y = observed frequency, both 0–1; the diagonal as a dashed
  line; one dot per non-empty bin with area by count; title with the ECE.
- **Skill per period:** one bar per test period (positive up, negative
  down), labelled `YYYY-MM`, a zero line, title with pooled skill and CI.

Both are fixed-size (480 × 320), use `currentColor`-free explicit colours
readable on white, and escape every text they embed.

## 6. Testing

- `svg.py` (unit): valid XML (`xml.etree` parses it); calibration has one
  dot per non-empty bin and the diagonal; skill bars count equals periods and
  negative bars sit below the zero line; text is escaped.
- `app.py` with a fake connection factory where possible, and `-m db` tests
  against the test database: every route's status and shape; unknown
  symbol / timeframe / run / model → 404; `bars` limits → 422; the forming
  bar is excluded; health STALE logic; the page and its static files are
  served; the vendored library is served from `/static/` (no external URL in
  `index.html`, asserted).
- No advice words in `index.html`, `app.js` and the JSON labels.
- One browser smoke test in the cloud session (Playwright, Chromium):
  start the server on the test database, open each tab, assert no console
  errors and that a chart canvas exists. Marked `slow`; skipped when
  Chromium is missing.
- Real-data check on the PC: `tb dashboard`, open the three tabs for
  BTCUSDT 1h and run 4; the user compares the candle chart with Binance.
