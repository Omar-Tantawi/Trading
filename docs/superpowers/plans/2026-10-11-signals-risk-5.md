# Grid Conditions and Risk Settings (Sub-project 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tb signals` states per coin whether grid conditions look normal or call for WAIT, with reasons, the suggested range and the capital limit from the user's risk settings (`tb risk show/set`); signals are stored and shown on a dashboard tab.

**Architecture:** New `signals/` package: `settings.py` (settings, DB), pure `engine.py`, `collect.py` (DB + saved `vol3` model), `store.py`, `render.py`; migration 007; CLI; dashboard endpoint and tab.

**Tech Stack:** Python 3.13, psycopg, Typer, FastAPI.

**Spec:** `docs/superpowers/specs/2026-10-11-signals-risk-5-design.md`

## Global Constraints

- Model: `vol3` `xgb_v1` horizon 24; P(wild) = class index 2.
- WAIT when P(wild) ≥ `wild_threshold` (equality is WAIT).
- Range: `backtest.grid.levels(close, atr_pct)` (±3 · atr_pct · √24, 10 grids); flag via `step_too_small(lv, fee)`.
- Stale rule: `ml.predict.is_stale` (older than 2 h 5 min).
- No advice words (buy, sell, long, short, enter, exit) in any output.
- `signals*` added to the pyproject package list.

## Review Focus

1. **`--no-build`** cannot know the quality result → the signal says "data quality not checked (--no-build)" rather than claiming PASS → test in Task 2.
2. **A coin the model was not trained on** → treated as "no forecast model" → test in Task 3.
3. **Several gates at once** list every reason → test in Task 2.
4. **`tb risk set` with bad values** → exit non-zero, settings unchanged → test in Task 4.
5. **Dashboard with no signals yet** → empty list and the default settings, no error → test in Task 5.

---

### Task 1: Migration and settings
**Files:** `data/storage/migrations/007_signals.sql`, `signals/__init__.py`, `signals/settings.py`, `tests/conftest.py` (delete list), `pyproject.toml`.
**Produces:** `RiskSettings(profile, wild_threshold, capital_usdt, max_coin_share, fee)`; `validate(s) -> None` (ValueError); `load_settings(conn)`; `save_settings(conn, s)`.
- [ ] Tests: defaults; each bad value raises; DB round trip; no row → defaults.
- [ ] Implement; commit.

### Task 2: Engine and render (pure)
**Files:** `signals/engine.py`, `signals/render.py`.
**Produces:** `Inputs(symbol, has_data, quality_fail: bool | None, bar_open_time, age, close, atr_pct, p_wild, model_problem)`; `Signal` (status, reasons, notes, range, step, capital limit, settings); `evaluate(inputs, settings) -> Signal`; `render(signals, settings) -> str`.
- [ ] Tests: each gate alone → WAIT with its reason; several gates → all reasons; P = threshold → WAIT; below → NORMAL; range/step maths; step flag; capital limit with/without capital; `--no-build` note; profile notes; no advice words.
- [ ] Implement; commit.

### Task 3: Collect and store (DB)
**Files:** `signals/collect.py`, `signals/store.py`.
**Produces:** `collect(conn, symbol, quality_fail, root, now) -> Inputs`; `save_signal(conn, sig) -> int`; `latest_signals(conn) -> list[dict]`.
- [ ] Tests: no candles → has_data False; features without model → model problem; trained vol3 model → a P(wild); symbol not trained → model problem; store round trip; newest per coin.
- [ ] Implement; commit.

### Task 4: CLI
**Files:** `data/cli.py` (`tb signals`, `risk_app` with `show`/`set`).
- [ ] Tests: `tb signals` with no data → WAIT "no market data", stored; with model → P(wild) line; `tb risk set` stores, rejects bad values; outputs without advice words.
- [ ] Implement; commit.

### Task 5: Dashboard, docs
**Files:** `dashboard/app.py` (`/api/signals`), `dashboard/static/index.html`, `app.js`, README, handoff.
- [ ] Tests: `/api/signals` empty and with rows (newest per coin); browser test opens the tab.
- [ ] Implement; full suite; review; commit; push.
