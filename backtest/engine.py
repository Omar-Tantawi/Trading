"""Strategies over a coin's test days (spec sections 3 and 4).

Pure: no database. A day runs from 00:00 to 24:00 UTC. A day is tradable
only when its 1440 minutes, its atr_pct and its P(wild) exist; on any other
day every strategy holds USDT (and the day is counted as skipped).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.grid import levels, path, run_day, step_too_small

START = 1000.0
SLIPPAGE = 0.0005
THRESHOLDS = {"grid_vol3_25": 0.25, "grid_vol3_30": 0.30, "grid_vol3_35": 0.35}
STRATEGIES = ("cash", "hold", "grid_always", *THRESHOLDS, "grid_random_30")
_DAY = pd.Timedelta(days=1)


@dataclass
class CoinData:
    minutes: pd.DataFrame     # open/high/low/close on a UTC minute index
    days: pd.DataFrame        # atr_pct, p_wild on the UTC day-start index


@dataclass
class StrategyResult:
    equity: pd.Series         # end-of-day equity in USDT
    active: pd.Series         # True on days the grid (or hold) was running
    round_trips: int
    fees: float
    skipped_days: int


def _calendar(data: CoinData) -> pd.DatetimeIndex:
    first, last = data.days.index.min(), data.days.index.max()
    return pd.date_range(first, last, freq="1D")


def _tradable(data: CoinData, cal: pd.DatetimeIndex, day_minutes: dict) -> pd.Series:
    inputs = data.days.reindex(cal)
    ok = inputs["atr_pct"].notna() & inputs["p_wild"].notna()
    full = pd.Series([len(day_minutes.get(d, ())) == 1440 for d in cal], index=cal)
    return ok & full


def _random_pauses(tradable: pd.Series, p_wild: pd.Series, seed: int = 0) -> set:
    days = tradable.index[tradable]
    k = int((p_wild.reindex(days) >= THRESHOLDS["grid_vol3_30"]).sum())
    rng = np.random.default_rng(seed)
    return set(rng.choice(np.asarray(days), size=k, replace=False)) if k else set()


def run_strategy(data: CoinData, name: str, fee: float,
                 slippage: float = SLIPPAGE) -> StrategyResult:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}")
    if data.days.empty:                       # coin not in the run
        none = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        return StrategyResult(none, none.astype(bool), 0, 0.0, 0)
    cal = _calendar(data)
    by_day = {d: g for d, g in data.minutes.groupby(data.minutes.index.floor("1D"))}
    tradable = _tradable(data, cal, by_day)
    p_wild = data.days["p_wild"]
    paused = _random_pauses(tradable, p_wild) if name == "grid_random_30" else set()

    cash, coin, fees, trips = START, 0.0, 0.0, 0
    last_price = None
    equity, active = [], []
    for day in cal:
        m = by_day.get(day)
        p_open = float(m["open"].iloc[0]) if m is not None and len(m) else last_price
        run = False
        if name == "hold":
            if coin == 0.0 and p_open is not None and cash > 0:
                price = p_open * (1 + slippage)
                coin = cash / (price * (1 + fee))
                fees += coin * price * fee
                cash = 0.0
            run = coin > 0
        elif name != "cash" and tradable[day]:
            if name == "grid_always":
                run = True
            elif name in THRESHOLDS:
                run = p_wild[day] < THRESHOLDS[name]
            else:                                   # grid_random_30
                run = day not in paused
        if name.startswith("grid") and run:
            x = path(*(m[c].to_numpy() for c in ("open", "high", "low", "close")))
            lv = levels(x[0], float(data.days.at[day, "atr_pct"]))
            run = not step_too_small(lv, fee)   # such a grid only pays fees
        if name.startswith("grid"):
            if run:
                r = run_day(x, lv, cash=cash, coin=coin, fee=fee, slippage=slippage)
                cash, coin = r.cash, r.coin
                fees += r.fees
                trips += r.round_trips
            elif coin > 0 and p_open is not None:   # WAIT: leave the market
                price = p_open * (1 - slippage)
                cash += coin * price * (1 - fee)
                fees += coin * price * fee
                coin = 0.0
        if m is not None and len(m):
            last_price = float(m["close"].iloc[-1])
        equity.append(cash + coin * (last_price or 0.0))
        active.append(bool(run))
    return StrategyResult(
        equity=pd.Series(equity, index=cal, name=name),
        active=pd.Series(active, index=cal, name=name),
        round_trips=trips, fees=fees,
        skipped_days=int((~tradable).sum()),
    )
