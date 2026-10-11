"""The grid backtest report (spec section 5): measurement, never advice."""
import pandas as pd

from backtest.engine import START, STRATEGIES, StrategyResult
from backtest.grid import N_GRIDS, RANGE_K
from backtest.metrics import by_year, diff_ci, summary

COMPARISONS = (("grid_vol3_30", "grid_always"), ("grid_random_30", "grid_always"),
               ("grid_vol3_30", "grid_random_30"),
               ("grid_keep_vol3_30", "grid_always"),
               ("grid_keep_random_30", "grid_always"),
               ("grid_keep_vol3_30", "grid_keep_random_30"))


def daily_returns(r: StrategyResult) -> pd.Series:
    prev = r.equity.shift(1).fillna(START)
    return r.equity / prev - 1


def _verdict(name: str, other: str, m: float, lo: float, hi: float) -> str:
    head = f"{name} vs {other}: {m:+.1f} bp/day (95% CI {lo:+.1f} to {hi:+.1f})"
    if lo > 0:
        return f"{head}: higher mean daily return than {other}"
    if hi < 0:
        return f"{head}: lower mean daily return than {other}"
    return f"{head}: not distinguishable from {other}"


def _pooled(coins: dict[str, dict[str, StrategyResult]], name: str) -> StrategyResult:
    """Equal-weight portfolio of the coins: mean daily return, compounded."""
    rets = pd.concat([daily_returns(r[name]) for r in coins.values()], axis=1)
    active = pd.concat([r[name].active for r in coins.values()], axis=1).fillna(False)
    ret = rets.mean(axis=1)
    return StrategyResult(
        equity=START * (1 + ret).cumprod(), active=active.any(axis=1).astype(bool),
        round_trips=sum(r[name].round_trips for r in coins.values()),
        fees=sum(r[name].fees for r in coins.values()) / len(coins), skipped_days=0)


def _table(results: dict[str, StrategyResult]) -> list[str]:
    lines = [f"  {'strategy':<20}{'total':>9}{'annual':>9}{'max DD':>9}{'Sharpe':>8}"
             f"{'Sortino':>8}{'active':>8}{'trips':>8}{'fees':>8}{'win':>7}{'streak':>7}"]
    for name in STRATEGIES:
        r = results[name]
        s = summary(r.equity, r.active, r.round_trips, r.fees, START)
        lines.append(
            f"  {name:<20}{s['total_return']:>+9.1%}{s['annual_return']:>+9.1%}"
            f"{s['max_drawdown']:>+9.1%}{s['sharpe']:>8.2f}{s['sortino']:>8.2f}"
            f"{s['active_days']:>8}{s['round_trips']:>8}{s['fees_pct']:>8.1%}"
            f"{s['win_share']:>7.0%}{s['longest_losing_streak']:>7}")
    return lines


def _per_year(label: str, results: dict[str, StrategyResult]) -> list[str]:
    years = {n: by_year(results[n].equity, START) for n in STRATEGIES}
    cols = list(years["cash"])
    if not cols:
        return []
    last = years["cash"][cols[-1]]["last_day"]
    heads = [str(y) for y in cols]
    if (last.month, last.day) != (12, 31):
        heads[-1] = f"{cols[-1]} (to {last:%b} {last.day})"
    widths = [max(9, len(h) + 2) for h in heads]
    lines = []
    for key, title in (("return", "return"),
                       ("max_drawdown", "worst drop within the year")):
        lines.append(f"Per year, {label} ({title}):")
        lines.append(f"  {'strategy':<20}" + "".join(
            f"{h:>{w}}" for h, w in zip(heads, widths)))
        for n in STRATEGIES:
            lines.append(f"  {n:<20}" + "".join(
                f"{years[n][y][key]:>+{w}.1%}" for y, w in zip(cols, widths)))
        lines.append("")
    return lines


def render(run_id: int, fee: float, slippage: float,
           coins: dict[str, dict[str, StrategyResult]]) -> str:
    lines = [f"Spot grid backtest: run {run_id} (vol3 forecasts, next 24h), "
             f"fee {fee:.2%} per fill, slippage {slippage:.2%} on market orders",
             f"Grid: {N_GRIDS} grids over +/-{RANGE_K:g} usual 24h moves, rebuilt daily "
             "at 00:00 UTC; start 1,000 USDT per coin. WAIT days: grid_vol3/random "
             "hold USDT (coin sold at the open); grid_keep_* cancel the orders and "
             "keep the coin.", ""]
    for symbol, results in coins.items():
        days = results["cash"].equity.index
        if not len(days):
            lines += [f"{symbol}: no test days", ""]
            continue
        lines.append(f"{symbol}: {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}, "
                     f"{len(days)} days ({results['cash'].skipped_days} without full data)")
        lines += _table(results)
        for a, b in COMPARISONS:
            lines.append("  " + _verdict(a, b, *diff_ci(daily_returns(results[a]),
                                                        daily_returns(results[b]))))
        lines.append("")
    with_days = {s: r for s, r in coins.items() if len(r["cash"].equity)}
    if len(with_days) > 1:
        lines.append("All coins, equal weight (mean daily return of the coins, "
                     "compounded; trips summed, fees averaged):")
        lines += _table({n: _pooled(with_days, n) for n in STRATEGIES})
        coins = with_days
        for a, b in COMPARISONS:
            ra = pd.concat([daily_returns(r[a]) for r in coins.values()], axis=1).mean(axis=1)
            rb = pd.concat([daily_returns(r[b]) for r in coins.values()], axis=1).mean(axis=1)
            lines.append("  " + _verdict(a, b, *diff_ci(ra, rb)))
        lines.append("")
        lines += _per_year("all coins", {n: _pooled(with_days, n) for n in STRATEGIES})
    elif len(with_days) == 1:
        symbol, results = next(iter(with_days.items()))
        lines += _per_year(symbol, results)
    lines.append("Notes: a day is simulated only when its full 1440 minutes, its "
                 "atr_pct and its forecast are stored; grids hold USDT on other days "
                 "(the hold strategy does not). The random pause days are drawn over "
                 "the whole period. Limit orders fill when the price touches their "
                 "level.")
    lines.append("Past simulated results; real fills, fees and outages differ. Not advice.")
    return "\n".join(lines)
