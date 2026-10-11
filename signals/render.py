"""Text output of `tb signals` and `tb risk show`. Describes; never advises."""
from datetime import datetime, timedelta

from features.summary import _age
from signals.engine import Signal
from signals.settings import RiskSettings

STATUS_WORDS = {"WAIT": "WAIT", "NORMAL": "GRID CONDITIONS NORMAL"}

# From the 2026-10-11 grid backtest (run 10, fee 0.075 %, all coins).
PROFILE_NOTES = {
    "defensive": ("Profile defensive: in the 2026-10-11 backtest (fee 0.075%), "
                  "WAIT days were spent in USDT; that gave the smallest drops "
                  "(2022: -19% vs -45% for a grid that always ran)."),
    "return": ("Profile return: in the 2026-10-11 backtest (fee 0.075%), WAIT "
               "days kept the coin and only cancelled the grid's orders; that "
               "gave a higher return, with drops like a grid that always ran."),
}


def settings_lines(s: RiskSettings) -> list[str]:
    capital = "not set" if s.capital_usdt is None else f"{s.capital_usdt:,.0f} USDT"
    return [f"profile:         {s.profile}",
            f"wild threshold:  {s.wild_threshold:.0%} (WAIT when P(wild) is at or above it)",
            f"capital:         {capital}",
            f"max coin share:  {s.max_coin_share:.0%} of capital per coin",
            f"fee per fill:    {s.fee:.3%}"]


def _capital_line(sig: Signal) -> str:
    share, capital = sig.settings["max_coin_share"], sig.settings["capital_usdt"]
    if capital is None:
        return (f"capital limit: {share:.0%} of your capital "
                "(set it with `tb risk set --capital`)")
    return (f"capital limit: {sig.capital_limit:,.0f} USDT ({share:.0%} "
            f"of {capital:,.0f} USDT)")


def _coin(sig: Signal) -> list[str]:
    lines = [f"{sig.symbol}  {STATUS_WORDS[sig.status]}"]
    lines += [f"  {r}" for r in sig.reasons]
    if sig.bar_open_time is not None:
        closed = sig.bar_open_time + timedelta(hours=1)
        ago = f", {_age(sig.age)}" if sig.age is not None else ""
        lines.append(f"  from the 1h bar that closed {closed:%Y-%m-%d %H:%M} UTC "
                     f"(close {sig.close:,.2f}){ago}")
    if sig.p_wild is not None:
        lines.append(f"  P(wild) next 24h: {sig.p_wild:.0%} (vol3 xgb_v1)")
    if sig.range_low is not None:
        lines.append(f"  suggested range: {sig.range_low:,.2f} to {sig.range_high:,.2f}, "
                     f"{sig.n_grids} grids, step {sig.step_price:,.2f} "
                     f"({sig.step_pct:.2%} at the top)")
    lines.append("  " + _capital_line(sig))
    lines += [f"  note: {n}" for n in sig.notes]
    return lines


def render(signals: list[Signal], s: RiskSettings, now: datetime) -> str:
    lines = [f"Grid conditions at {now:%Y-%m-%d %H:%M} UTC "
             f"(profile {s.profile}, WAIT when P(wild) >= {s.wild_threshold:.0%}, "
             f"fee {s.fee:.3%})", ""]
    for sig in signals:
        lines += _coin(sig) + [""]
    lines.append(PROFILE_NOTES[s.profile])
    lines.append("WAIT means conditions unlike the days the grid ran in the "
                 "backtest. The range is the backtest's grid around the last "
                 "close. The decision is yours. Not advice.")
    return "\n".join(lines)
