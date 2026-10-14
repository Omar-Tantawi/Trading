"""A bot against holding the coin (spec section 3). Pure.

Holding = buying the coin with the same USDT at the bot's start (one buy
fee) and keeping it. The bot's Total profit from the app is already after
its own fees.
"""
from dataclasses import asdict, dataclass

LEVEL_USDT = 0.01
LEVEL_SHARE = 0.0001     # 0.01 pts of the investment


def _verdict(diff: float, invest: float) -> str:
    if abs(diff) < max(LEVEL_USDT, LEVEL_SHARE * invest):
        return "level"
    return "ahead" if diff > 0 else "behind"


@dataclass
class Comparison:
    bot_profit: float
    bot_pct: float
    hold_profit: float
    hold_pct: float
    diff: float          # bot minus hold, USDT
    diff_pts: float      # the same as a share of the investment
    verdict: str         # ahead / behind / level
    p_start: float
    p_end: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare(invest: float, fee: float, profit: float, p_start: float,
            p_end: float) -> Comparison:
    hold = invest * (1 - fee) * p_end / p_start - invest
    diff = profit - hold
    return Comparison(bot_profit=profit, bot_pct=profit / invest,
                      hold_profit=hold, hold_pct=hold / invest, diff=diff,
                      diff_pts=diff / invest, verdict=_verdict(diff, invest),
                      p_start=p_start, p_end=p_end)


@dataclass
class Totals:
    n_bots: int
    n_compared: int
    invested: float
    bot_profit: float
    hold_profit: float
    diff: float
    bot_pct: float | None
    hold_pct: float | None
    diff_pts: float | None
    verdict: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def totals(rows: list[tuple[float, Comparison | None]]) -> Totals:
    """Sums over the bots that have a comparison; the others are counted."""
    done = [(inv, c) for inv, c in rows if c is not None]
    invested = sum(inv for inv, _ in done)
    bot = sum(c.bot_profit for _, c in done)
    hold = sum(c.hold_profit for _, c in done)
    diff = bot - hold
    share = (lambda v: v / invested) if invested else (lambda v: None)
    return Totals(n_bots=len(rows), n_compared=len(done), invested=invested,
                  bot_profit=bot, hold_profit=hold, diff=diff,
                  bot_pct=share(bot), hold_pct=share(hold), diff_pts=share(diff),
                  verdict=_verdict(diff, invested) if done else None)
