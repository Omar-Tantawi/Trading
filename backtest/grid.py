"""One day of a Binance-style arithmetic spot grid (spec section 2).

Pure NumPy. Grid j lies between levels L_j and L_{j+1}: holding USDT it
buys q_j coin when the price touches L_j; holding coin it sells q_j when the
price touches L_{j+1}. Limit fills happen at the level price.
"""
from dataclasses import dataclass

import numpy as np

RANGE_K = 3.0
N_GRIDS = 10
CASH_SHARE = 0.995          # keeps a fee cushion so USDT never goes negative


@dataclass
class DayResult:
    cash: float
    coin: float
    round_trips: int
    fees: float
    fills: int


def levels(p0: float, atr_pct: float, n: int = N_GRIDS) -> np.ndarray:
    """n + 1 equally spaced levels, ±RANGE_K usual 24 h moves around p0."""
    w = RANGE_K * atr_pct * np.sqrt(24)
    return np.linspace(p0 * (1 - w), p0 * (1 + w), n + 1)


def path(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Four prices per minute: open, low, high, close for a minute that
    closed up (or flat); open, high, low, close otherwise."""
    up = c >= o
    mid1 = np.where(up, l, h)
    mid2 = np.where(up, h, l)
    return np.column_stack([o, mid1, mid2, c]).ravel()


def run_day(prices: np.ndarray, lv: np.ndarray, *, cash: float, coin: float,
            fee: float, slippage: float) -> DayResult:
    """Rebalance at prices[0] to the grid's starting coin, then fill orders
    along `prices`. Returns the end-of-day USDT, coin and counts."""
    p0 = float(prices[0])
    n = len(lv) - 1
    lower, upper = lv[:-1], lv[1:]
    equity = cash + coin * p0
    q = CASH_SHARE * equity / n / lower
    holds = lower >= p0                       # grids entirely above the price
    fees, fills = 0.0, 0

    delta = float(q[holds].sum()) - coin
    if delta > 1e-12:
        price = p0 * (1 + slippage)
        cash -= delta * price * (1 + fee)
        fees += delta * price * fee
        fills += 1
    elif delta < -1e-12:
        price = p0 * (1 - slippage)
        cash += -delta * price * (1 - fee)
        fees += -delta * price * fee
        fills += 1

    round_trips = 0
    coin_end = 0.0
    x = np.asarray(prices, dtype=float)
    idx = np.arange(len(x) + 1)
    for j in range(n):
        # 1 = holding coin, 0 = holding USDT, -1 = no change at this point
        mark = np.where(x <= lower[j], 1, np.where(x >= upper[j], 0, -1))
        marks = np.concatenate([[int(holds[j])], mark])
        last = np.maximum.accumulate(np.where(marks >= 0, idx, 0))
        state = marks[last]
        steps = np.diff(state)
        buys = int((steps == 1).sum())
        sells = int((steps == -1).sum())
        if buys or sells:
            cost = q[j] * lower[j]
            gain = q[j] * upper[j]
            cash += -buys * cost * (1 + fee) + sells * gain * (1 - fee)
            fees += (buys * cost + sells * gain) * fee
            fills += buys + sells
            first_is_sell = steps[steps != 0][0] == -1
            round_trips += sells - int(first_is_sell)
        if state[-1] == 1:
            coin_end += q[j]
    return DayResult(cash=cash, coin=coin_end, round_trips=round_trips,
                     fees=fees, fills=fills)
