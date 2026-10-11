"""Financial metrics of a daily equity series (spec section 5)."""
import numpy as np
import pandas as pd


def _longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def summary(equity: pd.Series, active: pd.Series, round_trips: int,
            fees: float, start: float) -> dict:
    """Metrics of a daily equity series that starts from `start`."""
    values = equity.to_numpy(dtype=float)
    prev = np.concatenate([[start], values[:-1]])
    ret = values / prev - 1
    days = len(values)
    total = values[-1] / start - 1 if days else 0.0
    peak = np.maximum.accumulate(np.concatenate([[start], values]))
    dd = (np.concatenate([[start], values]) / peak - 1).min() if days else 0.0
    sd = ret.std(ddof=1) if days > 1 else 0.0
    down = ret[ret < 0]
    sd_down = np.sqrt((down ** 2).mean()) if len(down) else 0.0
    act = active.to_numpy(dtype=bool)
    act_ret = ret[act]
    return {
        "days": days,
        "total_return": float(total),
        "annual_return": float((1 + total) ** (365 / days) - 1) if days else 0.0,
        "max_drawdown": float(dd),
        "sharpe": float(ret.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        "sortino": float(ret.mean() / sd_down * np.sqrt(365)) if sd_down > 0 else 0.0,
        "active_days": int(act.sum()),
        "round_trips": int(round_trips),
        "fees_pct": float(fees / start),
        "win_share": float((act_ret > 0).mean()) if len(act_ret) else 0.0,
        "longest_losing_streak": _longest_run(act_ret < 0),
    }


def diff_ci(ret_a: pd.Series, ret_b: pd.Series, n: int = 1000,
            seed: int = 0) -> tuple[float, float, float]:
    """Mean daily return of a minus b in basis points, with a 95 % interval
    from resampling calendar weeks."""
    d = (ret_a - ret_b).dropna()
    week = d.index.tz_convert(None).to_period("W").asi8
    groups, inv = np.unique(week, return_inverse=True)
    sums = np.bincount(inv, d.to_numpy(), len(groups))
    counts = np.bincount(inv, minlength=len(groups)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(n, len(groups)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean() * 1e4), float(lo * 1e4), float(hi * 1e4)
