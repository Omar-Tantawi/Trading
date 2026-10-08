"""Scoring probability forecasts (spec section 6).

Pure NumPy. `p` is an n x 3 array of probabilities (down, flat, up) and `y`
the true classes 0, 1, 2.
"""
import numpy as np
import pandas as pd

_EPS = 1e-15


def row_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(row_log_loss(p, y).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(axis=1) == y).mean())


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Predicted probability against observed frequency, pooled over the
    three classes, in equal-width bins."""
    probs = p.ravel()
    hits = np.eye(p.shape[1])[y].ravel()
    which = np.minimum((probs * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        mask = which == b
        n = int(mask.sum())
        out.append({
            "lo": b / bins, "hi": (b + 1) / bins, "n": n,
            "mean_p": float(probs[mask].mean()) if n else None,
            "freq": float(hits[mask].mean()) if n else None,
        })
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: n-weighted mean |mean_p - freq|."""
    rows = reliability(p, y, bins)
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_p"] - r["freq"])
                     for r in rows if r["n"]) / total)


def skill(ll_model: float, ll_base: float) -> float:
    """Log-loss skill against the base rate: positive is better."""
    return 1.0 - ll_model / ll_base


def bootstrap_skill_ci(p_model: np.ndarray, p_base: np.ndarray, y: np.ndarray,
                       tau: pd.DatetimeIndex, n: int = 1000,
                       seed: int = 0) -> tuple[float, float]:
    """95 % interval for skill, resampling calendar weeks of tau with
    replacement so that autocorrelated (overlapping) labels stay together."""
    week = pd.DatetimeIndex(tau).tz_convert(None).to_period("W").asi8
    groups, inverse = np.unique(week, return_inverse=True)
    model_sum = np.bincount(inverse, row_log_loss(p_model, y), len(groups))
    base_sum = np.bincount(inverse, row_log_loss(p_base, y), len(groups))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(n, len(groups)))
    skills = 1.0 - model_sum[pick].sum(axis=1) / base_sum[pick].sum(axis=1)
    lo, hi = np.percentile(skills, [2.5, 97.5])
    return float(lo), float(hi)
