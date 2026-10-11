"""Baseline models (spec section 5.1).

The base rate knows nothing about the market; all skill is measured against
it. The rule baselines turn the classic EMA crossover, RSI and MACD rules
into probability models: the frequency of each class in each rule state,
counted on the training rows with +1 smoothing.
"""
import numpy as np
import pandas as pd

from ml.models import N_CLASSES, Model


def _shares(y: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=n).astype(float) + 1.0
    return counts / counts.sum()


class BaseRate(Model):
    name = "base_rate_v1"
    calibrate = False

    def _fit(self, X, y):
        # +1 smoothing, as the rule baselines: a class absent from the
        # training rows never gets probability 0 (which would inflate every
        # other model's skill).
        self.p = _shares(y, self.n_classes)

    def _raw_proba(self, X):
        return np.tile(self.p, (len(X), 1))


class RuleBaseline(Model):
    """P(class | rule state); an unknown or unseen state gets the base rate."""

    def state(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _fit(self, X, y):
        states = self.state(X).to_numpy(dtype=float)
        self.base = _shares(y, self.n_classes)
        self.table = {s: _shares(y[states == s], self.n_classes)
                      for s in np.unique(states[~np.isnan(states)])}

    def _raw_proba(self, X):
        states = self.state(X).to_numpy(dtype=float)
        out = np.tile(self.base, (len(X), 1))
        for s, p in self.table.items():
            out[states == s] = p
        return out


class EmaCross(RuleBaseline):
    name = "ema_cross_v1"

    def state(self, X):
        return np.sign(X["trend_duration"])


class RsiZone(RuleBaseline):
    name = "rsi_v1"

    def state(self, X):
        rsi = X["rsi_14"]
        return pd.Series(np.select([rsi < 30, rsi <= 70, rsi > 70], [0.0, 1.0, 2.0],
                                   default=np.nan), index=X.index)


class MacdSign(RuleBaseline):
    name = "macd_v1"

    def state(self, X):
        return np.sign(X["macd_hist_pct"])
