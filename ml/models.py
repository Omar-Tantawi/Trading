"""Machine-learning models and calibration (spec sections 5.2 and 5.3).

Every model has the same interface: fit(X_fit, y_fit, X_cal, y_cal) and
predict_proba(X) -> n x 3 (down, flat, up). After fitting on the fit rows, a
model fits one temperature on the calibration rows; predict_proba applies it.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

N_CLASSES = 3
_FLOOR = 1e-6

# v2: stops after 100 iterations instead of converging (up to 1000). On
# features of this size full convergence took ~250 iterations and ~40 s per
# fit without a better test score; stopping early also regularises.
LOGREG_PARAMS = {"C": 0.1, "max_iter": 100}
XGB_PARAMS = {
    "objective": "multi:softprob",
    "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 50, "tree_method": "hist",
    "seed": 0, "nthread": os.cpu_count() or 4, "verbosity": 0,
}
XGB_ROUNDS = 300


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _FLOOR, None)
    return p / p.sum(axis=1, keepdims=True)


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T): T > 1 softens, T < 1 sharpens."""
    z = np.log(np.clip(p, 1e-15, 1.0)) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """The T in [0.05, 20] that minimises log loss on (p, y)."""
    if len(y) == 0:
        return 1.0
    rows = np.arange(len(y))

    def loss(log_t):
        q = apply_temperature(p, float(np.exp(log_t)))
        return -np.log(np.clip(q[rows, y], 1e-15, 1.0)).mean()

    res = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)),
                          method="bounded")
    return float(np.exp(res.x))


class Model:
    name = "model"
    calibrate = True
    n_classes = N_CLASSES   # models pickled before 3b have no instance value

    def __init__(self, n_classes: int = N_CLASSES):
        self.n_classes = n_classes
        self.temperature = 1.0

    def fit(self, X_fit: pd.DataFrame, y_fit: np.ndarray,
            X_cal: pd.DataFrame, y_cal: np.ndarray) -> "Model":
        self._fit(X_fit, np.asarray(y_fit, dtype=int))
        self.temperature = 1.0
        if self.calibrate and len(y_cal):
            self.temperature = fit_temperature(self._raw_proba(X_cal),
                                               np.asarray(y_cal, dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return apply_temperature(self._raw_proba(X), self.temperature)

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class LogReg(Model):
    name = "logreg_v2"

    def _fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.columns = list(X.columns)
        median = X.median()
        self.fill = median.where(median.notna(), 0.0)
        Z = X.fillna(self.fill).to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(Z)
        self.single = None
        if len(np.unique(y)) < 2:
            # Nothing to separate: fall back to the smoothed class shares.
            counts = np.bincount(y, minlength=self.n_classes) + 1.0
            self.single = counts / counts.sum()
            return
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            # Stopping at max_iter is deliberate (see LOGREG_PARAMS).
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model = LogisticRegression(**LOGREG_PARAMS).fit(
                self.scaler.transform(Z), y)

    def _raw_proba(self, X):
        if self.single is not None:
            return np.tile(self.single, (len(X), 1))
        Z = X[self.columns].fillna(self.fill).to_numpy(dtype=float)
        p = np.zeros((len(X), self.n_classes))
        p[:, self.model.classes_] = self.model.predict_proba(self.scaler.transform(Z))
        return _normalise(p)


class XGB(Model):
    name = "xgb_v1"

    def _fit(self, X, y):
        import xgboost as xgb

        self.columns = list(X.columns)
        data = xgb.DMatrix(X.to_numpy(dtype=float), label=y)
        params = {**XGB_PARAMS, "num_class": self.n_classes}
        self.booster = xgb.train(params, data, num_boost_round=XGB_ROUNDS)

    def _raw_proba(self, X):
        import xgboost as xgb

        data = xgb.DMatrix(X[self.columns].to_numpy(dtype=float))
        return _normalise(self.booster.predict(data).reshape(len(X), self.n_classes))


def make_models(n_classes: int = N_CLASSES) -> list[Model]:
    """Fresh, unfitted models in report order, for `n_classes` classes."""
    from ml.baselines import BaseRate, EmaCross, MacdSign, RsiZone

    return [cls(n_classes) for cls in
            (BaseRate, EmaCross, RsiZone, MacdSign, LogReg, XGB)]
