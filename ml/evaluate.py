"""Walk-forward evaluation of every model for one horizon (spec sections 4.2
and 6). Pure: no database, no clock.

Each fold trains a fresh model on its fit rows, calibrates on its
calibration rows and predicts its test rows. Metrics pool all test rows and
compare each model with base_rate_v1 on exactly the same rows.
"""
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ml.dataset import Dataset
from ml.folds import Fold
from ml.metrics import (
    accuracy, bootstrap_skill_ci, brier, ece, log_loss, reliability, skill,
)
from ml.models import Model, make_models

BASE = "base_rate_v1"
MOVE3 = ("down", "flat", "up")


def _p(n: int) -> list[str]:
    return [f"p{i}" for i in range(n)]


@dataclass
class EvalResult:
    predictions: pd.DataFrame
    metrics: dict


def evaluate(ds: Dataset, horizon: int, folds: list[Fold],
             model_factory: Callable[[], list[Model]] | None = None,
             progress: Callable[[str], None] | None = None,
             classes: tuple[str, ...] = MOVE3) -> EvalResult:
    """`progress`, if given, receives one line per test period with the
    seconds each model took."""
    model_factory = model_factory or (lambda: make_models(len(classes)))
    parts = []
    for i, fold in enumerate(folds, 1):
        timings = []
        X_fit, y_fit = ds.X.iloc[fold.fit_idx], ds.y[fold.fit_idx]
        X_cal, y_cal = ds.X.iloc[fold.cal_idx], ds.y[fold.cal_idx]
        X_test = ds.X.iloc[fold.test_idx]
        for model in model_factory():
            began = time.monotonic()
            model.fit(X_fit, y_fit, X_cal, y_cal)
            p = model.predict_proba(X_test)
            timings.append(f"{model.name} {time.monotonic() - began:.0f}s")
            parts.append(pd.DataFrame({
                "model": model.name,
                "symbol": ds.symbol[fold.test_idx],
                "open_time": ds.open_time[fold.test_idx],
                "fold": fold.number,
                "label": ds.y[fold.test_idx],
                **{f"p{k}": p[:, k] for k in range(len(classes))},
            }))
        if progress:
            progress(f"  next {horizon}h: period {i}/{len(folds)} "
                     f"({fold.test_start:%Y-%m}), {len(fold.fit_idx):,} training "
                     f"rows: " + ", ".join(timings))
    pred = pd.concat(parts, ignore_index=True)
    metrics = summarise(pred, horizon, classes)
    metrics["folds"] = [{
        "number": f.number,
        "test_start": f.test_start.isoformat(),
        "test_end": f.test_end.isoformat() if f.test_end else None,
        "n_fit": len(f.fit_idx), "n_cal": len(f.cal_idx), "n_test": len(f.test_idx),
    } for f in folds]
    return EvalResult(pred, metrics)


def summarise(pred: pd.DataFrame, horizon: int,
              classes: tuple[str, ...] = MOVE3) -> dict:
    """Pooled, per-fold and per-symbol scores of every model (spec 6)."""
    cols = _p(len(classes))
    key = ["symbol", "open_time"]
    base = pred[pred["model"] == BASE].sort_values(key, kind="stable")
    y = base["label"].to_numpy(dtype=int)
    p_base = base[cols].to_numpy()
    tau = pd.DatetimeIndex(base["open_time"]) + pd.Timedelta(hours=1)
    folds_of = base["fold"].to_numpy()
    symbols = base["symbol"].to_numpy()
    starts = {}
    shares = np.bincount(y, minlength=len(classes)) / len(y)

    models = {}
    for name in pred["model"].unique():
        rows = pred[pred["model"] == name].sort_values(key, kind="stable")
        p = rows[cols].to_numpy()
        ll, ll_base = log_loss(p, y), log_loss(p_base, y)
        lo, hi = ((0.0, 0.0) if name == BASE
                  else bootstrap_skill_ci(p, p_base, y, tau))

        def part_skill(mask):
            return skill(log_loss(p[mask], y[mask]), log_loss(p_base[mask], y[mask]))

        per_fold = []
        for f in np.unique(folds_of):
            mask = folds_of == f
            starts.setdefault(int(f), tau[mask].min())
            per_fold.append({"fold": int(f), "test_start": starts[int(f)].isoformat(),
                             "n": int(mask.sum()), "skill": part_skill(mask)})
        models[name] = {
            "log_loss": ll, "brier": brier(p, y), "accuracy": accuracy(p, y),
            "ece": ece(p, y), "skill": 0.0 if name == BASE else skill(ll, ll_base),
            "skill_lo": lo, "skill_hi": hi,
            "reliability": reliability(p, y),
            "per_fold": per_fold,
            "per_symbol": {s: part_skill(symbols == s) for s in np.unique(symbols)},
        }
    return {
        "horizon": horizon,
        "n_test": int(len(y)),
        "class_shares": {c: float(v) for c, v in zip(classes, shares)},
        "models": models,
    }
