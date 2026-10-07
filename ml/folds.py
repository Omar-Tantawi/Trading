"""Walk-forward folds on decision time (spec sections 4.2 and 4.3).

Pure: no database, no clock. All boundaries are on tau, the time a row's
features are known. A row's label is known H hours after tau, so a training
row is kept only when its label is known before the next slice starts (the
purge). Nothing at or after the holdout boundary appears in a walk-forward
fold.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldConfig:
    first_test: datetime
    fold_months: int
    holdout_start: datetime
    cal_fraction: float


DEFAULT_FOLDS = FoldConfig(
    first_test=datetime(2020, 1, 1, tzinfo=timezone.utc),
    fold_months=6,
    holdout_start=datetime(2025, 10, 1, tzinfo=timezone.utc),
    cal_fraction=0.2,
)


@dataclass
class Fold:
    number: int
    test_start: datetime
    test_end: datetime | None   # None: open-ended (the holdout)
    fit_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray


def _split_train(tau: pd.DatetimeIndex, candidates: np.ndarray, horizon: int,
                 cal_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit and calibration positions out of `candidates`, the later
    `cal_fraction` by tau for calibration, with the purge between them."""
    if len(candidates) == 0:
        return candidates, candidates
    t = tau[candidates]
    boundary = t.sort_values()[int(len(t) * (1 - cal_fraction))
                               if cal_fraction > 0 else len(t) - 1]
    gap = pd.Timedelta(hours=horizon)
    fit = candidates[(t + gap <= boundary)]
    cal = candidates[t >= boundary] if cal_fraction > 0 else candidates[:0]
    return fit, cal


def final_split(tau: pd.DatetimeIndex, horizon: int,
                cal_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit and calibration positions over all rows (for `tb ml train`)."""
    return _split_train(tau, np.arange(len(tau)), horizon, cal_fraction)


def _fold(number, tau, horizon, cfg, start, end) -> Fold | None:
    start_ts = pd.Timestamp(start)
    in_test = tau >= start_ts
    if end is not None:
        in_test &= tau < pd.Timestamp(end)
    test = np.flatnonzero(in_test)
    candidates = np.flatnonzero(tau + pd.Timedelta(hours=horizon) <= start_ts)
    fit, cal = _split_train(tau, candidates, horizon, cfg.cal_fraction)
    if len(test) == 0 or len(fit) == 0:
        return None
    return Fold(number, start, end, fit, cal, test)


def walk_forward_folds(tau: pd.DatetimeIndex, horizon: int,
                       cfg: FoldConfig) -> list[Fold]:
    """Expanding-window folds with fold_months test windows from first_test
    up to holdout_start. Windows without test or fit rows are skipped."""
    folds: list[Fold] = []
    start = pd.Timestamp(cfg.first_test)
    holdout = pd.Timestamp(cfg.holdout_start)
    while start < holdout:
        end = min(start + pd.DateOffset(months=cfg.fold_months), holdout)
        fold = _fold(len(folds), tau, horizon, cfg,
                     start.to_pydatetime(), end.to_pydatetime())
        if fold is not None:
            folds.append(fold)
        start = end
    return folds


def holdout_fold(tau: pd.DatetimeIndex, horizon: int,
                 cfg: FoldConfig) -> Fold | None:
    """Train on everything before holdout_start, test on the rest."""
    return _fold(0, tau, horizon, cfg, cfg.holdout_start, None)
