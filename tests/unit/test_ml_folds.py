"""Walk-forward folds, purging and the holdout (spec sections 4.2-4.3)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ml.folds import DEFAULT_FOLDS, final_split, holdout_fold, walk_forward_folds

HOLDOUT = datetime(2025, 10, 1, tzinfo=timezone.utc)


def _tau(start="2018-01-01", end="2026-01-01"):
    return pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")


def test_window_boundaries():
    folds = walk_forward_folds(_tau(), 1, DEFAULT_FOLDS)
    assert len(folds) == 12
    assert folds[0].test_start == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert folds[-1].test_end == HOLDOUT
    assert [f.number for f in folds] == list(range(12))


def test_no_overlap_and_purge():
    tau = _tau()
    h = timedelta(hours=24)
    for f in walk_forward_folds(tau, 24, DEFAULT_FOLDS):
        assert tau[f.fit_idx].max() + h <= tau[f.cal_idx].min()
        assert tau[f.cal_idx].max() + h <= f.test_start
        assert tau[f.test_idx].min() >= f.test_start
        assert tau[f.test_idx].max() < f.test_end
        sets = [set(f.fit_idx), set(f.cal_idx), set(f.test_idx)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_nothing_from_holdout_in_walk_forward():
    tau = _tau()
    for f in walk_forward_folds(tau, 4, DEFAULT_FOLDS):
        for idx in (f.fit_idx, f.cal_idx, f.test_idx):
            assert (tau[idx] < HOLDOUT).all()


def test_holdout_fold():
    tau = _tau()
    f = holdout_fold(tau, 4, DEFAULT_FOLDS)
    assert (tau[f.test_idx] >= HOLDOUT).all() and len(f.test_idx) > 0
    train = np.concatenate([f.fit_idx, f.cal_idx])
    assert (tau[train] + timedelta(hours=4) <= HOLDOUT).all()


def test_no_holdout_rows_gives_none():
    assert holdout_fold(_tau(end="2025-06-01"), 1, DEFAULT_FOLDS) is None


def test_empty_window_skipped():
    tau = _tau()
    gap = (tau >= "2021-01-01") & (tau < "2021-07-01")
    folds = walk_forward_folds(tau[~gap], 1, DEFAULT_FOLDS)
    starts = [f.test_start for f in folds]
    assert datetime(2021, 1, 1, tzinfo=timezone.utc) not in starts
    assert len(folds) == 11


def test_final_split_purges():
    tau = _tau(end="2019-01-01")
    fit, cal = final_split(tau, 24, 0.2)
    assert tau[fit].max() + timedelta(hours=24) <= tau[cal].min()
    assert len(cal) == int(round(len(tau) * 0.2)) or abs(len(cal) - len(tau) * 0.2) < 2
