"""Leakage canaries (spec section 8.2): the most important ML tests.

1. On a random walk the whole pipeline must find no skill.
2. A leaked future return must be caught (proves 1 can fail).
3. A planted signal must be found (proves the harness can see information).
4. A 4h feature joined before its bar closes must leak; the real join must not.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml.baselines import BaseRate
from ml.dataset import asof_join, build_dataset
from ml.evaluate import evaluate
from ml.folds import FoldConfig, walk_forward_folds
from ml.models import XGB, LogReg
from tests.unit.ml_synthetic import (
    _resample, planted_signal_symbol, random_walk_bars, symbol_from_bars,
)

pytestmark = pytest.mark.slow

H = 4
START = "2020-01-01"
HOURS = 9000
CFG = FoldConfig(
    first_test=datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=180),
    fold_months=2,
    holdout_start=datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(hours=HOURS - 720),
    cal_fraction=0.2,
)


def _models():
    return [BaseRate(), LogReg(), XGB()]


def _run(ds, models=_models):
    return evaluate(ds, H, walk_forward_folds(ds.tau, H, CFG), models).metrics["models"]


@pytest.fixture(scope="module")
def walk():
    bars = {s: random_walk_bars(seed, START, HOURS)
            for seed, s in enumerate(("BTCUSDT", "ETHUSDT"))}
    data = {s: symbol_from_bars(b) for s, b in bars.items()}
    return bars, data, build_dataset(data, H)


def test_random_walk_has_no_skill(walk):
    _, _, ds = walk
    m = _run(ds)
    for name in ("logreg_v1", "xgb_v1"):
        assert m[name]["skill_lo"] <= 0.0, (name, m[name]["skill"], m[name]["skill_lo"])


def _per_row(ds, frames: dict[str, pd.Series]) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([ds.symbol, ds.open_time])
    full = pd.concat(frames, names=["symbol", "open_time"])
    return full.reindex(keys).to_numpy(dtype=float)


def test_leaked_future_return_is_caught(walk):
    bars, _, ds = walk
    leak = {s: np.log(b["close"].shift(-H) / b["close"]) for s, b in bars.items()}
    ds.X = ds.X.assign(leak=_per_row(ds, leak))
    try:
        m = _run(ds, lambda: [BaseRate(), XGB()])
    finally:
        ds.X = ds.X.drop(columns="leak")
    assert m["xgb_v1"]["skill"] > 0.20


def test_planted_signal_is_found():
    data = {s: planted_signal_symbol(seed, START, HOURS, H)
            for seed, s in enumerate(("BTCUSDT", "ETHUSDT"))}
    m = _run(build_dataset(data, H), lambda: [BaseRate(), XGB()])
    assert m["xgb_v1"]["skill_lo"] > 0.0


def _h4_return_joined(bars, ds, step):
    frames = {}
    for s, b in bars.items():
        b4 = _resample(b, "4h")
        ret = pd.DataFrame({"ret": np.log(b4["close"] / b4["open"])}, index=b4.index)
        frames[s] = asof_join(b.index, ret, step, "h4_")["h4_ret"]
    return _per_row(ds, frames)


def test_off_by_one_4h_join_leaks(walk):
    bars, _, ds = walk
    models = lambda: [BaseRate(), XGB()]
    ds.X = ds.X.assign(h4_bad=_h4_return_joined(bars, ds, timedelta(0)))
    try:
        wrong = _run(ds, models)
    finally:
        ds.X = ds.X.drop(columns="h4_bad")
    assert wrong["xgb_v1"]["skill_lo"] > 0.0

    ds.X = ds.X.assign(h4_ok=_h4_return_joined(bars, ds, timedelta(hours=4)))
    try:
        right = _run(ds, models)
    finally:
        ds.X = ds.X.drop(columns="h4_ok")
    assert right["xgb_v1"]["skill_lo"] <= 0.0
