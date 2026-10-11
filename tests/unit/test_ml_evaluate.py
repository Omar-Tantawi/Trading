"""Walk-forward evaluation (spec sections 4.2 and 6)."""
from datetime import datetime, timezone

import numpy as np

from ml.baselines import BaseRate
from ml.dataset import build_dataset
from ml.evaluate import evaluate
from ml.folds import FoldConfig, walk_forward_folds
from ml.models import LogReg
from tests.unit.ml_synthetic import random_walk_symbol

CFG = FoldConfig(first_test=datetime(2020, 4, 1, tzinfo=timezone.utc),
                 fold_months=1,
                 holdout_start=datetime(2020, 7, 1, tzinfo=timezone.utc),
                 cal_fraction=0.2)


def test_predictions_and_summary():
    data = {"BTCUSDT": random_walk_symbol(1, "2020-01-01", 4500),
            # listed later: no rows in the first test window
            "SOLUSDT": random_walk_symbol(2, "2020-04-20", 1000)}
    ds = build_dataset(data, 4)
    folds = walk_forward_folds(ds.tau, 4, CFG)
    res = evaluate(ds, 4, folds, lambda: [BaseRate(), LogReg()])
    pred = res.predictions
    n_test = sum(len(f.test_idx) for f in folds)
    assert len(pred) == 2 * n_test
    assert list(pred.columns) == ["model", "symbol", "open_time", "fold", "label",
                                  "p0", "p1", "p2"]
    assert np.allclose(pred[["p0", "p1", "p2"]].sum(axis=1), 1.0)
    m = res.metrics
    assert m["horizon"] == 4 and m["n_test"] == n_test
    assert m["models"]["base_rate_v1"]["skill"] == 0.0
    assert abs(sum(m["class_shares"].values()) - 1.0) < 1e-9
    first = m["models"]["logreg_v2"]["per_fold"][0]
    assert set(first) == {"fold", "test_start", "n", "skill"}
    assert set(m["models"]["logreg_v2"]["per_symbol"]) == {"BTCUSDT", "SOLUSDT"}
    sol_folds = {f["fold"] for f in m["models"]["logreg_v2"]["per_fold"]}
    assert sol_folds == {f.number for f in folds}


def test_progress_lines():
    data = {"BTCUSDT": random_walk_symbol(1, "2020-01-01", 4500)}
    ds = build_dataset(data, 4)
    folds = walk_forward_folds(ds.tau, 4, CFG)
    lines = []
    evaluate(ds, 4, folds, lambda: [BaseRate(), LogReg()], progress=lines.append)
    assert len(lines) == len(folds)
    assert lines[0].startswith(f"  next 4h: period 1/{len(folds)} (2020-04)")
    assert "base_rate_v1" in lines[0] and "logreg_v2" in lines[0] and "s" in lines[0]


def test_two_class_evaluation():
    from ml.targets import get_target
    data = {"BTCUSDT": random_walk_symbol(1, "2020-01-01", 4500)}
    ds = build_dataset(data, 4, target=get_target("dir2"))
    folds = walk_forward_folds(ds.tau, 4, CFG)
    res = evaluate(ds, 4, folds, lambda: [BaseRate(2), LogReg(2)],
                   classes=("down", "up"))
    assert list(res.predictions.columns)[-2:] == ["p0", "p1"]
    assert list(res.metrics["class_shares"]) == ["down", "up"]
    assert res.metrics["models"]["base_rate_v1"]["skill"] == 0.0
