"""Saved models (spec section 7.2)."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.artifacts import ArtifactMismatch, load_model, save_model
from ml.models import LogReg


def _fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = rng.integers(0, 3, 300)
    return LogReg().fit(X[:200], y[:200], X[200:], y[200:]), X


def test_save_load_round_trip(tmp_path):
    model, X = _fitted()
    path = save_model(model, 4, {"run_id": 3, "symbols": ["BTCUSDT"]}, tmp_path)
    assert path == tmp_path / "logreg_v1_h4"
    back, meta = load_model("logreg_v1", 4, tmp_path)
    assert np.allclose(back.predict_proba(X), model.predict_proba(X))
    assert meta["run_id"] == 3 and meta["columns"] == list("abcd")
    assert meta["horizon"] == 4 and meta["model"] == "logreg_v1"


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("xgb_v1", 4, tmp_path)


def test_load_refuses_other_feature_set(tmp_path):
    model, _ = _fitted()
    path = save_model(model, 1, {}, tmp_path)
    meta = json.loads((path / "meta.json").read_text())
    meta["feature_set"] += 1
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactMismatch, match="feature set"):
        load_model("logreg_v1", 1, tmp_path)
