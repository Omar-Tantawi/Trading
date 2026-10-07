import json

from ml.folds import DEFAULT_FOLDS
from ml.load import run_config


def test_run_config_is_json_serialisable():
    cfg = run_config(4, DEFAULT_FOLDS)
    back = json.loads(json.dumps(cfg))
    assert back["folds"]["holdout_start"].startswith("2025-10-01")
    assert back["label_k"] == 0.5 and back["horizon"] == 4
    assert back["models"][-1] == "xgb_v1"
