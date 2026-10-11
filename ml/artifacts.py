"""Trained models on disk (spec section 7.2).

Each model lives in models/<name>_h<H>/: model.joblib (the fitted object,
temperature included) and meta.json. A model built for another label set,
feature set or column list is refused rather than silently misused.
"""
import json
from pathlib import Path

import joblib

from features.pipeline import FEATURE_SET
from ml.targets import get_target
from ml.models import Model

MODELS_DIR = Path("models")
PREDICT_MODELS = ("logreg_v2", "xgb_v1")


class ArtifactMismatch(Exception):
    pass


def _dir(name: str, horizon: int, root: Path, target: str) -> Path:
    # move3 keeps the sub-project 3 paths, so models saved then still load
    prefix = "" if target == "move3" else f"{target}_"
    return Path(root) / f"{prefix}{name}_h{horizon}"


def save_model(model: Model, horizon: int, meta: dict,
               root: Path = MODELS_DIR, target: str = "move3") -> Path:
    path = _dir(model.name, horizon, root, target)
    path.mkdir(parents=True, exist_ok=True)
    full = {**meta, "model": model.name, "horizon": horizon, "target": target,
            "label_set": get_target(target).label_set, "feature_set": FEATURE_SET,
            "columns": list(model.columns)}
    joblib.dump(model, path / "model.joblib")
    (path / "meta.json").write_text(json.dumps(full, indent=2, default=str))
    return path


def load_model(name: str, horizon: int, root: Path = MODELS_DIR,
               target: str = "move3") -> tuple[Model, dict]:
    path = _dir(name, horizon, root, target)
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no saved model {path}")
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("feature_set") != FEATURE_SET:
        raise ArtifactMismatch(f"{path}: built for feature set "
                               f"{meta.get('feature_set')}, code is {FEATURE_SET}")
    saved_target = meta.get("target", "move3")   # models saved before 3b
    if saved_target != target:
        raise ArtifactMismatch(f"{path}: built for target {saved_target}, "
                               f"asked for {target}")
    label_set = get_target(target).label_set
    if meta.get("label_set") != label_set:
        raise ArtifactMismatch(f"{path}: built for label set "
                               f"{meta.get('label_set')}, code is {label_set}")
    return joblib.load(path / "model.joblib"), meta
