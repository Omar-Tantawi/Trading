"""Trained models on disk (spec section 7.2).

Each model lives in models/<name>_h<H>/: model.joblib (the fitted object,
temperature included) and meta.json. A model built for another label set,
feature set or column list is refused rather than silently misused.
"""
import json
from pathlib import Path

import joblib

from features.pipeline import FEATURE_SET
from ml.labels import LABEL_SET
from ml.models import Model

MODELS_DIR = Path("models")
PREDICT_MODELS = ("logreg_v2", "xgb_v1")


class ArtifactMismatch(Exception):
    pass


def _dir(name: str, horizon: int, root: Path) -> Path:
    return Path(root) / f"{name}_h{horizon}"


def save_model(model: Model, horizon: int, meta: dict,
               root: Path = MODELS_DIR) -> Path:
    path = _dir(model.name, horizon, root)
    path.mkdir(parents=True, exist_ok=True)
    full = {**meta, "model": model.name, "horizon": horizon,
            "label_set": LABEL_SET, "feature_set": FEATURE_SET,
            "columns": list(model.columns)}
    joblib.dump(model, path / "model.joblib")
    (path / "meta.json").write_text(json.dumps(full, indent=2, default=str))
    return path


def load_model(name: str, horizon: int,
               root: Path = MODELS_DIR) -> tuple[Model, dict]:
    path = _dir(name, horizon, root)
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no saved model {path}")
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("feature_set") != FEATURE_SET:
        raise ArtifactMismatch(f"{path}: built for feature set "
                               f"{meta.get('feature_set')}, code is {FEATURE_SET}")
    if meta.get("label_set") != LABEL_SET:
        raise ArtifactMismatch(f"{path}: built for label set "
                               f"{meta.get('label_set')}, code is {LABEL_SET}")
    return joblib.load(path / "model.joblib"), meta
