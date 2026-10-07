"""Loading training data from the database (spec section 7.3).

Features must be built, current and of this FEATURE_SET before a model may
train on them; check_fresh says which are not.
"""
from dataclasses import asdict

from data.quality.checks import STEP
from features.frame import build_cutoff, load_bars
from features.pipeline import FEATURE_SET
from features.store import has_stale_feature_set, last_built, read_features
from ml.dataset import SymbolData
from ml.folds import FoldConfig
from ml.labels import LABEL_K, LABEL_SET
from ml.models import LOGREG_PARAMS, XGB_PARAMS, XGB_ROUNDS, make_models

ML_TIMEFRAMES = ("1h", "4h", "1d")


class StaleFeaturesError(Exception):
    pass


def check_fresh(conn, symbols) -> None:
    """Raise StaleFeaturesError naming every (symbol, timeframe) whose
    features are missing, of another FEATURE_SET, or more than two bars
    behind the stored 1m candles."""
    problems = []
    for symbol in symbols:
        for tf in ML_TIMEFRAMES:
            cutoff = build_cutoff(conn, symbol, tf)
            built = last_built(conn, symbol, tf)
            if cutoff is None or built is None:
                problems.append(f"{symbol} {tf}: no features")
            elif has_stale_feature_set(conn, symbol, tf):
                problems.append(f"{symbol} {tf}: built by another feature set")
            elif built < cutoff - 2 * STEP[tf]:
                problems.append(f"{symbol} {tf}: newest bar {built:%Y-%m-%d %H:%M} UTC")
    if problems:
        raise StaleFeaturesError("features are not current: " + "; ".join(problems))


def load_symbol_data(conn, symbol: str) -> SymbolData:
    cutoff = build_cutoff(conn, symbol, "1h")
    close = load_bars(conn, symbol, "1h", None, cutoff)["close"]
    return SymbolData(
        close=close,
        f1h=read_features(conn, symbol, "1h"),
        f4h=read_features(conn, symbol, "4h"),
        f1d=read_features(conn, symbol, "1d"),
    )


def run_config(horizon: int, cfg: FoldConfig) -> dict:
    """Every constant that shaped a run, JSON-serialisable."""
    folds = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
             for k, v in asdict(cfg).items()}
    return {
        "horizon": horizon, "label_k": LABEL_K, "label_set": LABEL_SET,
        "feature_set": FEATURE_SET, "folds": folds,
        "models": [m.name for m in make_models()],
        "logreg": LOGREG_PARAMS, "xgb": {**XGB_PARAMS, "rounds": XGB_ROUNDS},
    }
