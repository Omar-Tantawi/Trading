"""Current probabilities for one symbol (spec section 7.3).

Reads the newest 1h feature row, applies the saved models and quotes the
skill each model showed on unseen data. Describes; never recommends.
"""
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from features.frame import build_cutoff, load_bars
from features.store import read_features
from ml import store
from ml.artifacts import MODELS_DIR, PREDICT_MODELS, ArtifactMismatch, load_model
from ml.dataset import SymbolData, encode, symbol_features
from ml.labels import HORIZONS, threshold
from ml.report import verdict

class NoRecentFeatures(Exception):
    pass


def check_symbol(symbol: str, meta: dict) -> None:
    """Refuse a symbol the model was not trained on."""
    if symbol not in meta.get("symbols", []):
        raise ArtifactMismatch(f"{symbol} was not among the training symbols "
                               f"({', '.join(meta.get('symbols', []))})")


# Enough history for the newest 1h row and the 4h and 1d rows it joins.
_RECENT = timedelta(days=3)


@dataclass
class HorizonPrediction:
    horizon: int
    model: str
    p_down: float | None
    p_flat: float | None
    p_up: float | None
    up_above: float | None
    down_below: float | None
    outcome_time: datetime
    skill: float | None
    skill_lo: float | None
    skill_hi: float | None
    note: str | None = None


@dataclass
class PredictionState:
    symbol: str
    bar_open_time: datetime
    close: float
    predictions: list[HorizonPrediction]

    def to_dict(self) -> dict:
        return asdict(self)


def predict_rows(model, X_row: pd.DataFrame, meta: dict) -> np.ndarray:
    """The model's probabilities for X_row, after checking that its columns
    are exactly the ones the model was trained on."""
    expected = meta["columns"]
    if set(X_row.columns) != set(expected):
        missing = sorted(set(expected) - set(X_row.columns))[:3]
        extra = sorted(set(X_row.columns) - set(expected))[:3]
        raise ArtifactMismatch(f"column mismatch: missing {missing}, extra {extra}")
    return model.predict_proba(X_row[expected])


def _recent(conn, symbol: str) -> tuple[SymbolData, float]:
    cutoff = build_cutoff(conn, symbol, "1h")
    start = cutoff - _RECENT
    bars = load_bars(conn, symbol, "1h", start, cutoff)
    sd = SymbolData(
        close=bars["close"],
        f1h=read_features(conn, symbol, "1h", start),
        f4h=read_features(conn, symbol, "4h", start),
        f1d=read_features(conn, symbol, "1d", start),
    )
    return sd, bars["close"]


def prediction_state(conn, symbol: str, root: Path = MODELS_DIR) -> PredictionState:
    """Raises FileNotFoundError when a model is missing and ArtifactMismatch
    when one no longer fits the code."""
    sd, closes = _recent(conn, symbol)
    if sd.f1h.empty:
        raise NoRecentFeatures(f"{symbol}: no 1h features from the last 3 days")
    open_time = sd.f1h.index[-1]
    close = float(closes.loc[open_time])
    atr_pct = sd.f1h["atr_pct"].iloc[-1]
    features = symbol_features(sd)
    preds = []
    for h in HORIZONS:
        for name in PREDICT_MODELS:
            model, meta = load_model(name, h, root)
            check_symbol(symbol, meta)
            X = encode(features, symbol, tuple(meta["symbols"])).iloc[[-1]]
            run = store.load_run(conn, meta["run_id"]) if meta.get("run_id") else None
            s = run["metrics"]["models"].get(name) if run else None
            outcome = (open_time + pd.Timedelta(hours=h + 1)).to_pydatetime()
            common = dict(horizon=h, model=name, outcome_time=outcome,
                          skill=s and s["skill"], skill_lo=s and s["skill_lo"],
                          skill_hi=s and s["skill_hi"])
            if pd.isna(atr_pct):
                preds.append(HorizonPrediction(
                    p_down=None, p_flat=None, p_up=None, up_above=None,
                    down_below=None, note="atr_pct missing; no prediction",
                    **common))
                continue
            p = predict_rows(model, X, meta)[0]
            thr = threshold(float(atr_pct), h)
            preds.append(HorizonPrediction(
                p_down=float(p[0]), p_flat=float(p[1]), p_up=float(p[2]),
                up_above=close * math.exp(thr), down_below=close * math.exp(-thr),
                **common))
    return PredictionState(symbol, open_time.to_pydatetime(), close, preds)


def render(state: PredictionState) -> str:
    closed = state.bar_open_time + timedelta(hours=1)
    lines = [f"{state.symbol} probabilities from the 1h bar that closed "
             f"{closed:%Y-%m-%d %H:%M} UTC (close {state.close:,.2f})", ""]
    for p in state.predictions:
        fmt = "%H:%M" if p.outcome_time.date() == closed.date() else "%Y-%m-%d %H:%M"
        head = f"next {p.horizon}h (to {p.outcome_time:{fmt}} UTC), {p.model}: "
        if p.note:
            lines.append(head + p.note)
        else:
            lines.append(
                head + f"down {p.p_down:.0%} | flat {p.p_flat:.0%} | up {p.p_up:.0%}  "
                f"(up = close above {p.up_above:,.0f}; down = below {p.down_below:,.0f})")
        if p.skill is None:
            lines.append("    no walk-forward run recorded for this model")
        else:
            line = "    " + verdict(p.skill, p.skill_lo, p.skill_hi)
            if p.skill_lo <= 0:
                line += "; this model has not shown skill on unseen data"
            lines.append(line)
    lines.append("")
    lines.append("Probabilities are measurements of how similar past situations "
                 "turned out, not advice.")
    return "\n".join(lines)
