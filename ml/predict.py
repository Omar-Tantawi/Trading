"""Current probabilities for one symbol (spec section 7.3).

Reads the newest 1h feature row, applies the saved models and quotes the
skill each model showed on unseen data. Describes; never recommends.
"""
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features.frame import build_cutoff, load_bars
from features.store import read_features
from features.summary import STALE_GRACE, _age
from ml import store
from ml.artifacts import MODELS_DIR, PREDICT_MODELS, ArtifactMismatch, load_model
from ml.dataset import SymbolData, encode, symbol_features
from ml.labels import HORIZONS, threshold
from ml.targets import TARGETS, VOL_QUIET, VOL_WILD
from ml.report import verdict

class NoRecentFeatures(Exception):
    pass


def check_symbol(symbol: str, meta: dict) -> None:
    """Refuse a symbol the model was not trained on."""
    if symbol not in meta.get("symbols", []):
        raise ArtifactMismatch(f"{symbol} was not among the training symbols "
                               f"({', '.join(meta.get('symbols', []))})")


_STEP = timedelta(hours=1)


def is_stale(age: timedelta) -> bool:
    """The tb analyze rule for a 1h bar: older than 2 bars plus grace."""
    return age > 2 * _STEP + STALE_GRACE


# Enough history for the newest 1h row and the 4h and 1d rows it joins.
_RECENT = timedelta(days=3)


TARGET_WORDS = {"move3": "move", "vol3": "volatility", "dir2": "direction"}


@dataclass
class HorizonPrediction:
    horizon: int
    model: str
    target: str
    classes: tuple[str, ...]
    probs: list[float] | None        # one per class; None with a note
    meaning: str | None              # what the classes mean in prices
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
    age: timedelta | None = None     # now minus the bar's close
    stale: bool = False
    skipped: list[str] = field(default_factory=list)   # targets without models

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
        close=bars["close"], high=bars["high"], low=bars["low"],
        f1h=read_features(conn, symbol, "1h", start),
        f4h=read_features(conn, symbol, "4h", start),
        f1d=read_features(conn, symbol, "1d", start),
    )
    return sd, bars["close"]


def _meaning(target: str, close: float, atr_pct: float, horizon: int) -> str | None:
    if target == "move3":
        thr = threshold(atr_pct, horizon)
        return (f"up = close above {close * math.exp(thr):,.0f}; "
                f"down = below {close * math.exp(-thr):,.0f}")
    if target == "vol3":
        unit = atr_pct * math.sqrt(horizon)
        quiet = math.exp(VOL_QUIET * unit)
        wild = math.exp(VOL_WILD * unit)
        return (f"quiet = stays between {close / quiet:,.0f} and {close * quiet:,.0f}; "
                f"wild = reaches {close / wild:,.0f} or {close * wild:,.0f}")
    return None


def _skip_lines(problems: dict, shown: set[str]) -> list[str]:
    """One line per target with nothing saved, else one per model that is
    missing or no longer fits."""
    lines = []
    for target in TARGETS:
        hint = f"run `tb ml train --target {target}`"
        word = TARGET_WORDS[target]
        mine = {m: r for (t, m), r in problems.items() if t == target}
        if not mine:
            continue
        if target not in shown and all(r is None for r in mine.values()):
            lines.append(f"{word}: no saved models; {hint}")
            continue
        for model, reason in mine.items():
            lines.append(f"{word} ({model}): {reason or 'no saved model'}; {hint}")
    return lines


def prediction_state(conn, symbol: str, root: Path = MODELS_DIR,
                     now: datetime | None = None) -> PredictionState:
    """Every target with saved models, for each horizon. Raises
    FileNotFoundError when no model of any target is saved, and
    ArtifactMismatch when one no longer fits the code."""
    sd, closes = _recent(conn, symbol)
    if sd.f1h.empty:
        raise NoRecentFeatures(f"{symbol}: no 1h features from the last 3 days")
    open_time = sd.f1h.index[-1]
    close = float(closes.loc[open_time])
    atr_pct = sd.f1h["atr_pct"].iloc[-1]
    features = symbol_features(sd)
    preds, problems = [], {}       # (target, model) -> reason
    for h in HORIZONS:
        for target in TARGETS.values():
            for name in PREDICT_MODELS:
                try:
                    model, meta = load_model(name, h, root, target.name)
                    check_symbol(symbol, meta)
                except FileNotFoundError:
                    problems.setdefault((target.name, name), None)
                    continue
                except ArtifactMismatch as exc:
                    # one stale target must not hide the others
                    problems.setdefault((target.name, name),
                                        str(exc).split(": ", 1)[-1])
                    continue
                X = encode(features, symbol, tuple(meta["symbols"])).iloc[[-1]]
                run = store.load_run(conn, meta["run_id"]) if meta.get("run_id") else None
                s = run["metrics"]["models"].get(name) if run else None
                common = dict(horizon=h, model=name, target=target.name,
                              classes=target.classes,
                              outcome_time=(open_time + pd.Timedelta(hours=h + 1)).to_pydatetime(),
                              skill=s and s["skill"], skill_lo=s and s["skill_lo"],
                              skill_hi=s and s["skill_hi"])
                if pd.isna(atr_pct):
                    preds.append(HorizonPrediction(
                        probs=None, meaning=None,
                        note="atr_pct missing; no prediction", **common))
                    continue
                p = predict_rows(model, X, meta)[0]
                preds.append(HorizonPrediction(
                    probs=[float(v) for v in p],
                    meaning=_meaning(target.name, close, float(atr_pct), h), **common))
    skipped = _skip_lines(problems, {p.target for p in preds})
    if not preds:
        raise FileNotFoundError("no saved models")
    now = now or datetime.now(timezone.utc)
    age = now - (open_time.to_pydatetime() + _STEP)
    return PredictionState(symbol, open_time.to_pydatetime(), close, preds,
                           age=age, stale=is_stale(age), skipped=skipped)


def render(state: PredictionState) -> str:
    closed = state.bar_open_time + timedelta(hours=1)
    ago = f", {_age(state.age)}" if state.age is not None else ""   # _age says "ago"
    lines = [f"{state.symbol} probabilities from the 1h bar that closed "
             f"{closed:%Y-%m-%d %H:%M} UTC (close {state.close:,.2f}){ago}"]
    if state.stale:
        lines.append(f"STALE: these probabilities are for a bar that closed "
                     f"{_age(state.age)}. Run `tb backfill` (or keep `tb live` "
                     "running), then `tb predict` again.")
    lines.append("")
    for p in state.predictions:
        fmt = "%H:%M" if p.outcome_time.date() == closed.date() else "%Y-%m-%d %H:%M"
        head = (f"next {p.horizon}h (to {p.outcome_time:{fmt}} UTC), {p.model}, "
                f"{TARGET_WORDS.get(p.target, p.target)}: ")
        if p.note:
            lines.append(head + p.note)
        else:
            probs = " | ".join(f"{c} {v:.0%}" for c, v in zip(p.classes, p.probs))
            lines.append(head + probs + (f"  ({p.meaning})" if p.meaning else ""))
        if p.skill is None:
            lines.append("    no walk-forward run recorded for this model")
        else:
            line = "    " + verdict(p.skill, p.skill_lo, p.skill_hi)
            if p.skill_lo <= 0:
                line += "; this model has not shown skill on unseen data"
            lines.append(line)
    if state.skipped:
        lines.append("")
        lines.extend(state.skipped)
    lines.append("")
    lines.append("Probabilities are measurements of how similar past situations "
                 "turned out, not advice.")
    return "\n".join(lines)
