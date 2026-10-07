"""Where does a model's skill come from? (follow-up to spec section 6)

Splits the three-way forecast into two yes/no questions, scored on stored
out-of-sample predictions against base_rate_v1 on the same rows:

* size: will the price move (up or down) or stay flat?
  P(move) = p_up + p_down.
* direction: given that it moved, which way? Scored only on rows whose
  outcome was up or down; P(up | move) = p_up / (p_up + p_down).

Skill on size alone says the model knows how much the price will move
(volatility), not where. Pure: no database, no clock.
"""
import numpy as np
import pandas as pd

from ml.evaluate import BASE
from ml.labels import FLAT, UP
from ml.metrics import bootstrap_skill_ci, log_loss, skill
from ml.report import MODEL_ORDER, verdict

_KEY = ["symbol", "open_time"]


def _score(p_model, p_base, y, tau, extra=None) -> dict:
    lo, hi = bootstrap_skill_ci(p_model, p_base, y, tau)
    out = {"n": int(len(y)), "skill": skill(log_loss(p_model, y), log_loss(p_base, y)),
           "skill_lo": lo, "skill_hi": hi}
    out.update(extra or {})
    return out


def _two(p_yes: np.ndarray) -> np.ndarray:
    p_yes = np.clip(p_yes, 1e-15, 1 - 1e-15)
    return np.column_stack([1 - p_yes, p_yes])


def split_skill(pred: pd.DataFrame) -> dict:
    """Per model (base rate excluded): {"size": {...}, "direction": {...}}."""
    base = pred[pred["model"] == BASE].sort_values(_KEY, kind="stable")
    y = base["label"].to_numpy(dtype=int)
    tau = pd.DatetimeIndex(base["open_time"]) + pd.Timedelta(hours=1)
    moved = y != FLAT
    up = (y == UP).astype(int)

    def parts(rows):
        p_move = rows["p_up"].to_numpy() + rows["p_down"].to_numpy()
        q_up = rows["p_up"].to_numpy() / p_move
        return p_move, q_up

    b_move, b_up = parts(base)
    out = {}
    for name in pred["model"].unique():
        if name == BASE:
            continue
        rows = pred[pred["model"] == name].sort_values(_KEY, kind="stable")
        m_move, m_up = parts(rows)
        out[name] = {
            "size": _score(_two(m_move), _two(b_move), moved.astype(int), tau),
            "direction": _score(
                _two(m_up[moved]), _two(b_up[moved]), up[moved], tau[moved],
                {"accuracy": float(((m_up[moved] > 0.5) == up[moved]).mean()),
                 "base_accuracy": float(((b_up[moved] > 0.5) == up[moved]).mean())}),
        }
    return out


def _line(label: str, s: dict) -> str:
    return f"  {label:<36}{verdict(s['skill'], s['skill_lo'], s['skill_hi'])}"


def render_diagnosis(run_id: int, horizon: int, split: dict) -> str:
    rank = {n: i for i, n in enumerate(MODEL_ORDER)}
    lines = [f"Run {run_id}, next {horizon}h: where the skill comes from", ""]
    for name in sorted(split, key=lambda n: (rank.get(n, len(rank)), n)):
        s = split[name]
        d = s["direction"]
        lines.append(f"{name}")
        lines.append(_line("size (move vs flat):", s["size"]))
        lines.append(_line("direction (up vs down, moves only):", d))
        lines.append(f"  {'':<36}side right in {d['accuracy']:.1%} of {d['n']:,} moves "
                     f"(base rate's side: {d['base_accuracy']:.1%})")
    lines.append("")
    lines.append("Size skill means knowing how much the price will move, not where. "
                 "Direction is scored only on hours that did move.")
    return "\n".join(lines)
