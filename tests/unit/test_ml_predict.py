"""tb predict's state and rendering (spec section 7.3)."""
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml.artifacts import ArtifactMismatch
from ml.models import LogReg
from ml.predict import HorizonPrediction, PredictionState, predict_rows, render

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)
BAR = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)


def _hp(h, model, note=None, skill=(0.008, 0.003, 0.013), target="move3"):
    classes = {"move3": ("down", "flat", "up"), "vol3": ("quiet", "normal", "wild"),
               "dir2": ("down", "up")}[target]
    probs = None if note else {"move3": [0.31, 0.42, 0.27], "vol3": [0.31, 0.51, 0.18],
                               "dir2": [0.49, 0.51]}[target]
    meaning = None if note else {
        "move3": "up = close above 84,600; down = below 83,300",
        "vol3": "quiet = stays between 83,650 and 84,350; wild = reaches 83,120 or 84,890",
        "dir2": None}[target]
    return HorizonPrediction(
        horizon=h, model=model, target=target, classes=classes, probs=probs,
        meaning=meaning, outcome_time=BAR + timedelta(hours=h + 1),
        skill=skill[0], skill_lo=skill[1], skill_hi=skill[2], note=note)


def _state(note=None):
    return PredictionState(
        symbol="BTCUSDT", bar_open_time=BAR, close=84_000.0,
        predictions=[_hp(h, m, note, (0.001, -0.004, 0.006) if m == "logreg_v2"
                         else (0.008, 0.003, 0.013))
                     for h in (1, 4, 24) for m in ("logreg_v2", "xgb_v1")])


def test_render_lines_and_no_advice_words():
    text = render(_state())
    assert not ADVICE.search(text), ADVICE.search(text)
    assert "next 4h (to 13:00 UTC), xgb_v1, move: down 31% | flat 42% | up 27%" in text
    assert "(up = close above 84,600; down = below 83,300)" in text
    assert "has not shown skill on unseen data" in text      # logreg CI spans 0
    assert "not advice" in text
    d = _state().to_dict()
    assert d["predictions"][0]["probs"][1] == 0.42 and d["symbol"] == "BTCUSDT"


def test_render_null_atr_note():
    text = render(_state(note="atr_pct missing; no prediction"))
    assert "atr_pct missing; no prediction" in text
    assert not ADVICE.search(text)


def test_predict_rows_refuses_column_mismatch():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    model = LogReg().fit(X, rng.integers(0, 3, 100), X[:0], np.array([], int))
    meta = {"columns": list("abc")}
    assert predict_rows(model, X[["c", "a", "b"]].iloc[:1], meta).shape == (1, 3)
    with pytest.raises(ArtifactMismatch):
        predict_rows(model, X[["a", "b"]].iloc[:1], meta)
    with pytest.raises(ArtifactMismatch):
        predict_rows(model, X.assign(d=1.0).iloc[:1], meta)


def test_check_symbol_refuses_untrained_symbol():
    from ml.predict import check_symbol
    check_symbol("BTCUSDT", {"symbols": ["BTCUSDT"]})
    with pytest.raises(ArtifactMismatch, match="DOGEUSDT"):
        check_symbol("DOGEUSDT", {"symbols": ["BTCUSDT"]})


def test_python_m_data_cli_lists_ml_commands():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "data.cli", "predict", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_render_shows_age_and_stale_warning():
    fresh = _state()
    fresh.age, fresh.stale = timedelta(minutes=30), False
    text = render(fresh)
    assert "30m ago" in text and "STALE" not in text
    old = _state()
    old.age, old.stale = timedelta(hours=21, minutes=4), True
    text = render(old)
    assert "21h 4m ago" in text and "ago ago" not in text
    assert "STALE: these probabilities are for a bar that closed 21h 4m ago" in text
    assert "tb backfill" in text
    assert not ADVICE.search(text)


def test_stale_rule_matches_tb_analyze():
    from ml.predict import is_stale
    assert not is_stale(timedelta(hours=2, minutes=5))
    assert is_stale(timedelta(hours=2, minutes=6))


def test_render_vol3_dir2_and_skip_lines():
    state = _state()
    state.predictions = [_hp(4, "xgb_v1", target="vol3"), _hp(4, "xgb_v1", target="dir2")]
    state.skipped = ["next 24h, direction: no saved models; run `tb ml train --target dir2`"]
    text = render(state)
    assert ("next 4h (to 13:00 UTC), xgb_v1, volatility: quiet 31% | normal 51% | wild 18%"
            "  (quiet = stays between 83,650 and 84,350; wild = reaches 83,120 or 84,890)") in text
    assert "next 4h (to 13:00 UTC), xgb_v1, direction: down 49% | up 51%\n" in text
    assert "run `tb ml train --target dir2`" in text
    assert not ADVICE.search(text)


def test_vol3_meaning_uses_exact_prices():
    from ml.predict import _meaning
    m = _meaning("vol3", 84_000.0, 0.01, 4)          # unit 0.02
    assert m == "quiet = stays between 83,164 and 84,844; wild = reaches 81,926 or 86,126"
