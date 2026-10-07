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


def _hp(h, model, note=None, skill=(0.008, 0.003, 0.013)):
    p = (None, None, None) if note else (0.31, 0.42, 0.27)
    return HorizonPrediction(
        horizon=h, model=model, p_down=p[0], p_flat=p[1], p_up=p[2],
        up_above=None if note else 84_600.0, down_below=None if note else 83_300.0,
        outcome_time=BAR + timedelta(hours=h + 1),
        skill=skill[0], skill_lo=skill[1], skill_hi=skill[2], note=note)


def _state(note=None):
    return PredictionState(
        symbol="BTCUSDT", bar_open_time=BAR, close=84_000.0,
        predictions=[_hp(h, m, note, (0.001, -0.004, 0.006) if m == "logreg_v1"
                         else (0.008, 0.003, 0.013))
                     for h in (1, 4, 24) for m in ("logreg_v1", "xgb_v1")])


def test_render_lines_and_no_advice_words():
    text = render(_state())
    assert not ADVICE.search(text), ADVICE.search(text)
    assert "next 4h (to 13:00 UTC), xgb_v1: down 31% | flat 42% | up 27%" in text
    assert "(up = close above 84,600; down = below 83,300)" in text
    assert "has not shown skill on unseen data" in text      # logreg CI spans 0
    assert "not advice" in text
    d = _state().to_dict()
    assert d["predictions"][0]["p_flat"] == 0.42 and d["symbol"] == "BTCUSDT"


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
