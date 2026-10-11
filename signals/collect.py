"""The inputs of one coin's signal (spec section 2), read from the database
and the saved vol3 24 h model."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.storage.repository import last_candle_time
from ml.artifacts import MODELS_DIR, ArtifactMismatch, load_model
from ml.dataset import encode, symbol_features
from ml.predict import _recent, check_symbol, predict_rows
from signals.engine import Inputs

MODEL, HORIZON, TARGET, WILD = "xgb_v1", 24, "vol3", 2


def _p_wild(sd, symbol: str, root: Path) -> tuple[float | None, str | None]:
    try:
        model, meta = load_model(MODEL, HORIZON, root, TARGET)
        check_symbol(symbol, meta)
        X = encode(symbol_features(sd), symbol, tuple(meta["symbols"])).iloc[[-1]]
        return float(predict_rows(model, X, meta)[0][WILD]), None
    except FileNotFoundError:
        return None, "no saved model"
    except ArtifactMismatch as exc:
        return None, str(exc).split(": ", 1)[-1]


def collect(conn, symbol: str, quality_fail: bool | None,
            root: Path = MODELS_DIR, now: datetime | None = None) -> Inputs:
    empty = dict(bar_open_time=None, age=None, close=None, atr_pct=None,
                 p_wild=None, model_problem=None)
    if last_candle_time(conn, symbol) is None:
        return Inputs(symbol, False, quality_fail, **empty)
    sd, closes = _recent(conn, symbol)
    if sd.f1h.empty:
        return Inputs(symbol, True, quality_fail, **empty)
    open_time = sd.f1h.index[-1]
    atr = sd.f1h["atr_pct"].iloc[-1]
    atr = None if atr != atr else float(atr)          # NaN -> None
    p_wild, problem = _p_wild(sd, symbol, root)
    if atr is None:
        p_wild = None              # the model was not trained on such rows
    now = now or datetime.now(timezone.utc)
    return Inputs(symbol, True, quality_fail,
                  bar_open_time=open_time.to_pydatetime(),
                  age=now - (open_time.to_pydatetime() + timedelta(hours=1)),
                  close=float(closes.loc[open_time]), atr_pct=atr,
                  p_wild=p_wild, model_problem=problem)
