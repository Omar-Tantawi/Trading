"""Grid conditions for one coin (spec section 2). Pure: inputs in, signal out.

Each failing gate adds a reason and makes the status WAIT; so does a
forecast P(wild) at or above the user's threshold. Nothing here recommends
an order; WAIT means "unlike the days the grid ran in the backtest".
"""
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from backtest.grid import N_GRIDS, levels, step_too_small
from features.summary import _age
from ml.predict import is_stale
from signals.settings import RiskSettings

MODEL_HINT = "run `tb ml train --target vol3`"


@dataclass
class Inputs:
    symbol: str
    has_data: bool                    # any 1m candles stored
    quality_fail: bool | None         # None: not checked (--no-build)
    bar_open_time: datetime | None    # newest 1h feature row
    age: timedelta | None             # now minus that bar's close
    close: float | None
    atr_pct: float | None
    p_wild: float | None
    model_problem: str | None         # None when the vol3 24h model loaded
    build_problem: str | None = None  # the feature build raised or skipped a timeframe


@dataclass
class Signal:
    symbol: str
    status: str                       # "WAIT" or "NORMAL"
    reasons: list[str]
    p_wild: float | None
    bar_open_time: datetime | None
    age: timedelta | None
    close: float | None
    range_low: float | None = None
    range_high: float | None = None
    step_price: float | None = None
    step_pct: float | None = None     # the smallest step, as a share of its level
    n_grids: int = N_GRIDS
    capital_limit: float | None = None
    notes: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(v: float) -> str:
    """A threshold, e.g. 25% or 30.5%."""
    return f"{round(v * 100, 1):g}%"


def _pct_down(v: float) -> str:
    """A probability rounded down to 0.1 %, so 0.2496 shows as 24.9% and never
    seems to equal a 25% threshold it is below."""
    return f"{math.floor(v * 1000 + 1e-9) / 10:.1f}%"


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def evaluate(x: Inputs, s: RiskSettings) -> Signal:
    reasons: list[str] = []
    notes: list[str] = []
    if not x.has_data:
        reasons.append("no market data")
    if x.build_problem is not None:
        reasons.append(f"feature build failed ({x.build_problem}); data quality "
                       "not checked and features may be out of date")
    if x.quality_fail:
        reasons.append("data quality FAIL")
    elif x.quality_fail is None and x.build_problem is None:
        notes.append("data quality not checked (--no-build)")
    if x.has_data and x.bar_open_time is None:
        reasons.append("no 1h features from the last 3 days (run `tb features build`)")
    elif x.age is not None and is_stale(x.age):
        reasons.append(f"data STALE (newest 1h bar closed {_age(x.age)})")
    if x.model_problem is not None:
        detail = "" if x.model_problem == "no saved model" else f": {x.model_problem}"
        reasons.append(f"no forecast model{detail} ({MODEL_HINT})")
    if x.bar_open_time is not None and _missing(x.atr_pct):
        reasons.append("volatility measure missing")

    sig = Signal(symbol=x.symbol, status="NORMAL", reasons=reasons, p_wild=x.p_wild,
                 bar_open_time=x.bar_open_time, age=x.age, close=x.close,
                 notes=notes, settings=s.to_dict())
    if not _missing(x.atr_pct) and not _missing(x.close):
        lv = levels(x.close, x.atr_pct)
        steps = (lv[1:] - lv[:-1]) / lv[:-1]
        sig.range_low, sig.range_high = float(lv[0]), float(lv[-1])
        sig.step_price, sig.step_pct = float(lv[1] - lv[0]), float(steps.min())
        if step_too_small(lv, s.fee):
            reasons.append(f"grid step {sig.step_pct:.2%} cannot pay two fees "
                           f"({2 * s.fee:.2%})")
    if s.capital_usdt is not None:
        sig.capital_limit = s.capital_usdt * s.max_coin_share

    if x.p_wild is not None and x.p_wild >= s.wild_threshold:
        reasons.append(f"wild day expected (P(wild) {_pct_down(x.p_wild)} >= "
                       f"{_pct(s.wild_threshold)})")
    if not reasons and x.p_wild is None:
        reasons.append("no forecast available")
    if reasons:
        sig.status = "WAIT"
    else:
        reasons.append(f"P(wild) {_pct_down(x.p_wild)} < {_pct(s.wild_threshold)}")
    return sig


def error_signal(symbol: str, message: str, s: RiskSettings) -> Signal:
    """WAIT for a coin whose inputs could not be read at all."""
    return Signal(symbol=symbol, status="WAIT", reasons=[f"could not evaluate: {message}"],
                  p_wild=None, bar_open_time=None, age=None, close=None,
                  capital_limit=None if s.capital_usdt is None
                  else s.capital_usdt * s.max_coin_share,
                  settings=s.to_dict())
