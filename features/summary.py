"""The readable market state (spec sections 3.6 and 6.2).

`state_from_rows` is pure: it turns the newest feature row of each timeframe
into a `MarketState`. `market_state` reads those rows from the database.
The summary describes; it never recommends, so the rendered text contains no
advice words.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from data.quality.checks import STEP
from features.pipeline import FEATURE_TIMEFRAMES
from features.store import latest_features

# Nothing keeps features fresh yet. A newest bar older than 2 x the
# timeframe plus this grace means the features have not been built lately.
STALE_GRACE = timedelta(minutes=5)

_STRUCTURE = {1: "up", -1: "down", 0: "mixed"}


@dataclass
class TimeframeState:
    timeframe: str
    built: bool                    # False: no feature rows yet; the rest are None
    bar_close: datetime | None     # open_time + step
    age: timedelta | None
    stale: bool
    trend_regime: str | None
    adx: float | None
    structure: str | None          # "up" | "down" | "mixed" | None
    trend_duration: int | None     # signed: + when EMA 20 is above EMA 50
    rsi: float | None
    volatility_regime: str | None
    vol_pct: float | None
    volume_ratio: float | None
    buy_share: float | None
    dist_ema_200: float | None


@dataclass
class MarketState:
    symbol: str
    as_of: datetime
    timeframes: list[TimeframeState]   # always all five, in FEATURE_TIMEFRAMES order

    def agreement(self) -> dict[str, int]:
        """How many timeframes are bullish, bearish and sideways by trend regime."""
        counts = {"bullish": 0, "bearish": 0, "sideways": 0}
        for t in self.timeframes:
            regime = t.trend_regime
            if regime is None:
                continue
            if regime.endswith("bullish"):
                counts["bullish"] += 1
            elif regime.endswith("bearish"):
                counts["bearish"] += 1
            elif regime == "sideways":
                counts["sideways"] += 1
        return counts

    def to_dict(self) -> dict:
        """A JSON-serialisable form: datetimes as ISO strings, timedeltas as
        seconds, missing values as None."""
        frames = []
        for t in self.timeframes:
            d = dict(vars(t))
            d["bar_close"] = t.bar_close.isoformat() if t.bar_close else None
            d["age"] = t.age.total_seconds() if t.age is not None else None
            frames.append(d)
        return {"symbol": self.symbol, "as_of": self.as_of.isoformat(),
                "timeframes": frames, "agreement": self.agreement()}

    def render(self) -> str:
        lines = [f"{self.symbol} market state as of "
                 f"{self.as_of:%Y-%m-%d %H:%M} UTC", ""]
        for t in self.timeframes:
            lines.extend(_render_timeframe(t))
        agree = self.agreement()
        lines.append(f"Timeframes: {agree['bullish']} bullish, "
                     f"{agree['bearish']} bearish, {agree['sideways']} sideways "
                     "(by trend regime)")
        return "\n".join(lines)


def _float(value) -> float | None:
    if value is None or value is pd.NA:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _int(value) -> int | None:
    if value is None or value is pd.NA:
        return None
    return int(value)


def _text(value) -> str | None:
    return None if value is None or value is pd.NA else str(value)


def _timeframe_state(timeframe: str, row: dict | None,
                     now: datetime) -> TimeframeState:
    if row is None:
        return TimeframeState(timeframe, False, None, None, False, None, None,
                              None, None, None, None, None, None, None, None)
    step = STEP[timeframe]
    bar_close = row["open_time"] + step
    age = now - bar_close
    structure = _int(row["structure"])
    return TimeframeState(
        timeframe=timeframe, built=True, bar_close=bar_close, age=age,
        stale=age > 2 * step + STALE_GRACE,
        trend_regime=_text(row["trend_regime"]),
        adx=_float(row["adx_14"]),
        structure=None if structure is None else _STRUCTURE[structure],
        trend_duration=_int(row["trend_duration"]),
        rsi=_float(row["rsi_14"]),
        volatility_regime=_text(row["volatility_regime"]),
        vol_pct=_float(row["vol_pct_365d"]),
        volume_ratio=_float(row["volume_ratio_20"]),
        buy_share=_float(row["buy_share"]),
        dist_ema_200=_float(row["dist_ema_200"]),
    )


def state_from_rows(symbol: str, rows: dict[str, dict | None],
                    now: datetime) -> MarketState:
    """Pure: no database, no clock. `rows` maps a timeframe to its newest
    feature row (as `latest_features` returns it) or None; a missing key
    counts as None."""
    return MarketState(symbol, now, [
        _timeframe_state(tf, rows.get(tf), now) for tf in FEATURE_TIMEFRAMES])


def market_state(conn, symbol: str, now: datetime | None = None) -> MarketState:
    """The current state of `symbol`, from the newest stored row per timeframe."""
    now = now or datetime.now(timezone.utc)
    rows = {tf: latest_features(conn, symbol, tf) for tf in FEATURE_TIMEFRAMES}
    return state_from_rows(symbol, rows, now)


# --- rendering ---------------------------------------------------------------

def _num(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def _age(age: timedelta) -> str:
    minutes = int(age.total_seconds() // 60)
    if minutes < 0:
        return "in the future"
    if minutes < 120:
        return f"{minutes}m ago"
    if minutes < 48 * 60:
        return f"{minutes // 60}h {minutes % 60}m ago"
    return f"{minutes // (24 * 60)}d ago"


def _render_timeframe(t: TimeframeState) -> list[str]:
    head = f"{t.timeframe:<4}"
    if not t.built:
        return [f"{head} not built yet (run tb features build)", ""]
    stale = "  STALE" if t.stale else ""
    lines = [f"{head} bar closed {t.bar_close:%Y-%m-%d %H:%M} UTC, "
             f"{_age(t.age)}{stale}"]
    pad = " " * 5
    lines.append(f"{pad}trend {t.trend_regime or 'n/a'} "
                 f"(ADX {_num(t.adx, '.1f')}), structure "
                 f"{t.structure or 'n/a'}, {_duration(t.trend_duration)}")
    lines.append(f"{pad}RSI {_num(t.rsi, '.1f')}, volatility "
                 f"{t.volatility_regime or 'n/a'} "
                 f"(percentile {_num(t.vol_pct, '.0f')})")
    lines.append(f"{pad}{_volume(t)}, distance to EMA 200 "
                 f"{_num(t.dist_ema_200, '+.2%')}")
    lines.append("")
    return lines


def _duration(bars: int | None) -> str:
    if bars is None:
        return "EMA 20 vs EMA 50 n/a"
    if bars == 0:
        return "EMA 20 equals EMA 50"
    side = "above" if bars > 0 else "below"
    return f"EMA 20 {side} EMA 50 for {abs(bars)} bars"


def _volume(t: TimeframeState) -> str:
    if t.volume_ratio is None or t.buy_share is None:
        return "volume n/a"
    return f"volume x{t.volume_ratio:.2f}, buyer share {t.buy_share:.0%}"
