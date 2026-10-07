import json
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from features.pipeline import FEATURE_TIMEFRAMES
from features.summary import STALE_GRACE, state_from_rows

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
NAN = float("nan")


def _row(open_time, **over):
    """A latest_features-shaped row: NaN floats, pd.NA ints, None text."""
    row = {
        "open_time": open_time,
        "trend_regime": "strong_bullish", "adx_14": 31.25,
        "structure": 1, "trend_duration": 14, "rsi_14": 61.34,
        "volatility_regime": "normal", "vol_pct_365d": 54.0,
        "volume_ratio_20": 1.2, "buy_share": 0.523, "dist_ema_200": 0.0145,
    }
    row.update(over)
    return row


def _hour_row(closed_ago, **over):
    """A 1h row whose bar closed `closed_ago` before NOW."""
    return _row(NOW - closed_ago - timedelta(hours=1), **over)


def _by_tf(state):
    return {t.timeframe: t for t in state.timeframes}


def test_state_maps_rows_and_marks_missing_timeframes():
    rows = {"1h": _hour_row(timedelta(minutes=30))}
    state = state_from_rows("BTCUSDT", rows, NOW)

    assert state.symbol == "BTCUSDT"
    assert state.as_of == NOW
    assert [t.timeframe for t in state.timeframes] == list(FEATURE_TIMEFRAMES)
    tfs = _by_tf(state)
    h = tfs["1h"]
    assert h.built is True
    assert h.bar_close == NOW - timedelta(minutes=30)
    assert h.age == timedelta(minutes=30)
    assert h.stale is False
    assert (h.trend_regime, h.adx, h.structure, h.trend_duration) == (
        "strong_bullish", 31.25, "up", 14)
    assert (h.rsi, h.volatility_regime, h.vol_pct) == (61.34, "normal", 54.0)
    assert (h.volume_ratio, h.buy_share, h.dist_ema_200) == (1.2, 0.523, 0.0145)
    for tf in ("5m", "15m", "4h", "1d"):
        t = tfs[tf]
        assert t.built is False
        assert t.stale is False
        assert t.bar_close is None and t.age is None
        assert t.trend_regime is None and t.rsi is None and t.structure is None


def test_a_none_row_and_an_absent_key_are_the_same():
    a = state_from_rows("BTCUSDT", {"1h": None}, NOW)
    b = state_from_rows("BTCUSDT", {}, NOW)
    assert a.to_dict() == b.to_dict()


def test_missing_values_in_a_built_row_become_none():
    row = _hour_row(timedelta(minutes=1), adx_14=NAN, rsi_14=NAN,
                    trend_regime=None, structure=pd.NA,
                    trend_duration=pd.NA, buy_share=NAN)
    h = _by_tf(state_from_rows("BTCUSDT", {"1h": row}, NOW))["1h"]
    assert h.built is True
    assert h.adx is None and h.rsi is None and h.buy_share is None
    assert h.trend_regime is None and h.structure is None
    assert h.trend_duration is None


def test_stale_threshold():
    # 1h: stale when age > 2 h + 5 min
    assert STALE_GRACE == timedelta(minutes=5)
    fresh = state_from_rows("X", {"1h": _hour_row(timedelta(hours=2, minutes=4))}, NOW)
    stale = state_from_rows("X", {"1h": _hour_row(timedelta(hours=2, minutes=6))}, NOW)
    edge = state_from_rows("X", {"1h": _hour_row(timedelta(hours=2, minutes=5))}, NOW)
    assert _by_tf(fresh)["1h"].stale is False
    assert _by_tf(stale)["1h"].stale is True
    assert _by_tf(edge)["1h"].stale is False  # strictly greater than


def test_stale_threshold_scales_with_the_timeframe():
    # 5m: stale when age > 10 min + 5 min = 15 min
    five = lambda ago: _row(NOW - ago - timedelta(minutes=5))
    s = state_from_rows("X", {"5m": five(timedelta(minutes=14))}, NOW)
    t = state_from_rows("X", {"5m": five(timedelta(minutes=16))}, NOW)
    assert _by_tf(s)["5m"].stale is False
    assert _by_tf(t)["5m"].stale is True


def test_agreement_counts():
    def r(regime):
        return _row(NOW - timedelta(days=1), trend_regime=regime)
    rows = {
        "5m": r("weak_bullish"), "15m": r("strong_bullish"),
        "1h": r("strong_bearish"), "4h": r("sideways"), "1d": r(None),
    }
    assert state_from_rows("X", rows, NOW).agreement() == {
        "bullish": 2, "bearish": 1, "sideways": 1}

    rows = {"5m": r("weak_bearish"), "15m": r("strong_bearish"), "1h": None}
    assert state_from_rows("X", rows, NOW).agreement() == {
        "bullish": 0, "bearish": 2, "sideways": 0}


def test_structure_labels():
    def label(v):
        row = _hour_row(timedelta(minutes=1), structure=v)
        return _by_tf(state_from_rows("X", {"1h": row}, NOW))["1h"].structure
    assert label(1) == "up"
    assert label(-1) == "down"
    assert label(0) == "mixed"
    assert label(pd.NA) is None


def test_to_dict_is_json_serialisable():
    rows = {"1h": _hour_row(timedelta(minutes=30)),
            "4h": _row(NOW - timedelta(days=2), adx_14=NAN)}
    d = state_from_rows("BTCUSDT", rows, NOW).to_dict()
    text = json.dumps(d)  # must not raise, and NaN must not leak through
    assert "NaN" not in text
    assert d["symbol"] == "BTCUSDT"
    assert d["as_of"] == NOW.isoformat()
    tfs = {t["timeframe"]: t for t in d["timeframes"]}
    assert list(tfs) == list(FEATURE_TIMEFRAMES)
    assert tfs["1h"]["age"] == 1800.0
    assert tfs["1h"]["bar_close"] == (NOW - timedelta(minutes=30)).isoformat()
    assert tfs["1h"]["structure"] == "up"
    assert tfs["4h"]["adx"] is None
    assert tfs["5m"]["built"] is False and tfs["5m"]["age"] is None
    assert d["agreement"] == {"bullish": 2, "bearish": 0, "sideways": 0}


def test_render_has_every_timeframe_and_no_advice_words():
    rows = {
        "5m": _row(NOW - timedelta(hours=3)),                  # stale
        "15m": _row(NOW - timedelta(minutes=30)),
        "1h": _hour_row(timedelta(minutes=10), trend_regime="strong_bearish",
                        structure=-1, adx_14=NAN),
        "4h": _row(NOW - timedelta(hours=6), structure=0,
                   trend_regime="sideways"),
    }  # 1d has no rows
    text = state_from_rows("BTCUSDT", rows, NOW).render()

    assert "BTCUSDT" in text
    for tf in FEATURE_TIMEFRAMES:
        assert re.search(rf"^\s*{tf}\b", text, re.M), tf
    assert "STALE" in text
    assert text.count("STALE") == 1  # only the 5m frame is stale
    assert "volume x1.20, buyer share 52%" in text
    assert "not built" in text
    assert re.search(r"\b(buy|sell|long|short|enter|exit)\b", text, re.I) is None
    assert "nan" not in text.lower()
