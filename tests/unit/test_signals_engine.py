import math
import re
from datetime import datetime, timedelta, timezone

import pytest

from signals.engine import Inputs, evaluate
from signals.render import render
from signals.settings import RiskSettings

ADVICE = re.compile(r"\b(buy|sell|long|short|enter|exit)\b", re.IGNORECASE)
T = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
MODEL_HINT = "no forecast model (run `tb ml train --target vol3`)"


def ok(**change) -> Inputs:
    base = dict(symbol="BTCUSDT", has_data=True, quality_fail=False,
                bar_open_time=T, age=timedelta(minutes=20), close=60_000.0,
                atr_pct=0.006, p_wild=0.12, model_problem=None)
    return Inputs(**{**base, **change})


def test_all_clear_is_normal_with_the_comparison():
    s = evaluate(ok(), RiskSettings())
    assert s.status == "NORMAL"
    assert s.reasons == ["P(wild) 12.0% < 25%"]


@pytest.mark.parametrize("change, reason", [
    ({"has_data": False}, "no market data"),
    ({"quality_fail": True}, "data quality FAIL"),
    ({"age": timedelta(hours=3, minutes=10)},
     "data STALE (newest 1h bar closed 3h 10m ago)"),
    ({"bar_open_time": None, "age": None, "close": None},
     "no 1h features from the last 3 days (run `tb features build`)"),
    ({"model_problem": "no saved model", "p_wild": None}, MODEL_HINT),
    ({"atr_pct": None}, "volatility measure missing"),
    ({"atr_pct": float("nan")}, "volatility measure missing"),
])
def test_each_gate_alone_forces_wait(change, reason):
    s = evaluate(ok(**change), RiskSettings())
    assert s.status == "WAIT"
    assert s.reasons == [reason]


def test_model_mismatch_reason_names_the_problem():
    s = evaluate(ok(model_problem="built for feature set 3, code is 4", p_wild=None),
                 RiskSettings())
    assert s.reasons == ["no forecast model: built for feature set 3, code is 4 "
                         "(run `tb ml train --target vol3`)"]


def test_several_gates_list_every_reason():
    s = evaluate(ok(quality_fail=True, age=timedelta(hours=5), p_wild=0.5),
                 RiskSettings())
    assert s.status == "WAIT"
    assert s.reasons == ["data quality FAIL",
                         "data STALE (newest 1h bar closed 5h 0m ago)",
                         "wild day expected (P(wild) 50.0% >= 25%)"]


def test_threshold_boundary_is_wait():
    s = evaluate(ok(p_wild=0.25), RiskSettings())
    assert s.status == "WAIT" and s.reasons == ["wild day expected (P(wild) 25.0% >= 25%)"]
    assert evaluate(ok(p_wild=0.2499), RiskSettings()).status == "NORMAL"
    assert evaluate(ok(p_wild=0.31), RiskSettings(wild_threshold=0.35)).status == "NORMAL"


def test_range_and_step():
    s = evaluate(ok(), RiskSettings())
    w = 3 * 0.006 * math.sqrt(24)
    assert s.range_low == pytest.approx(60_000 * (1 - w))
    assert s.range_high == pytest.approx(60_000 * (1 + w))
    assert s.step_price == pytest.approx(60_000 * 2 * w / 10)
    # the smallest step in % is the top one
    assert s.step_pct == pytest.approx(s.step_price / (s.range_high - s.step_price))
    assert s.n_grids == 10


def test_step_too_small_for_two_fees_is_wait():
    # atr 0.0001 -> step about 0.03 %, two fees 0.15 %
    s = evaluate(ok(atr_pct=0.0001), RiskSettings())
    assert s.status == "WAIT"
    assert s.reasons == ["grid step 0.03% cannot pay two fees (0.15%)"]


def test_capital_limit():
    assert evaluate(ok(), RiskSettings()).capital_limit is None
    s = evaluate(ok(), RiskSettings(capital_usdt=2000.0, max_coin_share=0.25))
    assert s.capital_limit == pytest.approx(500.0)


def test_no_build_note_and_no_quality_reason():
    s = evaluate(ok(quality_fail=None), RiskSettings())
    assert s.status == "NORMAL"
    assert "data quality not checked (--no-build)" in s.notes


def test_settings_snapshot_is_recorded():
    st = RiskSettings(profile="return", wild_threshold=0.3)
    assert evaluate(ok(), st).settings == st.to_dict()


def test_render_names_everything_without_advice_words():
    sigs = [evaluate(ok(), RiskSettings(capital_usdt=2000.0)),
            evaluate(ok(symbol="ETHUSDT", p_wild=0.4), RiskSettings(capital_usdt=2000.0)),
            evaluate(ok(symbol="SOLUSDT", has_data=False, bar_open_time=None, age=None,
                        close=None, atr_pct=None, p_wild=None,
                        model_problem="no saved model"), RiskSettings())]
    for profile in ("defensive", "return"):
        out = render(sigs, RiskSettings(profile=profile, capital_usdt=2000.0), now=T)
        assert "BTCUSDT  GRID CONDITIONS NORMAL" in out
        assert "ETHUSDT  WAIT" in out and "wild day expected" in out
        assert "SOLUSDT  WAIT" in out and "no market data" in out
        assert "P(wild) next 24h: 12.0% (WAIT at 25% or more; vol3 xgb_v1)" in out
        assert "capital limit: 500 USDT (25% of 2,000 USDT)" in out
        assert "10 grids" in out
        assert "Not advice." in out
        assert not ADVICE.search(out), ADVICE.search(out)
    assert "spent in USDT" in render(sigs, RiskSettings(), now=T)
    assert "kept the coin" in render(sigs, RiskSettings(profile="return"), now=T)


def test_render_without_capital_shows_the_share():
    out = render([evaluate(ok(), RiskSettings())], RiskSettings(), now=T)
    assert "capital limit: 25% of your capital (set it with `tb risk set --capital`)" in out


def test_shown_numbers_never_contradict_the_comparison():
    s = evaluate(ok(p_wild=0.2496), RiskSettings())
    assert s.status == "NORMAL" and s.reasons == ["P(wild) 24.9% < 25%"]
    s = evaluate(ok(p_wild=0.3), RiskSettings(wild_threshold=0.305))
    assert s.reasons == ["P(wild) 30.0% < 30.5%"]


def test_build_problem_forces_wait_without_the_no_build_note():
    s = evaluate(ok(quality_fail=None, build_problem="error: boom"), RiskSettings())
    assert s.status == "WAIT"
    assert s.reasons == ["feature build failed (error: boom); data quality not "
                         "checked and features may be out of date"]
    assert s.notes == []


def test_error_signal():
    from signals.engine import error_signal
    s = error_signal("SOLUSDT", "broken model", RiskSettings(capital_usdt=100.0))
    assert s.status == "WAIT" and s.reasons == ["could not evaluate: broken model"]
    out = render([s], RiskSettings(), now=T)
    assert "SOLUSDT  WAIT" in out and "could not evaluate: broken model" in out


def test_small_prices_keep_their_digits():
    out = render([evaluate(ok(close=0.0123, atr_pct=0.01), RiskSettings())],
                 RiskSettings(), now=T)
    assert "close 0.012300" in out and "step 0.000" in out and "step 0.00 " not in out
